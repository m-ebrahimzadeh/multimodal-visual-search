/**
 * CLIP's text tower, running in the visitor's tab.
 *
 * Nothing in this module executes until someone types free text. The example
 * chips carry their vectors precomputed, so the page answers instantly and
 * this file has not even been fetched yet.
 *
 * Why in the browser rather than on a server? Because every hosted option is a
 * thing that expires. Workers AI has no CLIP model at all -- its catalogue is
 * text-embedding and generation, and a text-only embedding model is useless
 * here: it maps into its own space with no geometric relationship to CLIP's
 * joint image-text one. The Hugging Face Inference API would work, but needs a
 * token that has to be stored, rotated and budgeted. This demo is linked from
 * a CV; it has to still work in a year with nobody maintaining it. So the
 * encoder ships to the client and the deployment keeps no secrets.
 *
 * The price is a one-time ~62 MB download. Most of that is not the transformer
 * -- CLIP's token embedding table is 49,408 x 512, a third of the text tower on
 * its own, and it does not quantise away. The browser caches it afterwards.
 *
 * These are int8 weights, not the fp32 checkpoint the index was built from, so
 * the two halves of the search are no longer bit-identical. That gap is
 * measurable and the page measures it rather than assuming it away.
 */

// Pinned, and `+esm` rather than the package root or the raw dist file. The
// root resolves to the *Node* CJS bundle; the raw `dist/transformers.web.js`
// is unbundled and imports `onnxruntime-web/webgpu` by bare specifier, which
// a browser cannot resolve without an import map. `+esm` is the build with
// those rewritten to CDN URLs.
const LIBRARY = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm";
const MODEL = "Xenova/clip-vit-base-patch32";

// Measured, not chosen by size alone. The fp16 and q4f16 exports of this repo
// fail to build an ONNX Runtime session on the WASM backend (a layer-norm
// fusion refers to a tensor the graph does not contain), and fp32 is 242 MB.
// That leaves the 8-bit family, of which this is the variant that loads.
const DTYPE = "q8";

// Same origin as the page. `web/src/index.js` serves this prefix out of R2,
// and the layout beneath it mirrors the Hub's, so the weights resolve to
// /models/Xenova/clip-vit-base-patch32/onnx/text_model_quantized.onnx.
const MODELS = "/models/";

/**
 * The weights, relative to the model directory.
 *
 * Spelled out rather than derived because `explain` has to probe this exact
 * object, and it is the one most likely to be absent -- 61.5 MB against four
 * files that total 2 MB. transformers.js builds the same name from `DTYPE`
 * (`q8` maps to the `_quantized` suffix) and `text_model` is the tower this
 * page loads, since it never embeds an image. Change `DTYPE` and this moves.
 */
const WEIGHTS = "onnx/text_model_quantized.onnx";

export const ENCODER = { library: LIBRARY, model: MODEL, dtype: DTYPE, models: MODELS };

/**
 * The encoder could not be fetched or started.
 *
 * Distinct from a failure to *run* it, which callers must not report as a
 * download problem: a dimension mismatch is a broken pin, not a blocked
 * network, and telling a visitor to check their connection would send them
 * looking in the wrong place.
 *
 * `message` is a bare clause, lowercase and unpunctuated, so the caller can
 * compose it into a sentence without ending up with two clause joins in one.
 */
export class EncoderUnavailable extends Error {
  constructor(message) {
    super(message);
    this.name = "EncoderUnavailable";
  }
}

/** In-flight or settled load. One per page; the weights are fetched once. */
let pending = null;
/** Set after the first successful encode, so callers can skip the progress UI. */
let resident = false;

/**
 * Fetch the tokenizer and text tower, reporting download progress.
 *
 * Concurrent callers share one load. A *failed* load is forgotten so the next
 * query retries -- the usual cause is a dropped connection mid-download, and
 * making the page permanently broken because of one is the wrong response.
 */
function acquire(onProgress) {
  if (pending) return pending;

  pending = (async () => {
    const library = await import(LIBRARY).catch(() => {
      throw new EncoderUnavailable(`the runtime did not load from ${host(LIBRARY)}`);
    });
    const { AutoTokenizer, CLIPTextModelWithProjection, env } = library;

    // The weights come from this origin now. 61.5 MB in one file is 2.5x what
    // a Cloudflare static asset may be, so they are in R2 with a Worker in
    // front rather than in `public/` -- but from here that is invisible, and
    // this is just a path.
    env.allowLocalModels = true;
    env.localModelPath = MODELS;

    // The line that makes a gap in the bucket visible. Left at its default, a
    // missing object falls back to huggingface.co and quietly succeeds -- for
    // everyone except the visitors this whole change exists for, whose network
    // is precisely what blocks that host. The bug would pass every test that
    // could be run and be live only for the people it was meant to fix.
    env.allowRemoteModels = false;

    try {
      const [tokenizer, model] = await Promise.all([
        AutoTokenizer.from_pretrained(MODEL),
        CLIPTextModelWithProjection.from_pretrained(MODEL, {
          dtype: DTYPE,
          progress_callback: onProgress,
        }),
      ]);
      return { tokenizer, model };
    } catch (cause) {
      // Worth naming the host, and worth saying why. A blocked download does
      // not surface here as "Failed to fetch": the loader carries the missing
      // response a long way before dereferencing it, and the message that comes
      // out is "Cannot read properties of undefined (reading 'tokenizer_class')".
      // That reads like a bug in this page rather than a request the network
      // refused, so `explain` goes and finds out which it was.
      throw new EncoderUnavailable(await explain(cause));
    }
  })();

  pending.catch(() => {
    pending = null;
  });
  return pending;
}

/** Whether the weights are already in memory. */
export function isResident() {
  return resident;
}

/**
 * Embed strings into the index's space. Returns one unit-norm Float32Array per
 * input, in order.
 */
export async function encode(texts, onProgress) {
  const { tokenizer, model } = await acquire(onProgress);
  resident = true;

  const inputs = await tokenizer(texts, { padding: true, truncation: true });
  const { text_embeds: embeds } = await model(inputs);

  const [rows, dim] = embeds.dims;
  return Array.from({ length: rows }, (_, row) =>
    normalize(embeds.data.slice(row * dim, (row + 1) * dim))
  );
}

/** Hostname of a URL, for error messages. Falls back to the raw value. */
function host(url) {
  try {
    return new URL(url).host;
  } catch {
    return String(url);
  }
}

/**
 * Work out why the weights did not arrive, and say so in one clause.
 *
 * Much shorter than it was, and the shrinking is the point. While the weights
 * came from huggingface.co this had to separate "the host is unreachable" from
 * "the host answered but this browser would not hand the page the bytes" --
 * a CORS-layer refusal, invisible to a single request, which took two probes in
 * different modes to tell apart. Serving them from this origin deletes that
 * distinction rather than diagnosing it better: the page itself arrived over
 * this connection, so the connection works, and a failure here is a gap in the
 * bucket that a status code names outright.
 *
 * Runs only after a load has already failed, so the extra requests cost nothing
 * in the normal case.
 */
async function explain(cause) {
  // Weights first: the biggest file and the one an incomplete upload is most
  // likely to have dropped. Probing `config.json` alone would find it present
  // and report the library's own message instead -- which names the missing
  // path inside a sentence about `local_files_only`, and reads like a bug in
  // this page rather than an object that is not in the bucket.
  for (const file of [WEIGHTS, "config.json", "tokenizer.json"]) {
    const status = await probe(`${MODELS}${MODEL}/${file}`);
    if (status === 0) return `this page could not reach its own origin for ${file}`;
    if (status !== 200 && status !== 206) {
      return `this deployment is not serving ${file} (HTTP ${status})`;
    }
  }
  return `${MODEL} was served but would not load (${cause.message})`;
}

/**
 * Status of one model file, or 0 if the request could not be made at all.
 *
 * A single byte: this runs only after a load has already failed, and pulling
 * 62 MB to find out why 62 MB did not arrive would be its own bug. `no-store`
 * because a cached 404 would describe the deployment as it was, not as it is.
 */
async function probe(url) {
  try {
    const response = await fetch(url, {
      cache: "no-store",
      headers: { range: "bytes=0-0" },
    });
    return response.status;
  } catch {
    return 0;
  }
}

/**
 * Scale to unit length.
 *
 * CLIP's projection head does not normalise, but the index does. Skipping this
 * would not reorder a single query's results -- every score scales by the same
 * constant -- so the bug would be invisible in the grid while making the
 * displayed cosines wrong and the parity measurement meaningless.
 */
function normalize(vector) {
  let sum = 0;
  for (const value of vector) sum += value * value;
  const norm = Math.sqrt(sum);
  if (!(norm > 0)) throw new Error("encoder returned a zero vector");
  return vector.map((value) => value / norm);
}
