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

export const ENCODER = { library: LIBRARY, model: MODEL, dtype: DTYPE };

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

    // The page is served from this origin, the weights are not. This is already
    // the default in a browser -- the library derives it from the environment --
    // and is pinned here so the assumption is visible at the point it matters.
    env.allowLocalModels = false;

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
      throw new EncoderUnavailable(await explain(env.remoteHost, cause));
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
 * Work out why the weights did not arrive, and say so in one sentence.
 *
 * Runs only after a load has already failed, so the extra requests cost nothing
 * in the normal case. The two modes fail for different reasons, which is the
 * point: `no-cors` resolves to an unreadable opaque response whenever the
 * connection itself works, while `cors` additionally needs the host to permit
 * this origin. So no-cors failing means the host is unreachable, and no-cors
 * succeeding while cors fails means it was reached but something between the
 * two -- a filtering proxy, an extension blocking third-party reads -- would
 * not let the page have the bytes.
 *
 * The probe is the model's own `config.json` rather than the host root: the
 * root is an HTML page that sends no `Access-Control-Allow-Origin`, so a cors
 * probe against it fails on a *healthy* network and would blame one wrongly.
 *
 * Each branch returns what was *observed*, not what is responsible for it. An
 * earlier version blamed "a proxy or an extension blocking cross-origin reads",
 * which the evidence does not support: on the network where this fires, the
 * cross-origin import of the runtime from a CDN succeeds on the same page. Only
 * this host is affected, and from inside the tab there is no way to see what is
 * doing it. Naming a culprit sends the reader to check the wrong thing.
 */
async function explain(remoteHost, cause) {
  const name = host(remoteHost);
  const probe = `${String(remoteHost).replace(/\/$/, "")}/${MODEL}/resolve/main/config.json`;
  const reachable = async (mode) => {
    try {
      await fetch(probe, { mode });
      return true;
    } catch {
      return false;
    }
  };

  if (!(await reachable("no-cors"))) {
    return `the network refused the connection to ${name}`;
  }
  if (!(await reachable("cors"))) {
    return `${name} answered, but this browser would not hand the response to the page`;
  }
  return `${name} served ${MODEL}, but it would not load (${cause.message})`;
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
