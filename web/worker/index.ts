/**
 * The only thing this Worker does is turn a sentence into a vector.
 *
 * Ranking, filtering and re-ranking all happen in the browser against
 * `embeddings.bin`, which is ~192 KB and cached after first load. That split
 * is deliberate rather than lazy:
 *
 *   - Facet filters and top-k changes need no round trip, so they feel
 *     instant instead of costing a request each.
 *   - The example queries ship with precomputed vectors, so they take the
 *     exact same client-side path and work with the Worker unreachable, the
 *     AI quota spent, or the account gone. A link on a CV degrades to
 *     "examples only" instead of to a 500.
 *   - Worker CPU time stays near zero; the model call is I/O, not compute.
 *
 * The model is `@cf/openai/clip-vit-base-patch32` -- the same checkpoint the
 * index was built from. That matters: the project's own benchmark measured
 * int8 CLIP *text* parity against fp32 at 0.8830, so serving a quantized text
 * tower (as an in-browser runtime would) queries the index in a measurably
 * different space than the one it was built in.
 */

interface Env {
  AI: {
    run(model: string, input: { text: string | string[] }): Promise<unknown>;
  };
  ASSETS: Fetcher;
}

/** Dimension of `openai/clip-vit-base-patch32`, and of `embeddings.bin`. */
const EXPECTED_DIM = 512;

const MODEL = "@cf/openai/clip-vit-base-patch32";

/**
 * Long inputs are truncated by CLIP's 77-token context anyway, so this caps a
 * pointless upload rather than imposing a real limit on the query.
 */
const MAX_QUERY_CHARS = 512;

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      // The bundle is immutable per deploy and the answer for a given query
      // never changes, so let the edge absorb repeat traffic.
      "cache-control": status === 200 ? "public, max-age=3600" : "no-store",
    },
  });

/**
 * Pull the embedding out of whatever shape Workers AI returned.
 *
 * Embedding responses across the catalogue variously look like
 * `{data: [[...]]}`, `{data: [...]}` or a bare array, and the shape is not
 * pinned by the binding's type. Rather than trusting one of them, probe for
 * the first thing that is a flat numeric array of the right length.
 */
function extractVector(response: unknown): number[] {
  const candidates: unknown[] = [response];
  if (response && typeof response === "object") {
    const data = (response as { data?: unknown }).data;
    candidates.push(data);
    if (Array.isArray(data)) candidates.push(data[0]);
  }

  for (const candidate of candidates) {
    if (
      Array.isArray(candidate) &&
      candidate.length === EXPECTED_DIM &&
      typeof candidate[0] === "number"
    ) {
      return candidate as number[];
    }
  }

  // Deliberately loud. A wrong-length vector would still produce a ranked
  // list downstream -- just a meaningless one -- so this must not fall back
  // to zeros or to a reshaped guess.
  const shape = Array.isArray(response)
    ? `array(${response.length})`
    : typeof response === "object" && response !== null
      ? `object(${Object.keys(response).join(",")})`
      : typeof response;
  throw new Error(`expected a ${EXPECTED_DIM}-d embedding from ${MODEL}, got ${shape}`);
}

/**
 * Scale to unit length so the browser's dot product is a cosine.
 *
 * The stored vectors are already unit-norm (the exporter refuses to write
 * them otherwise). Whether Workers AI normalises its output is not documented,
 * and an unnormalised query does not error -- it just tilts every score by a
 * constant factor and stops being a cosine. Cheap to guarantee here.
 */
function normalize(vector: number[]): number[] {
  let sumSquares = 0;
  for (const component of vector) sumSquares += component * component;
  const norm = Math.sqrt(sumSquares);
  if (!Number.isFinite(norm) || norm === 0) {
    throw new Error("model returned a zero or non-finite embedding");
  }
  return vector.map((component) => component / norm);
}

async function embed(request: Request, env: Env): Promise<Response> {
  if (request.method !== "POST") {
    return json({ error: "POST a JSON body of {\"text\": \"...\"}" }, 405);
  }

  let text: unknown;
  try {
    ({ text } = (await request.json()) as { text?: unknown });
  } catch {
    return json({ error: "body is not valid JSON" }, 400);
  }

  if (typeof text !== "string" || text.trim() === "") {
    return json({ error: "field 'text' must be a non-empty string" }, 400);
  }

  try {
    const response = await env.AI.run(MODEL, { text: text.slice(0, MAX_QUERY_CHARS) });
    return json({ model: MODEL, dim: EXPECTED_DIM, vector: normalize(extractVector(response)) });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("embed failed", message);
    // 503, not 500: the page treats this as "free-text is unavailable, the
    // precomputed examples still work" rather than as a broken deploy.
    return json({ error: `embedding unavailable: ${message}` }, 503);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { pathname } = new URL(request.url);

    if (pathname === "/api/embed") return embed(request, env);
    if (pathname === "/api/health") {
      return json({ ok: true, model: MODEL, dim: EXPECTED_DIM });
    }

    // Unreached in practice: static assets are matched before the Worker, so
    // anything arriving here is a path with no asset behind it.
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
