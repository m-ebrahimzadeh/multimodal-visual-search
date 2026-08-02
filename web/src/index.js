/**
 * The only thing this deployment executes.
 *
 * Everything a visitor sees is a static asset. Ranking, filtering and the text
 * encoder all run in the tab. This Worker exists for one reason:
 * `onnx/text_model_quantized.onnx` is 61.5 MB and Cloudflare caps an individual
 * static asset at 25 MiB on every plan, so the weights cannot ship in
 * `public/`. They live in R2 instead, and something has to read them out.
 *
 * It is emphatically not an inference endpoint. An earlier revision ran a
 * Worker that called Workers AI for embeddings; that model does not exist, and
 * the Hugging Face Inference API needs a token to store, rotate and budget on a
 * page whose whole job is to still answer in a year unattended. This forwards
 * bytes and decides nothing. No token, no quota, no model in the request path.
 *
 * Why not leave the weights on huggingface.co, where they came from? Because a
 * cross-origin fetch is something a network between the visitor and this page
 * can refuse -- and on at least one corporate network it does, leaving the
 * demo's headline feature dead while the page around it loads normally. Served
 * from this origin there is no second host to block: anything that can reach
 * the demo can reach its encoder.
 */

const PREFIX = "/models/";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Static assets are matched ahead of this Worker, so arriving here at all
    // means nothing in `public/` claimed the path. Hand anything that is not
    // ours back to the asset handler for its own 404.
    if (!url.pathname.startsWith(PREFIX)) return env.ASSETS.fetch(request);

    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method not allowed", { status: 405, headers: { allow: "GET, HEAD" } });
    }

    const key = decodeURIComponent(url.pathname.slice(PREFIX.length));
    if (!key || key.endsWith("/") || key.split("/").includes("..")) {
      return new Response("Not found", { status: 404 });
    }

    // Range and conditional headers are forwarded rather than dropped. This is
    // a 62 MB body: a browser resuming a broken download, or revalidating a
    // cached one, should not have to start over.
    const object = await env.MODELS.get(key, {
      range: request.headers,
      onlyIf: request.headers,
    });

    if (object === null) {
      // Worth naming the key. The page sets `allowRemoteModels = false`, so a
      // gap here is a hard failure rather than a silent fall back to the Hub --
      // which is the point, but it means this 404 is the whole explanation
      // anyone gets.
      return new Response(`No such object: ${key}`, { status: 404 });
    }

    const headers = new Headers();
    object.writeHttpMetadata(headers);
    headers.set("etag", object.httpEtag);
    // A given key's bytes never change: the model is pinned by name, and a new
    // export would be a new key. So a returning visitor re-downloads nothing.
    headers.set("cache-control", "public, max-age=31536000, immutable");

    // `onlyIf` turns a satisfied precondition into a bodyless R2Object. That is
    // a 304, not a 200 that happens to be empty.
    if (!("body" in object) || object.body === null) {
      return new Response(null, { status: 304, headers });
    }

    // Keyed off the *request*, not off `object.range`. R2 fills `range` in
    // either case -- a plain GET comes back with `{offset: 0, length: size}` --
    // so trusting it answered 206 to clients that never asked for a range, with
    // a `Content-Range` spanning the whole body. Unrequested partial content is
    // a protocol violation, and the clients that do notice cache it wrongly.
    if (request.headers.has("range") && object.range) {
      headers.set("content-range", contentRange(object.range, object.size));
      return new Response(object.body, { status: 206, headers });
    }

    headers.set("content-length", String(object.size));
    return new Response(object.body, { status: 200, headers });
  },
};

/**
 * Build a `Content-Range` value from whichever shape R2 parsed.
 *
 * `R2Range` is a union, not a fixed pair: a `Range: bytes=-1024` arrives as
 * `{suffix}`, an open-ended `bytes=5-` as `{offset}` alone. Assuming
 * `{offset, length}` produces a header that is wrong rather than absent, and a
 * wrong `Content-Range` corrupts the assembled file instead of failing.
 */
function contentRange(range, size) {
  if ("suffix" in range && range.suffix !== undefined) {
    return `bytes ${size - range.suffix}-${size - 1}/${size}`;
  }
  const start = range.offset ?? 0;
  const end = range.length === undefined ? size - 1 : start + range.length - 1;
  return `bytes ${start}-${end}/${size}`;
}
