# Live demo — static page on Cloudflare

A search page with no backend. Ranking, facet filtering, top-k **and the query encoder** all run in
the visitor's tab; Cloudflare serves files.

`public/data/embeddings.bin` is the float32 block written straight out of `index.faiss`, so the
deployed demo scores the same bytes the README's metrics were measured on, and changing a filter
costs no network round trip.

## Why the Worker does not run a model

The first version of this ran a Worker whose only job was `POST /api/embed` → Workers AI
`@cf/openai/clip-vit-base-patch32`. That model does not exist. Workers AI's catalogue has no CLIP
model at all, and its text-embedding models cannot stand in: they embed into their own space with no
geometric relationship to CLIP's joint image-text one, and their dimensions (384/768/1024) are not
even 512.

The remaining hosted option, the Hugging Face Inference API, works but costs a token that has to be
stored as a secret and rotated, plus a quota and a third-party dependency on a page whose whole job
is to still work in a year. It also has a failure mode worth naming: CLIP ViT-B/32's `pooler_output`
and `text_embeds` are *both* 512-d, so a wrong-space vector would pass a dimension check silently.

So the text tower ships to the client instead, and nothing is evaluated server-side. There is a
`main` in `wrangler.jsonc` again, but it is not a step back towards any of that:
[`src/index.js`](src/index.js) reads bytes out of R2 and decides nothing. No token, no quota, no
model in the request path.

It exists because of a number. `onnx/text_model_quantized.onnx` is 61.5 MB and a Cloudflare static
asset is capped at **25 MiB on every plan**, so the weights cannot live in `public/` however much
tidier that would be.

## Why the weights are served from here

They used to come straight from `huggingface.co`, which is where they are published and where
transformers.js looks by default. On a corporate network that is a dead demo: the page loads, the
runtime loads from a CDN, and then the one request that matters is refused — the host answers, but
the browser is not handed the response. Everything a visitor came to try is the part that breaks,
and the rest of the page works well enough to make it look like the demo's fault.

That network is not fixable from here, and diagnosing it more precisely does not help someone who
only wanted to type a query. Serving the weights from this origin removes the question instead:
there is no second host to block, so anything that can reach the demo can reach its encoder.

Two consequences worth stating plainly:

- **`allowRemoteModels` is off.** At its default, an object missing from the bucket falls back to
  `huggingface.co` and succeeds — for everyone except the visitors this change exists for, whose
  network is precisely what blocks that host. The bug would pass every test that can be run here and
  be live only for the people it was meant to fix. With it off, a gap fails everywhere, loudly, and
  CI checks for one after each deploy.
- **The weights are the one part of the deploy not reproducible from a checkout.** They are 62 MB in
  a bucket, uploaded once by hand. `wrangler deploy` neither reads nor writes them.

## What that costs

The in-browser tower is `Xenova/clip-vit-base-patch32` at int8 — a ~62 MB one-time download, fetched
only when someone types free text, then cached by the browser. Most of that is not the transformer:
CLIP's token embedding table is 49,408 × 512, a third of the text tower on its own, and it does not
quantise away.

int8 is not a free win, and this project's benchmark table already says so — it measures its own
ONNX int8 CLIP *text* export at 0.8830 parity against fp32. The deployed encoder is a different
export, so the page measures it rather than inheriting that number. Measured on a laptop:

| | |
|---|---|
| mean cosine vs fp32 | **0.9336** |
| worst | 0.8950 |
| top-1 agreement | 5/6 |
| mean top-10 overlap | 7.8/10 |

Two mitigations, both structural rather than cosmetic:

- **The example chips are exact.** Their vectors are fp32, encoded at export time by the same
  PyTorch encoder that built the index. So the showcased results have no quantization error, and
  they answer instantly — before the encoder has been fetched at all.
- **The page reports its own number.** After the first free-text query it re-encodes those same
  example strings and prints the cosine against the stored fp32 vectors, in the footer. The gap is
  measured on the visitor's machine, not asserted from this file.

`fp16` and `q4f16` would be closer, but both fail to create an ONNX Runtime session on the WASM
backend (a layer-norm fusion refers to a tensor the graph does not contain). `fp32` is 242 MB.

## Deploying

Pushing to `main` deploys. The `deploy` job in
[`../.github/workflows/ci.yml`](../.github/workflows/ci.yml) waits for the test suite, uploads
`public/`, and then verifies the result.

It needs two repository secrets, once:

| Secret | Where it comes from |
|---|---|
| `CLOUDFLARE_API_TOKEN` | dash.cloudflare.com → My Profile → API Tokens → **Edit Cloudflare Workers** template, scoped to this account |
| `CLOUDFLARE_ACCOUNT_ID` | dash.cloudflare.com → Workers & Pages → Account ID in the sidebar |

There are no runtime secrets and no AI quota. The token is the only credential in the system, and it
exists because CI uploads rather than because the page needs it.

### The bucket, once

The encoder's weights are not in this repo and CI does not upload them. Create the bucket and fill
it once, from a machine where `wrangler login` has been run:

```bash
cd web
npx wrangler r2 bucket create multimodal-visual-search-models
npm run models:pull     # Hugging Face -> ./models  (gitignored, ~64 MB)
npm run models:push     # ./models     -> the bucket
```

[`scripts/models.mjs`](scripts/models.mjs) holds the file list, and holds it in one place on
purpose: `allowRemoteModels` is off in the browser, so a file missing from the bucket is free-text
search broken rather than a silent fetch from the Hub. The list is exactly what
`CLIPTextModelWithProjection` and `AutoTokenizer` request at `q8` — the repo also carries a vision
tower and seven other quantisations of the text one, about 2.5 GB this demo never touches.

The deploy token needs no R2 permission. The objects never change: the model is pinned by name, so a
different export would be a different key.

### Why the lockfile is committed

The first CI deploy failed, and not on the credentials. `cloudflare/wrangler-action` probes
`npx --no-install wrangler --version` and installs Wrangler itself when that probe fails; with no
lockfile there was nothing to find, so it took its own install path and that exited 1.

`npm ci` before the action makes the probe succeed, so the action adopts the Wrangler already there
and never installs anything. That needs `package-lock.json` in the repo, which is worth having
regardless: `package.json` asks for `^4.0.0`, so without a lockfile every deploy silently resolves
whatever 4.x is newest that day. The job also pins Node 22 — Wrangler declares `node >=22` and the
runner image is mid-migration from 20 to 24, so leaving it to the image makes the deploy depend on
something no commit here controls.

### Why the bundle is committed

`public/data/` is generated by `vsearch export-web`, and it used to be gitignored — it copies
third-party dataset thumbnails, and a working-tree deploy did not need them in the repo.
Push-to-deploy changes that calculation. `wrangler deploy` uploads `public/` as the deployment's
*entire* asset manifest, so files missing from the checkout are missing from the site. An ignored
bundle would mean every push deployed a page whose data 404s — and that failure is quiet, because
`app.js` renders the empty state rather than an error.

So the bundle is tracked, `ATTRIBUTION.md` ships next to the images, and re-exporting becomes a
commit:

```bash
uv run vsearch export-web --corpus fashion --encoder clip
git add public/data && git commit -m "chore(web): re-export the bundle"
```

Before uploading anything, CI runs
[`check_web_bundle.py`](../.github/scripts/check_web_bundle.py) over the checkout: that
`embeddings.bin` is exactly `count × dim × 4` bytes, that every thumbnail a card references is
present, that `examples.json` still carries its fp32 vectors, and that no `NaN` slipped into
`corpus.json`. It runs before the upload because afterwards is too late — the bundle it would have
replaced is already gone. Run it yourself before committing an export:

```bash
python ../.github/scripts/check_web_bundle.py public/data
```

After the upload, CI fetches `data/corpus.json` and `data/embeddings.bin` back from the edge and
compares SHA-256 against the committed files. `wrangler deploy` exiting 0 says the upload was
accepted, not that those bytes are what a visitor gets.

### Deploying by hand

Still supported, and the only path that runs the stronger check:

```bash
npx wrangler login && npx wrangler deploy
```

## Checking it works

```bash
uv run vsearch verify-web https://<your-deployment>.workers.dev
```

This fetches what the deployment actually serves and scores it against the local FAISS index —
byte-for-byte, then by ranking random probe vectors both ways. The failure it exists to catch is a
stale or half-finished upload, which does not announce itself: the page still renders a plausible
ranked grid, just of the wrong results.

CI cannot run this: a checkout has no `artifacts/`, so there is no index to compare against. Its
post-deploy check compares the edge against the *committed* bundle instead, which catches a stale
deployment but would pass a bundle exported from the wrong run directory. That gap is why this
command stays a manual step after re-exporting.

The other claim — that the in-browser encoder shares the index's vector space — is measured in the
page itself, because it is a property of the visitor's browser rather than of the deployment.

## Local development

`wrangler dev` serves the page and the weights together, which is the only way to exercise the whole
thing:

```bash
cd web && npm install && npm run models:pull && npm run models:push:local && npm run dev
```

`models:push:local` puts the files in Wrangler's local R2 emulation rather than the real bucket.
Both model steps are one-time.

A plain static server still renders the page, browses the corpus and answers the example chips:

```bash
python -m http.server 8788 --directory public
```

Free text will not work there, and it should not: nothing serves `/models/`, so the encoder 404s.
That is worth seeing on purpose — it is exactly what a deployment with an empty bucket does, and the
notice it produces is the one a visitor would get.

The notice names the cause, because the library does not. A failed load reaches the loader as a null
dereference — `Cannot read properties of undefined (reading 'tokenizer_class')` — which reads like a
bug in this page rather than a file that did not arrive. So on failure the encoder re-requests the
model's `config.json` and reports the status.

That check used to be much larger. While the weights came from `huggingface.co` it had to separate
"unreachable" from "reached, but this browser would not hand the page the bytes" — a CORS-layer
refusal that a single request cannot distinguish, needing two probes in different modes. Same-origin
there is no such layer, and the reasoning collapsed into a status code.

## Layout

```
wrangler.jsonc      assets + an R2 binding; see "Why the Worker does not run a model"
package-lock.json   pins Wrangler for CI; see "Why the lockfile is committed"
src/index.js        the Worker: reads /models/* out of R2, delegates everything else
scripts/models.mjs  one-time, by hand: the Hub -> ./models -> the bucket
public/index.html   markup
public/app.js       loading, ranking, filtering, rendering, parity measurement
public/encoder.js   lazily-loaded CLIP text tower, fetched from this origin
public/styles.css   light/dark, follows the visitor's system theme
public/data/        generated by `vsearch export-web` -- committed, so CI can deploy it
models/             gitignored staging for the weights; never deployed, never committed
```

Outside this directory:

```
../.github/workflows/ci.yml            the `deploy` job: push to main -> live
../.github/scripts/check_web_bundle.py pre-flight, run before anything is uploaded
```

## Attribution

`public/data/images/` holds thumbnails redistributed from the source dataset so results can be
rendered; `vsearch export-web` writes an `ATTRIBUTION.md` beside them naming it. They belong to
their original creators.
