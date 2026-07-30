# Deployment

The container serves the Gradio UI and the JSON API from one process, and never
embeds a corpus itself: it pulls a prebuilt index published by an ingest run.

## 1. Build and publish an index

Ingest runs wherever there is a GPU (Colab, a cloud box), then publishes:

```bash
vsearch ingest --corpus fashion --encoder clip --device auto
vsearch publish --corpus fashion --encoder clip --repo <user>/vsearch-index
```

Artifacts go to a Hub **dataset** repo — they are data (vectors, metadata,
thumbnails), not weights.

## 2. Run locally with Docker

```bash
docker build -f docker/Dockerfile -t vsearch .
```

```bash
docker run --rm -p 7860:7860 -e VSEARCH_ARTIFACT_REPO=<user>/vsearch-index -e HF_TOKEN=$HF_TOKEN vsearch
```

Then open http://localhost:7860.

Without `VSEARCH_ARTIFACT_REPO` the container still starts; `/health` reports
`degraded` and explains what is missing, rather than crash-looping.

## 3. Deploy to Hugging Face Spaces

Create a Space with **SDK: Docker**, then push this repository to it. The Space
needs `deploy/space/README.md`'s front matter at its repo root:

```bash
cp deploy/space/README.md README.md
```

Set these as Space **secrets** (Settings → Variables and secrets):

| name | required | why |
|---|---|---|
| `HF_TOKEN` | for gated models / private index | DINOv3 is licence-gated; a private artifact repo also needs it |
| `VSEARCH_ARTIFACT_REPO` | yes | which prebuilt index to pull at startup |
| `VSEARCH_DEFAULT_CORPUS` | no | defaults to `fashion` |
| `VSEARCH_TEXT_ENCODER` | no | defaults to `clip` |

### Notes

- The free CPU tier is 2 vCPU. The image is deliberately CPU-only: torch
  resolves from the PyTorch CPU index on Linux, since the default PyPI Linux
  wheel bundles ~2.5 GB of CUDA that a CPU Space can never use.
- Spaces runs the container as uid 1000. `HF_HOME` is set under that user's
  home so the hub cache is writable.
- If `HF_TOKEN` is absent or the DINOv3 licence has not been accepted, the
  encoder registry falls back to the ungated `dinov2-small` and logs it. The
  demo degrades; it does not fail.
