# Multimodal Visual Search Engine

Search a large image collection two ways — by **natural language** ("red leather ankle boots on a
white background") or **by example image** ("find me more like this") — and get back ranked, scored
results in a web UI.

> **Status:** in development. Sections marked _(pending)_ fill in as phases land.

---

## What this is

A production-shaped retrieval system built on vision foundation models, with an explicit focus on the
dimension most CV portfolios skip: **efficiency and deployment**. Every reported speed number is
paired with a retrieval-quality number, so quantization is presented as a trade-off rather than a
free win.

```
        text query ──> CLIP / SigLIP2 text tower ──┐
                                                   ├──> shared embedding space ──> FAISS ──> ranked results
       image query ──> CLIP / DINOv3 image tower ──┘
```

### Design spine: split at the artifact boundary

Embedding a corpus is a GPU-bound one-time batch job. Serving a query is a CPU-bound latency job.
Treating them as one program forces you to rent a GPU for a workload that is 99% idle.

```
Colab GPU: embed ~75k images ──> embeddings + manifest + FAISS index
                                          │
                                  push to HF Hub (versioned dataset repo)
                                          │
                  HF Space (CPU basic) pulls the prebuilt index ──> serves queries
```

The Space never re-embeds a corpus: cold start is seconds, hosting is free, and ingestion is
reproducible rather than something that happened once on a laptop.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Text↔image encoder | `openai/clip-vit-base-patch32` | 512-dim, ungated, fast on CPU |
| Text↔image (quality) | `google/siglip2-base-patch16-224` | Stronger recall, slower — the quality end of the benchmark |
| Image↔image encoder | `facebook/dinov3-vits16-pretrain-lvd1689m` | Self-supervised, 21M params, strong pure-vision similarity |
| Image↔image fallback | `facebook/dinov2-small` | DINOv3 is gated; the registry degrades instead of failing |
| Vector index | FAISS (`IndexFlatIP` / `IndexHNSWFlat`) | Native Windows wheels, no service to run |
| API | FastAPI | `/search/text`, `/search/image` |
| UI | Gradio | Mounted on the same process as the API |
| Packaging | uv + Docker | One-command setup |

DINOv3 has no text tower, so text→image runs on CLIP/SigLIP2 while image→image can use DINOv3, CLIP,
or a weighted fusion. That contrast is itself an evaluation result.

---

## Corpora

| Role | Dataset | Size | Purpose |
|---|---|---|---|
| Evaluation | [`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k) | 31,014 images | Canonical 1000-image test split × 5 captions = 5,000 text queries |
| Demo | [`benitomartin/fashion-product-images-small-384x512`](https://huggingface.co/datasets/benitomartin/fashion-product-images-small-384x512) | 44,072 images @ 384×512 | Product search with real metadata facets |

Flickr30k ships the standard split, so Recall@k here is directly comparable to published CLIP
numbers rather than self-defined. The fashion corpus carries eight metadata facets
(`articleType`, `baseColour`, `gender`, `season`, …) which back the search filters.

---

## Quickstart

```bash
uv sync --all-extras --group dev
uv run vsearch info
```

`vsearch info` reports the resolved compute device and output paths — the first thing worth checking
on a new machine.

Full ingest, serve, and evaluate instructions land with their phases.

---

## Results

_(pending — retrieval metrics table)_

## Benchmarks

_(pending — latency, throughput, index build time, and Recall@k across encoder × runtime × hardware)_

---

## Development

```bash
uv run ruff check .      # lint
uv run mypy src          # types
uv run pytest -m "not slow"   # tests that need no model download
```

Tests requiring model weights or network are marked `slow` and excluded by default.

## Configuration

Copy `.env.example` to `.env`. `HF_TOKEN` is required for the gated DINOv3 encoder and for
publishing index artifacts to the Hub.

## License

MIT
