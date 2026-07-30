---
title: Multimodal Visual Search
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Search images by text or by example, with a CPU-only ONNX pipeline.
---

# Multimodal Visual Search

Search a product image collection two ways:

- **By natural language** — "red leather ankle boots on a white background"
- **By example image** — upload a photo and find visually similar items

Text queries run through CLIP's shared text/image embedding space. Image
queries can additionally use DINOv3, a self-supervised vision encoder, or fuse
both rankings with reciprocal rank fusion.

Retrieval is exact inner-product search over L2-normalised embeddings in FAISS,
with pre-filtered facet search (colour, category, gender, season) applied inside
the index rather than by discarding results afterwards.

## How this is deployed

Embedding the corpus is a GPU batch job that runs elsewhere and publishes its
output to a Hub dataset repo. This Space pulls that prebuilt index at startup
and only serves queries — so it cold-starts in seconds on a free CPU box
instead of spending an hour re-embedding tens of thousands of images.

## API

The JSON API is served alongside the UI:

- `POST /search/text` — `{"query": "...", "k": 24, "filters": {...}}`
- `POST /search/image` — multipart image upload
- `GET /health` — loaded encoders, index sizes, device
- `GET /docs` — OpenAPI

Source and benchmarks: see the linked repository.
