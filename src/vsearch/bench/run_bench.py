"""Benchmark runner.

    python -m vsearch.bench.run_bench --encoder clip --batch 1 --batch 32

Measures encode latency, throughput, model size, memory and -- critically --
embedding parity against the torch baseline, so a speedup is never reported
without the quality cost that bought it.

Writes results/bench.json and results/bench.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vsearch.bench.export import export_encoder
from vsearch.bench.latency import BenchRow, hardware_label, measure, to_markdown_table
from vsearch.config import get_settings
from vsearch.encoders import load_encoder
from vsearch.encoders.onnx_encoder import OnnxEncoder
from vsearch.encoders.registry import get_spec

logger = logging.getLogger(__name__)

SAMPLE_TEXTS = [
    "red leather ankle boots on a white background",
    "silver wrist watch",
    "a navy blue formal shirt",
    "black leather handbag",
]


def _sample_images(count: int, seed: int = 0) -> list[Image.Image]:
    """Deterministic noise images.

    Content does not affect latency -- the graph runs the same ops regardless
    -- and generating them avoids making the benchmark depend on a corpus
    being downloaded first.
    """
    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)) for _ in range(count)
    ]


@dataclass(frozen=True)
class Parity:
    """How closely a backend reproduces the torch baseline's embeddings."""

    mean_cosine: float
    min_cosine: float

    def to_dict(self) -> dict[str, float]:
        return {
            "parity_mean_cos": round(self.mean_cosine, 5),
            "parity_min_cos": round(self.min_cosine, 5),
        }


def embedding_parity(baseline: np.ndarray, candidate: np.ndarray) -> Parity:
    """Cosine similarity between two backends' embeddings, row by row.

    Both are already L2-normalised by BaseEncoder, so a row-wise dot product
    is the cosine. Parity below ~0.99 means the quantized index and the
    queries no longer agree, and retrieval quality will move with it.
    """
    similarities = np.sum(baseline * candidate, axis=1)
    return Parity(
        mean_cosine=float(np.mean(similarities)),
        min_cosine=float(np.min(similarities)),
    )


def benchmark_encoder(
    name: str,
    *,
    batches: tuple[int, ...] = (1, 32),
    runs: int = 20,
    export_root: Path,
    threads: int | None = None,
    token: str | None = None,
    skip_onnx: bool = False,
) -> list[BenchRow]:
    """Benchmark one encoder across torch and ONNX backends."""
    spec = get_spec(name)
    hardware = hardware_label()
    rows: list[BenchRow] = []

    # allow_fallback=False: a row labelled dinov3 must be dinov3.
    torch_encoder = load_encoder(name, device="cpu", allow_fallback=False, token=token)
    largest = max(batches)
    images = _sample_images(largest)
    baseline_images = torch_encoder.encode_image(images)
    baseline_texts = (
        torch_encoder.encode_text(SAMPLE_TEXTS) if spec.supports_text else np.zeros((0, spec.dim))
    )

    for batch in batches:
        payload = images[:batch]
        rows.append(
            BenchRow(
                encoder=name,
                runtime="pytorch",
                precision="fp32",
                hardware=hardware,
                modality="image",
                stats=measure(
                    lambda payload=payload: torch_encoder.encode_image(payload),  # type: ignore[misc]
                    batch_size=batch,
                    runs=runs,
                ),
                extra={"parity_mean_cos": 1.0, "parity_min_cos": 1.0},
            )
        )

    if spec.supports_text:
        rows.append(
            BenchRow(
                encoder=name,
                runtime="pytorch",
                precision="fp32",
                hardware=hardware,
                modality="text",
                stats=measure(
                    lambda: torch_encoder.encode_text(SAMPLE_TEXTS),
                    batch_size=len(SAMPLE_TEXTS),
                    runs=runs,
                ),
                extra={"parity_mean_cos": 1.0, "parity_min_cos": 1.0},
            )
        )

    if skip_onnx:
        return rows

    for export in export_encoder(name, export_root, quantize=True, token=token):
        onnx_encoder = OnnxEncoder(
            spec, export.model_dir, batch_size=largest, threads=threads, token=token
        )
        parity = embedding_parity(baseline_images, onnx_encoder.encode_image(images))

        for batch in batches:
            payload = images[:batch]
            rows.append(
                BenchRow(
                    encoder=name,
                    runtime="onnxruntime",
                    precision=export.precision,
                    hardware=hardware,
                    modality="image",
                    model_mb=export.image_bytes / (1024 * 1024),
                    stats=measure(
                        # Bound as defaults: both are rebound each iteration,
                        # and a late-binding closure would time the wrong one.
                        lambda payload=payload, enc=onnx_encoder: enc.encode_image(payload),  # type: ignore[misc]
                        batch_size=batch,
                        runs=runs,
                    ),
                    extra=parity.to_dict(),
                )
            )

        if spec.supports_text and baseline_texts.size:
            text_parity = embedding_parity(baseline_texts, onnx_encoder.encode_text(SAMPLE_TEXTS))
            rows.append(
                BenchRow(
                    encoder=name,
                    runtime="onnxruntime",
                    precision=export.precision,
                    hardware=hardware,
                    modality="text",
                    model_mb=export.text_bytes / (1024 * 1024),
                    stats=measure(
                        lambda enc=onnx_encoder: enc.encode_text(SAMPLE_TEXTS),  # type: ignore[misc]
                        batch_size=len(SAMPLE_TEXTS),
                        runs=runs,
                    ),
                    extra=text_parity.to_dict(),
                )
            )

    return rows


def write_results(rows: list[BenchRow], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"hardware": hardware_label(), "rows": [r.to_dict() for r in rows]}
    json_path = results_dir / "bench.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (results_dir / "bench.md").write_text(to_markdown_table(rows) + "\n", encoding="utf-8")
    logger.info("Wrote %s and bench.md", json_path)
    return json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark encoder backends.")
    parser.add_argument("--encoder", action="append", dest="encoders", default=None)
    parser.add_argument("--batch", action="append", dest="batches", type=int, default=None)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    encoders = args.encoders or ["clip"]
    batches = tuple(args.batches or (1, 32))

    rows: list[BenchRow] = []
    for name in encoders:
        logger.info("Benchmarking %s", name)
        rows.extend(
            benchmark_encoder(
                name,
                batches=batches,
                runs=args.runs,
                export_root=settings.artifacts_dir / "onnx",
                threads=args.threads,
                token=settings.token,
                skip_onnx=args.skip_onnx,
            )
        )

    write_results(rows, settings.results_dir)
    print(to_markdown_table(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
