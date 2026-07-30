"""Evaluation runner.

    python -m vsearch.eval.run_eval --corpus flickr30k --encoder clip

Writes results/metrics.json plus a markdown table for the README.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vsearch.config import get_settings, resolve_device
from vsearch.encoders import load_encoder
from vsearch.eval.protocols import (
    evaluate_image_to_image,
    evaluate_text_to_image,
    flickr_text_queries,
)
from vsearch.eval.retrieval import DEFAULT_KS, RetrievalMetrics, evaluate, to_markdown_table
from vsearch.index import FaissStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalRun:
    """One evaluated configuration."""

    corpus: str
    encoder: str
    model_id: str
    protocol: str
    device: str
    k: int
    metrics: RetrievalMetrics

    @property
    def label(self) -> str:
        return f"{self.encoder} / {self.corpus} / {self.protocol}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "encoder": self.encoder,
            "model_id": self.model_id,
            "protocol": self.protocol,
            "device": self.device,
            "k": self.k,
            "metrics": self.metrics.to_dict(),
        }


def run_text_to_image(
    artifacts_dir: Path,
    corpus: str,
    encoder_name: str,
    *,
    k: int = 10,
    split: str | None = "test",
    max_queries: int | None = None,
    device: str | None = None,
) -> EvalRun:
    """Evaluate text->image retrieval against caption ground truth."""
    store = FaissStore.load(artifacts_dir / f"{corpus}__{encoder_name}" / "index")
    queries = flickr_text_queries(store, split=split, max_queries=max_queries)
    if not queries:
        msg = (
            f"no captioned queries found in {corpus}__{encoder_name}"
            + (f" for split {split!r}" if split else "")
            + ". Was the corpus ingested with its caption column?"
        )
        raise ValueError(msg)

    # allow_fallback=False: silently measuring a different model than the one
    # named would make the reported table wrong.
    encoder = load_encoder(encoder_name, device=device, allow_fallback=False)
    logger.info("Scoring %d captions against %d images", len(queries), len(store))

    metrics = evaluate(evaluate_text_to_image(store, encoder, queries, k=k))
    return EvalRun(
        corpus=corpus,
        encoder=encoder_name,
        model_id=encoder.spec.model_id,
        protocol="text->image",
        device=encoder.device,
        k=k,
        metrics=metrics,
    )


def run_image_to_image(
    artifacts_dir: Path,
    corpus: str,
    encoder_name: str,
    *,
    k: int = 10,
    max_queries: int | None = 1000,
    device: str | None = None,
) -> EvalRun:
    """Evaluate image->image retrieval under the documented label proxy.

    Query vectors are read back out of the index rather than re-encoded: they
    are the same vectors, and reconstructing avoids decoding thousands of
    images again just to reproduce them.
    """
    run_dir = artifacts_dir / f"{corpus}__{encoder_name}"
    store = FaissStore.load(run_dir / "index")

    ids = store.ids()
    if max_queries is not None:
        # Even stride rather than the first N, so the sample is not biased
        # toward whatever the corpus happens to be sorted by.
        stride = max(1, len(ids) // max_queries)
        ids = ids[::stride][:max_queries]

    vectors = store.vectors_for(ids)
    metrics = evaluate(evaluate_image_to_image(store, ids, vectors, k=k))

    spec_model = load_encoder(encoder_name, device=device, allow_fallback=False).spec.model_id
    return EvalRun(
        corpus=corpus,
        encoder=encoder_name,
        model_id=spec_model,
        protocol="image->image (label proxy)",
        device=device or resolve_device(get_settings().device),
        k=k,
        metrics=metrics,
    )


def write_results(runs: list[EvalRun], results_dir: Path, ks: tuple[int, ...] = DEFAULT_KS) -> Path:
    """Persist metrics as JSON and a markdown table."""
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {"runs": [run.to_dict() for run in runs]}
    json_path = results_dir / "metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = {run.label: run.metrics.summary_row(ks) for run in runs}
    table = to_markdown_table(rows, ks)
    (results_dir / "metrics.md").write_text(table + "\n", encoding="utf-8")

    logger.info("Wrote %s and metrics.md", json_path)
    return json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--corpus", default="flickr30k")
    parser.add_argument("--encoder", action="append", dest="encoders")
    parser.add_argument("--protocol", choices=["text", "image"], default="text")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--split", default="test", help="Internal split to restrict to; 'all' for none."
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    encoders = args.encoders or [settings.text_encoder]
    split = None if args.split == "all" else args.split

    runs: list[EvalRun] = []
    for encoder_name in encoders:
        if args.protocol == "text":
            runs.append(
                run_text_to_image(
                    settings.artifacts_dir,
                    args.corpus,
                    encoder_name,
                    k=args.k,
                    split=split,
                    max_queries=args.max_queries,
                    device=args.device,
                )
            )
        else:
            runs.append(
                run_image_to_image(
                    settings.artifacts_dir,
                    args.corpus,
                    encoder_name,
                    k=args.k,
                    max_queries=args.max_queries,
                    device=args.device,
                )
            )

    write_results(runs, settings.results_dir)
    print(to_markdown_table({run.label: run.metrics.summary_row() for run in runs}, DEFAULT_KS))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
