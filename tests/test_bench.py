"""Benchmark harness tests.

Timing itself is not asserted (it is machine-dependent); what is pinned is
the statistics, the parity computation, and the table rendering -- the parts
that decide whether a published number is honest.
"""

from __future__ import annotations

import numpy as np
import pytest

from vsearch.bench import BenchRow, LatencyStats, measure, percentile, to_markdown_table
from vsearch.bench.latency import hardware_label
from vsearch.bench.run_bench import embedding_parity

# --- percentile ------------------------------------------------------------


def test_percentile_returns_an_observed_value() -> None:
    """Nearest-rank, not interpolated: an invented latency never happened."""
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 0.5) in values
    assert percentile(values, 0.95) in values


def test_percentile_orders_correctly() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 1.0) == 5.0


def test_p95_is_at_least_p50() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0.95) >= percentile(values, 0.5)


def test_percentile_handles_empty_input() -> None:
    assert percentile([], 0.5) == 0.0


def test_percentile_handles_single_value() -> None:
    assert percentile([7.0], 0.95) == 7.0


# --- measure ---------------------------------------------------------------


def test_measure_runs_warmup_then_measures() -> None:
    calls = {"n": 0}

    def operation() -> None:
        calls["n"] += 1

    stats = measure(operation, batch_size=4, runs=5, warmup=2)
    assert calls["n"] == 7
    assert stats.runs == 5
    assert stats.batch_size == 4


def test_measure_reports_throughput_per_item() -> None:
    """Items per second, so batch sizes are directly comparable."""
    stats = measure(lambda: None, batch_size=32, runs=3, warmup=0)
    assert stats.throughput_per_s > 0
    assert stats.p95_ms >= stats.p50_ms
    assert stats.min_ms <= stats.mean_ms


def test_latency_stats_serialize_roundly() -> None:
    payload = LatencyStats(
        runs=10,
        batch_size=1,
        p50_ms=12.345,
        p95_ms=20.987,
        mean_ms=13.5,
        min_ms=11.0,
        throughput_per_s=74.07,
        peak_rss_mb=512.4,
    ).to_dict()
    assert payload["p50_ms"] == 12.35
    assert payload["batch_size"] == 1


# --- parity ----------------------------------------------------------------


def test_parity_is_one_for_identical_embeddings() -> None:
    vectors = np.eye(4, dtype=np.float32)
    parity = embedding_parity(vectors, vectors)
    assert parity.mean_cosine == pytest.approx(1.0)
    assert parity.min_cosine == pytest.approx(1.0)


def test_parity_detects_divergence() -> None:
    """A quantized backend that stops agreeing with the index must show up."""
    baseline = np.eye(3, dtype=np.float32)
    candidate = np.roll(baseline, 1, axis=1)
    assert embedding_parity(baseline, candidate).mean_cosine == pytest.approx(0.0)


def test_parity_reports_the_worst_row_not_just_the_mean() -> None:
    """One badly-diverged row is hidden by an average over many good ones."""
    baseline = np.eye(3, dtype=np.float32)
    candidate = baseline.copy()
    candidate[2] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    parity = embedding_parity(baseline, candidate)
    assert parity.mean_cosine == pytest.approx(2 / 3)
    assert parity.min_cosine == pytest.approx(0.0)


# --- table -----------------------------------------------------------------


def _row(runtime: str, precision: str) -> BenchRow:
    return BenchRow(
        encoder="clip",
        runtime=runtime,
        precision=precision,
        hardware="CPU test",
        modality="image",
        model_mb=150.0,
        stats=LatencyStats(
            runs=10,
            batch_size=1,
            p50_ms=10.0,
            p95_ms=15.0,
            mean_ms=11.0,
            min_ms=9.0,
            throughput_per_s=90.0,
            peak_rss_mb=400.0,
        ),
        extra={"parity_mean_cos": 0.999},
    )


def test_table_renders_every_row() -> None:
    table = to_markdown_table([_row("pytorch", "fp32"), _row("onnxruntime", "int8")])
    assert "| encoder |" in table
    assert "pytorch" in table
    assert "onnxruntime" in table
    assert table.count("\n") == 3  # header, separator, two rows


def test_table_handles_no_rows() -> None:
    assert to_markdown_table([]) == "_no benchmark rows_"


def test_row_serialization_keeps_parity_alongside_speed() -> None:
    """A speedup reported without its quality cost is not a result."""
    payload = _row("onnxruntime", "int8").to_dict()
    assert payload["precision"] == "int8"
    assert payload["parity_mean_cos"] == 0.999
    assert "p50_ms" in payload


def test_hardware_label_is_populated() -> None:
    """A latency number without the hardware it ran on is not a result."""
    label = hardware_label()
    assert label
    assert label.startswith(("CPU", "GPU"))
