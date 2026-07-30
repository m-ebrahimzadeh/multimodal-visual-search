"""Latency, throughput and memory measurement.

Reports p50 and p95 rather than a mean. A mean over a handful of runs is
dominated by whichever iteration happened to collide with a GC pause or a
background process, and it hides the tail that a user actually feels.
"""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LatencyStats:
    """Timing for one measured configuration."""

    runs: int
    batch_size: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    throughput_per_s: float
    peak_rss_mb: float = 0.0
    """Absolute process RSS. Cumulative when several backends are benchmarked
    in one process, so it is *not* attributable to a single backend."""

    rss_delta_mb: float = 0.0
    """Working-set growth during the measurement window. This is the figure
    attributable to running the operation, and the one worth comparing."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "runs": self.runs,
            "batch_size": self.batch_size,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "throughput_per_s": round(self.throughput_per_s, 1),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "rss_delta_mb": round(self.rss_delta_mb, 1),
        }


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Not interpolated: with 20 samples an interpolated p95 invents a value
    between two observations, and for latency it is more honest to report a
    measurement that actually happened.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(fraction * len(ordered) + 0.5)))
    return ordered[rank - 1]


def current_rss_mb() -> float:
    """Resident set size in MB, or 0 when psutil is unavailable."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is in the onnx extra
        return 0.0
    return float(psutil.Process().memory_info().rss) / (1024 * 1024)


def measure(
    operation: Callable[[], Any],
    *,
    batch_size: int,
    runs: int = 20,
    warmup: int = 3,
) -> LatencyStats:
    """Time an operation, discarding warmup iterations.

    Warmup matters more than it looks: the first ONNX Runtime call allocates
    its arenas and picks kernels, and the first torch call may still be
    lazily initialising. Including those makes a fast backend look slow.
    """
    for _ in range(warmup):
        operation()

    # A collection landing mid-measurement shows up as a spurious tail.
    gc.collect()
    baseline_rss = current_rss_mb()
    peak_rss = baseline_rss

    timings: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        operation()
        timings.append((time.perf_counter() - started) * 1000)
        peak_rss = max(peak_rss, current_rss_mb())

    mean_ms = statistics.fmean(timings)
    return LatencyStats(
        runs=runs,
        batch_size=batch_size,
        p50_ms=percentile(timings, 0.50),
        p95_ms=percentile(timings, 0.95),
        mean_ms=mean_ms,
        min_ms=min(timings),
        # Items per second, so batch sizes are directly comparable.
        throughput_per_s=(batch_size / (mean_ms / 1000)) if mean_ms > 0 else 0.0,
        peak_rss_mb=peak_rss,
        rss_delta_mb=max(0.0, peak_rss - baseline_rss),
    )


@dataclass
class BenchRow:
    """One row of the published benchmark table."""

    encoder: str
    runtime: str
    precision: str
    hardware: str
    modality: str
    stats: LatencyStats
    model_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder,
            "runtime": self.runtime,
            "precision": self.precision,
            "hardware": self.hardware,
            "modality": self.modality,
            "model_mb": round(self.model_mb, 1),
            **self.stats.to_dict(),
            **self.extra,
        }


def to_markdown_table(rows: list[BenchRow]) -> str:
    """Render benchmark rows as a markdown table."""
    if not rows:
        return "_no benchmark rows_"

    header = [
        "encoder",
        "runtime",
        "precision",
        # Without this, text rows (4 short strings) sit beside image rows and
        # read as the same measurement at a different batch size.
        "modality",
        "batch",
        "p50 ms",
        "p95 ms",
        "items/s",
        "model MB",
        "ΔRSS MB",
        "parity cos",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        parity = row.extra.get("parity_mean_cos")
        lines.append(
            "| "
            + " | ".join(
                [
                    row.encoder,
                    row.runtime,
                    row.precision,
                    row.modality,
                    str(row.stats.batch_size),
                    f"{row.stats.p50_ms:.1f}",
                    f"{row.stats.p95_ms:.1f}",
                    f"{row.stats.throughput_per_s:.1f}",
                    f"{row.model_mb:.0f}" if row.model_mb else "-",
                    f"{row.stats.rss_delta_mb:.0f}",
                    f"{parity:.4f}" if isinstance(parity, float) else "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def hardware_label() -> str:
    """Short description of the machine, for the benchmark table.

    A latency number without the hardware it was measured on is not a result.
    """
    import platform

    import torch

    if torch.cuda.is_available():
        return f"GPU {torch.cuda.get_device_name(0)}"

    processor = platform.processor() or platform.machine()
    try:
        import psutil

        threads = psutil.cpu_count(logical=True)
    except ImportError:  # pragma: no cover
        threads = None
    suffix = f" ({threads}t)" if threads else ""
    return f"CPU {processor}{suffix}"
