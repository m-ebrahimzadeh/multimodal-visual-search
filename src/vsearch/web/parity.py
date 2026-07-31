"""Measure whether the deployed Worker embeds into the index's vector space.

The demo rests on a claim: that Workers AI's ``@cf/openai/clip-vit-base-patch32``
produces text embeddings interchangeable with the local PyTorch fp32 ones the
index was built from. Same checkpoint name, so it should. But "should" is how
the ONNX int8 text tower got assumed equivalent right up until it was measured
at 0.8830 -- the weakest parity number in the benchmark table.

So this measures it. ``examples.json`` already holds locally-encoded vectors
for the example queries; this sends the same strings to the deployment and
reports the cosine between the two. It needs no GPU and no model download,
because the local half of the comparison was computed at export time.

A low number here does not break the demo -- both halves are still CLIP, and
ranking degrades gradually. It does mean the README should say so.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vsearch.web.export import EXAMPLES_FILENAME

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class Parity:
    """Agreement between locally- and remotely-encoded query vectors."""

    queries: int
    cosines: list[float]
    texts: list[str]

    @property
    def mean(self) -> float:
        return float(np.mean(self.cosines)) if self.cosines else float("nan")

    @property
    def worst(self) -> float:
        return float(np.min(self.cosines)) if self.cosines else float("nan")

    def worst_query(self) -> str:
        return self.texts[int(np.argmin(self.cosines))] if self.cosines else ""


def _embed_remotely(base_url: str, text: str) -> np.ndarray:
    # The scheme is checked by the caller, so this cannot be pointed at file://
    # or a custom opener.
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/embed",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        payload: dict[str, Any] = json.loads(response.read())
    return np.asarray(payload["vector"], dtype=np.float32)


def measure_parity(base_url: str, bundle_dir: Path) -> Parity:
    """Compare the deployment's query embeddings with the exported local ones."""
    if not base_url.startswith(("http://", "https://")):
        msg = f"base_url must be an http(s) URL, got {base_url!r}"
        raise ValueError(msg)

    examples = json.loads((bundle_dir / EXAMPLES_FILENAME).read_text(encoding="utf-8"))["examples"]
    if not examples:
        msg = (
            f"{bundle_dir / EXAMPLES_FILENAME} holds no example vectors; "
            "re-export without --skip-examples to have a local side to compare against"
        )
        raise ValueError(msg)

    cosines: list[float] = []
    texts: list[str] = []
    for entry in examples:
        local = np.asarray(entry["vector"], dtype=np.float32)
        try:
            remote = _embed_remotely(base_url, entry["text"])
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            msg = f"could not embed {entry['text']!r} at {base_url}: {exc}"
            raise RuntimeError(msg) from exc

        # Both sides are unit-norm by construction (the exporter asserts it,
        # the Worker normalises), so the inner product is the cosine directly.
        cosines.append(float(np.dot(local, remote)))
        texts.append(entry["text"])

    parity = Parity(queries=len(cosines), cosines=cosines, texts=texts)
    logger.info(
        "parity over %d queries: mean %.4f, worst %.4f (%r)",
        parity.queries,
        parity.mean,
        parity.worst,
        parity.worst_query(),
    )
    return parity
