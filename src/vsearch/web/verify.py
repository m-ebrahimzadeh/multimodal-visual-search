"""Check that a deployment ranks by the index the metrics were measured on.

The demo's one substantive claim is that its results come from the same vectors
the README's Recall@k came from. Everything between those two things is a place
that can silently stop being true: a stale ``public/data`` left over from an
older ingest, a half-finished upload, a CDN serving a cached previous version,
a re-export against a different run directory.

None of those fail loudly. The page still renders a ranked grid with plausible
scores; it is just ranking by something else. So this fetches what the
deployment actually serves and scores it against the local FAISS index --
first byte-for-byte, then, because equal bytes are necessary but the ordering
is what users see, by comparing top-k on random probe vectors.

This replaced a check that measured a query encoder hosted on Workers AI. That
encoder does not exist -- Workers AI has no CLIP model -- and the text tower
now runs in the visitor's browser, where the page measures its own parity
against the fp32 vectors in ``examples.json``. That measurement belongs there,
not here: it is a property of the visitor's machine, not of the deployment.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vsearch import __version__
from vsearch.index import FaissStore
from vsearch.web.export import CORPUS_FILENAME, EMBEDDINGS_FILENAME

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 60
_DTYPE = np.dtype("<f4")
_DEFAULT_PROBES = 25
_DATA_PREFIX = "data"


@dataclass(frozen=True)
class Deployment:
    """What a deployment serves, checked against the local index."""

    url: str
    count: int
    dim: int
    ids_match: bool
    max_abs_difference: float
    probes: int
    mismatches: int

    @property
    def identical(self) -> bool:
        """Whether the served vectors are the index's own, to float32 exactness."""
        return self.ids_match and self.max_abs_difference == 0.0

    @property
    def ok(self) -> bool:
        return self.identical and self.mismatches == 0


def _fetch(url: str) -> bytes:
    # The scheme is validated by the caller, so this cannot be redirected to
    # file:// or a custom opener.
    #
    # The user agent is set rather than left at urllib's default because
    # Cloudflare's managed rules answer `Python-urllib/*` with a 403 on
    # reputation alone -- which would report a perfectly healthy deployment as
    # unreachable. This says what the request is, which is also just better
    # manners than a library default in someone's logs.
    request = urllib.request.Request(
        url,
        headers={"accept": "*/*", "user-agent": f"vsearch-verify-web/{__version__}"},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        return bytes(response.read())


def verify_deployment(
    base_url: str,
    run_dir: Path,
    *,
    probes: int = _DEFAULT_PROBES,
    seed: int = 0,
) -> Deployment:
    """Compare a deployment's served bundle against the local FAISS index."""
    if not base_url.startswith(("http://", "https://")):
        msg = f"base_url must be an http(s) URL, got {base_url!r}"
        raise ValueError(msg)

    root = f"{base_url.rstrip('/')}/{_DATA_PREFIX}"
    try:
        corpus = json.loads(_fetch(f"{root}/{CORPUS_FILENAME}"))
        raw = _fetch(f"{root}/{EMBEDDINGS_FILENAME}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        msg = f"could not read the bundle at {root}: {exc}"
        raise RuntimeError(msg) from exc

    count, dim = int(corpus["count"]), int(corpus["dim"])
    expected_bytes = count * dim * _DTYPE.itemsize
    if len(raw) != expected_bytes:
        # Row order is the only join between the two files, so a length
        # disagreement means every label is attached to the wrong vector.
        msg = (
            f"{root}/{EMBEDDINGS_FILENAME} is {len(raw)} bytes but "
            f"{CORPUS_FILENAME} declares {count} x {dim} ({expected_bytes} bytes)"
        )
        raise RuntimeError(msg)

    served = np.frombuffer(raw, dtype=_DTYPE).reshape(count, dim)
    store = FaissStore.load(run_dir / "index")
    ids = store.ids()
    # The same call the exporter makes, so a difference here is a difference
    # introduced after the export -- which is the whole question.
    local = store.vectors_for(ids)

    if local.shape != served.shape:
        msg = (
            f"the deployment serves {served.shape[0]} x {served.shape[1]} vectors but "
            f"{run_dir / 'index'} holds {local.shape[0]} x {local.shape[1]} -- "
            "re-export and redeploy, or point --run-dir at the run that was deployed"
        )
        raise RuntimeError(msg)

    served_ids = [str(item["id"]) for item in corpus["items"]]
    deployment = Deployment(
        url=base_url,
        count=count,
        dim=dim,
        ids_match=served_ids == ids,
        max_abs_difference=float(np.abs(served - local).max()),
        probes=probes,
        mismatches=_count_ranking_mismatches(served, store, probes=probes, seed=seed),
    )

    logger.info(
        "%s: %d x %d, ids %s, max |diff| %.3g, %d/%d probes disagree",
        base_url,
        deployment.count,
        deployment.dim,
        "match" if deployment.ids_match else "DIFFER",
        deployment.max_abs_difference,
        deployment.mismatches,
        deployment.probes,
    )
    return deployment


def _count_ranking_mismatches(
    served: np.ndarray,
    store: FaissStore,
    *,
    probes: int,
    seed: int,
    k: int = 10,
) -> int:
    """Rank random queries both ways and count the orderings that disagree.

    Identical bytes already imply identical rankings, so this is redundant on a
    healthy deployment -- deliberately. It is the check that keeps meaning
    something if the served format ever changes (a different dtype, a transpose,
    a normalisation applied on the way out) in a way that survives a byte
    comparison being loosened.
    """
    rng = np.random.default_rng(seed)
    ids = store.ids()
    mismatches = 0

    for _ in range(probes):
        query = rng.normal(size=(1, served.shape[1])).astype(np.float32)
        query /= np.linalg.norm(query)
        from_bundle = [ids[i] for i in np.argsort(-(served @ query[0]))[:k]]
        from_faiss = [hit.id for hit in store.search(query, k=k)[0]]
        if from_bundle != from_faiss:
            mismatches += 1

    return mismatches


def describe(deployment: Deployment) -> Sequence[tuple[str, str]]:
    """Rows for the CLI table -- claim on the left, what was found on the right."""
    return (
        ("vectors served", f"{deployment.count} x {deployment.dim}"),
        ("ids in index order", "yes" if deployment.ids_match else "NO"),
        ("max |served - index|", f"{deployment.max_abs_difference:.3g}"),
        (
            "ranking probes",
            f"{deployment.probes - deployment.mismatches}/{deployment.probes} agree",
        ),
    )
