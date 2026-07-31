"""Refuse to ship a bundle that is missing pieces.

``wrangler deploy`` uploads ``web/public/`` as the deployment's entire asset
manifest: what is not in the upload is not on the site. So the checkout CI
deploys from *is* what a visitor gets, and a half-committed bundle does not
announce itself -- ``app.js`` requests ``data/corpus.json``, receives a 404,
and renders the empty state. That reads as "no results", not as a broken
deployment.

This is the pre-flight for that, and it runs before the upload rather than
after, because after is too late: the previous deployment's assets are already
gone. It is deliberately *not* ``vsearch verify-web``, which answers a
different question -- whether a live deployment still matches the local FAISS
index. That needs the index, which CI does not have. This asks only whether
the bundle about to replace the live site is internally whole, which is a
question a checkout can answer on its own, with no weights and no index.

Runs on a bare Python 3: no numpy, no project install. Usable locally too::

    python .github/scripts/check_web_bundle.py web/public/data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_BUNDLE = Path("web/public/data")

CORPUS = "corpus.json"
EMBEDDINGS = "embeddings.bin"
EXAMPLES = "examples.json"

FLOAT32_BYTES = 4

# Enough to see the shape of a failure without printing 2,000 lines of it.
_MAX_LISTED = 10


def _reject_constant(token: str) -> object:
    """Fail on the float literals Python emits and no browser accepts.

    ``json.loads`` reads a bare ``NaN``/``Infinity`` happily; ``JSON.parse``
    rejects the *whole document*. The exporter already maps non-finite floats
    to null, so this guards against that guarantee regressing rather than
    against a known bug -- one ungrounded ``year`` would take the entire
    corpus down, not one card.
    """
    msg = f"bare {token} token, which JSON.parse rejects outright"
    raise ValueError(msg)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _listed(items: list[str]) -> str:
    shown = ", ".join(items[:_MAX_LISTED])
    return shown if len(items) <= _MAX_LISTED else f"{shown} ... (+{len(items) - _MAX_LISTED} more)"


def _check_embeddings(bundle: Path, count: int, dim: int) -> list[str]:
    """Size is the only integrity check the binary supports, and it is enough.

    ``embeddings.bin`` has no header and no id column: row order is the sole
    join to ``corpus.json``. A length disagreement therefore does not mean
    "some vectors are missing", it means every label after the discrepancy is
    attached to a different item's vector -- a page that ranks confidently and
    labels wrongly.
    """
    actual = (bundle / EMBEDDINGS).stat().st_size
    expected = count * dim * FLOAT32_BYTES
    if actual != expected:
        return [
            f"{EMBEDDINGS} is {actual} bytes; {CORPUS} declares {count} x {dim} "
            f"({expected} bytes). Re-run `vsearch export-web` -- the two files are "
            f"joined by row order alone, so this bundle mislabels its results."
        ]
    return []


def _check_thumbnails(bundle: Path, items: list[dict[str, Any]]) -> list[str]:
    """Every thumbnail a card points at has to be in the upload.

    A missing one is cosmetic per card -- the grid renders a placeholder -- but
    a *systematically* missing set means the export copied nothing, and that is
    worth failing on rather than deploying a grid of grey boxes.
    """
    missing = [
        str(reference)
        for item in items
        if (reference := item.get("image")) and not (bundle / str(reference)).is_file()
    ]
    if not missing:
        return []
    return [f"{len(missing)} referenced thumbnail(s) absent from the bundle: {_listed(missing)}"]


def _check_examples(bundle: Path, dim: int) -> list[str]:
    """The examples are the page's floor, so an empty set is a failure.

    They carry fp32 query vectors, which is what lets the page answer before
    the 62 MB in-browser encoder has been fetched -- and still answer if it
    cannot be fetched at all. Ship the bundle without them and a visitor on a
    network that blocks huggingface.co gets a search box that does nothing.
    """
    payload = _load(bundle / EXAMPLES)
    examples = payload.get("examples")
    if not examples:
        return [
            f"{EXAMPLES} carries no example vectors, so the page has no offline "
            f"fallback. Re-export with an encoder available."
        ]

    wrong = [
        str(example.get("text")) for example in examples if len(example.get("vector") or []) != dim
    ]
    if wrong:
        return [f"{EXAMPLES} has {len(wrong)} vector(s) that are not {dim}-d: {_listed(wrong)}"]
    return []


def check(bundle: Path) -> list[str]:
    """Return every problem found, so one run reports all of them."""
    absent = [name for name in (CORPUS, EMBEDDINGS, EXAMPLES) if not (bundle / name).is_file()]
    if absent:
        # Nothing downstream is answerable without these, so stop here.
        return [
            f"{bundle} is missing {', '.join(absent)}. Run "
            f"`uv run vsearch export-web --corpus fashion --encoder clip` and commit the result."
        ]

    try:
        corpus = _load(bundle / CORPUS)
    except ValueError as exc:
        return [f"{CORPUS} is not loadable as browser-safe JSON: {exc}"]

    count, dim = int(corpus["count"]), int(corpus["dim"])
    items = corpus["items"]

    problems: list[str] = []
    if len(items) != count:
        problems.append(f"{CORPUS} declares count={count} but carries {len(items)} items")
    problems += _check_embeddings(bundle, count, dim)
    problems += _check_thumbnails(bundle, items)

    try:
        problems += _check_examples(bundle, dim)
    except ValueError as exc:
        problems.append(f"{EXAMPLES} is not loadable as browser-safe JSON: {exc}")

    return problems


def main(argv: list[str]) -> int:
    bundle = Path(argv[1]) if len(argv) > 1 else DEFAULT_BUNDLE
    problems = check(bundle)
    if problems:
        print(f"{bundle} is not deployable:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    corpus = _load(bundle / CORPUS)
    print(
        f"{bundle}: {corpus['count']} x {corpus['dim']} vectors, "
        f"{corpus['model_id']}, thumbnails present. Deployable."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
