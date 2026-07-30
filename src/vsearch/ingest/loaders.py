"""Corpus loaders.

Two corpora, one generic path. Each is described by a ``CorpusSpec`` so that
adding a third is a table entry rather than new code.

* ``flickr30k`` -- the evaluation corpus. Ships five human captions per image
  and a canonical train/val/test assignment, which is what makes reported
  Recall@k comparable to published numbers instead of self-defined.
* ``fashion`` -- the demo corpus. Product photos with eight metadata facets
  that back the search filters.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image


@dataclass(frozen=True)
class ImageRecord:
    """One corpus item: an image, a stable id, and its metadata."""

    id: str
    image: Image
    payload: dict[str, Any]


@dataclass(frozen=True)
class CorpusSpec:
    """Declarative description of a Hub image dataset."""

    name: str
    hf_id: str
    split: str

    revision: str | None = None
    """Hub revision to load from.

    Needed for script-based datasets: `datasets` v3 removed loading-script
    support, so those repos now fail with "Dataset scripts are no longer
    supported". The Hub keeps an auto-converted parquet export on the
    ``refs/convert/parquet`` branch, which loads normally. (The dataset viewer
    reads that same export, which is why such a repo looks fine on the web
    while load_dataset fails.)
    """

    image_column: str = "image"
    id_column: str | None = None
    """Column holding a stable id. ``None`` falls back to the row index."""

    payload_columns: tuple[str, ...] = ()

    text_columns: tuple[str, ...] = ()
    """Payload columns that are free text rather than facets.

    Stored on every hit and shown in the UI, but kept out of the index's
    inverted facet map: they are near-unique per row, so faceting them builds
    one singleton set per record and no filter can usefully target them.
    """

    caption_column: str | None = None
    """Column of ground-truth captions, when the corpus has them."""

    split_column: str | None = None
    """Column carrying a canonical train/val/test assignment, when the whole
    dataset arrives as a single Hub split."""

    description: str = ""


CORPORA: dict[str, CorpusSpec] = {
    "flickr30k": CorpusSpec(
        name="flickr30k",
        hf_id="nlphuji/flickr30k",
        # The entire dataset arrives as one Hub split named "test"; the real
        # train/val/test assignment lives in the `split` column below.
        split="test",
        # nlphuji/flickr30k ships a loading script, which datasets v3 refuses.
        # The auto-converted parquet branch has the same content and loads.
        revision="refs/convert/parquet",
        id_column="img_id",
        payload_columns=("filename", "split"),
        text_columns=("filename",),
        caption_column="caption",
        split_column="split",
        description="31k images with five captions each; canonical retrieval benchmark.",
    ),
    "fashion": CorpusSpec(
        name="fashion",
        hf_id="benitomartin/fashion-product-images-small-384x512",
        split="train",
        id_column="id",
        payload_columns=(
            "productDisplayName",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "gender",
            "season",
            "usage",
            "year",
        ),
        # Near-unique per product: shown on a hit, useless as a facet.
        text_columns=("productDisplayName",),
        description="44k product photos at 384x512 with eight filterable facets.",
    ),
}


def available_corpora() -> list[str]:
    return sorted(CORPORA)


def get_corpus_spec(name: str) -> CorpusSpec:
    try:
        return CORPORA[name]
    except KeyError:
        msg = f"unknown corpus {name!r}; available: {', '.join(available_corpora())}"
        raise KeyError(msg) from None


def _build_payload(spec: CorpusSpec, row: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column in spec.payload_columns:
        if column in row:
            payload[column] = row[column]
    if spec.caption_column and spec.caption_column in row:
        captions = row[spec.caption_column]
        # Flickr30k stores five captions per image; keep them as a list so the
        # facet index skips them (they are free text, not a filterable facet).
        payload["captions"] = list(captions) if isinstance(captions, list) else [captions]
    return payload


def iter_corpus(
    name: str,
    *,
    limit: int | None = None,
    skip: int = 0,
    split_filter: str | None = None,
    token: str | None = None,
    streaming: bool = False,
) -> Iterator[ImageRecord]:
    """Yield records from a corpus.

    ``split_filter`` selects on the corpus's internal split column -- passing
    ``"test"`` for flickr30k yields exactly the canonical 1000-image
    evaluation set.

    ``skip`` advances past the first N records without decoding their images,
    which is what makes resuming an interrupted ingest cheap rather than a
    full re-decode of everything already done.

    ``streaming`` avoids materialising the dataset on disk, which is useful
    for a quick smoke run but slower per image; a full ingest should download.
    """
    from datasets import load_dataset

    spec = get_corpus_spec(name)
    dataset = load_dataset(
        spec.hf_id,
        split=spec.split,
        revision=spec.revision,
        streaming=streaming,
        token=token,
    )

    if split_filter is not None:
        if spec.split_column is None:
            msg = f"corpus {name!r} has no split column to filter on"
            raise ValueError(msg)
        column = spec.split_column
        dataset = dataset.filter(lambda row: row[column] == split_filter)

    if skip:
        # Both paths skip without touching the image column, so no decoding
        # happens for records already ingested.
        if streaming:
            dataset = dataset.skip(skip)
        else:
            total = len(dataset)
            if skip >= total:
                return
            dataset = dataset.select(range(skip, total))

    for position, row in enumerate(dataset):
        if limit is not None and position >= limit:
            break

        raw_id = row[spec.id_column] if spec.id_column else skip + position
        yield ImageRecord(
            id=f"{spec.name}-{raw_id}",
            image=row[spec.image_column],
            payload=_build_payload(spec, row),
        )
