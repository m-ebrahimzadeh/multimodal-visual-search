/**
 * Client-side retrieval.
 *
 * The index is a `count x dim` block of unit-norm float32 written straight out
 * of the FAISS artifact, so scoring is a dot product and nothing here needs a
 * library. Three consequences shape the code below:
 *
 *   - A query is embedded once and kept. Changing a facet or the result count
 *     re-ranks from the cached vector, so filters cost nothing beyond a scan.
 *   - Example queries arrive with their vectors already computed, so they skip
 *     the encoder entirely -- which is what makes the page useful in the first
 *     second, before 62 MB of weights have been fetched.
 *   - Those same precomputed vectors are fp32, produced by the PyTorch encoder
 *     that built the index. The in-browser encoder is int8. So the page can
 *     re-encode the example strings and measure the gap between the two,
 *     against ground truth, on the visitor's machine. See `measureParity`.
 */

import { EncoderUnavailable, encode, isResident } from "./encoder.js";

const DATA = "data";
const state = {
  corpus: null,
  /** Float32Array, count * dim, row-major. */
  embeddings: null,
  examples: [],
  /** Cached vector of the last successful query, so filters re-rank locally. */
  queryVector: null,
  queryText: "",
  /**
   * How `queryVector` was produced: "precomputed" for an example chip's fp32
   * vector, "encoded" for one this tab's int8 tower made.
   *
   * On state rather than passed into the renderer, because re-ranks are the
   * case that gets it wrong. Changing a facet re-renders without re-encoding,
   * so any provenance inferred from that call's timings reports "precomputed"
   * for a vector the encoder had just produced -- on a page whose whole claim
   * is that it measures its own quantization error.
   */
  querySource: null,
  /** What the encoder cost, when it was involved. */
  embedMs: 0,
};

const el = (id) => document.getElementById(id);
const FACET_FIELDS = ["articleType", "baseColour", "gender"];

/* -- Loading -------------------------------------------------------------- */

/**
 * Fetch one file of the bundle, failing on the status rather than on the body.
 *
 * A missing file is not a network error and does not reject: the host answers
 * 404 with an HTML page, `r.json()` then chokes on the leading `<`, and the
 * visitor is told `Unexpected token '<'` -- which names the symptom and buries
 * the cause. Checking `ok` first means the message names the file.
 */
async function bundleFile(name, read) {
  const response = await fetch(`${DATA}/${name}`);
  if (!response.ok) throw new Error(`${DATA}/${name} returned HTTP ${response.status}`);
  return read(response);
}

async function load() {
  const [corpus, buffer, examples] = await Promise.all([
    bundleFile("corpus.json", (r) => r.json()),
    bundleFile("embeddings.bin", (r) => r.arrayBuffer()),
    // Examples are a convenience, not a dependency; a bundle exported with
    // --skip-examples has no such file and the page is still usable.
    bundleFile("examples.json", (r) => r.json()).catch(() => ({ examples: [] })),
  ]);

  state.corpus = corpus;
  state.embeddings = new Float32Array(buffer);
  state.examples = examples.examples ?? [];

  const expected = corpus.count * corpus.dim;
  if (state.embeddings.length !== expected) {
    // corpus.json and embeddings.bin are joined by row order alone. If the
    // lengths disagree, every label is attached to the wrong vector, and the
    // page would look like it worked while ranking nonsense.
    throw new Error(
      `embeddings.bin holds ${state.embeddings.length} floats, expected ` +
        `${expected} (${corpus.count} x ${corpus.dim}) — the bundle is inconsistent`
    );
  }

  el("stat-count").textContent = corpus.count.toLocaleString();
  el("stat-dim").textContent = corpus.dim;
  el("stat-bytes").textContent = `${Math.round(buffer.byteLength / 1024)} KB`;

  populateFacets();
  renderExamples();
}

function populateFacets() {
  for (const field of FACET_FIELDS) {
    const select = el(`facet-${field}`);
    if (!select) continue;
    const values = state.corpus.facets[field] ?? [];
    for (const value of values) {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = String(value);
      select.append(option);
    }
    select.disabled = values.length === 0;
  }
}

function renderExamples() {
  const host = el("examples");
  for (const example of state.examples) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = example.text;
    chip.addEventListener("click", () => {
      el("query").value = example.text;
      run(example.text, Float32Array.from(example.vector));
    });
    host.append(chip);
  }
}

/* -- Retrieval ------------------------------------------------------------ */

/** Row indices allowed by the current facet selection. */
function allowedRows() {
  const active = FACET_FIELDS.map((field) => [field, el(`facet-${field}`)?.value]).filter(
    ([, value]) => value
  );

  const rows = [];
  for (let i = 0; i < state.corpus.count; i++) {
    const payload = state.corpus.items[i].payload;
    if (active.every(([field, value]) => String(payload[field]) === value)) rows.push(i);
  }
  return rows;
}

/**
 * Exact top-k by cosine. Unit-norm rows and a unit-norm query make the inner
 * product the cosine directly, so there is no normalisation step here.
 */
function topK(query, rows, k) {
  const { dim } = state.corpus;
  const scored = rows.map((row) => {
    const offset = row * dim;
    let score = 0;
    for (let j = 0; j < dim; j++) score += state.embeddings[offset + j] * query[j];
    return { row, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

/** Embed free text here in the tab, fetching the weights the first time. */
async function embedLocally(text) {
  // A retry after a part-finished download restarts the accounting. The byte
  // counters are cumulative across files, so leaving the abandoned attempt's
  // entries in place inflates the denominator and the bar opens partway along
  // a download that has not begun.
  const downloading = !isResident();
  if (downloading) downloads.clear();

  const [vector] = await encode([text], downloading ? reportProgress : undefined);
  if (vector.length !== state.corpus.dim) {
    // Cannot happen with the pinned model, and is worth catching loudly if the
    // pin is ever changed: a 768-d query silently scored against a 512-d index
    // would produce a ranked list of noise, not an error.
    throw new Error(`query is ${vector.length}-d but the index is ${state.corpus.dim}-d`);
  }
  return vector;
}

async function run(text, precomputed = null) {
  const button = el("submit");
  button.disabled = true;
  notice(null);

  try {
    let vector = precomputed;
    let embedMs = 0;
    if (!vector) {
      const started = performance.now();
      vector = await embedLocally(text);
      embedMs = performance.now() - started;
    }

    state.queryVector = vector;
    state.queryText = text;
    state.querySource = precomputed ? "precomputed" : "encoded";
    state.embedMs = embedMs;
    rankAndRender();
    if (!precomputed) void measureParity();
  } catch (error) {
    notice(describeFailure(text, error));
    markResultsStale(text);
  } finally {
    progress(null);
    button.disabled = false;
  }
}

/**
 * One sentence for a query that could not be answered.
 *
 * Split on the cause, because the two want different words. Only an
 * `EncoderUnavailable` is about the download; a dimension mismatch means the
 * model pin changed, and sending that visitor to check their network points
 * them at the one part that is working.
 *
 * The bare clause arrives from `EncoderUnavailable` and the sentence is built
 * here, in one place. Interpolating a finished sentence into another is what
 * put two em-dashes in one breath in the released version.
 */
function describeFailure(text, error) {
  if (!(error instanceof EncoderUnavailable)) {
    return `Could not answer “${text}”: ${error.message}.`;
  }
  return (
    `Free-text search needs a one-time download of the CLIP text encoder, and this ` +
    `browser or network could not complete it: ${error.message}. ` +
    `The example queries still work — their vectors ship with the page.`
  );
}

/**
 * Say that the grid underneath answers an older question.
 *
 * Without it the page puts an error about the query that just failed directly
 * above a grid captioned with the one before it, and nothing distinguishes
 * them. Nothing is cleared: the previous answer is still a real answer, and
 * throwing it away to prove a point would leave the visitor with less.
 */
function markResultsStale(failed) {
  if (!state.queryVector) return;
  el("summary").textContent =
    `“${failed}” could not be answered — the results below still answer “${state.queryText}”.`;
}

/* -- Self-measurement ----------------------------------------------------- */

/**
 * Re-encode the example strings and compare against their stored fp32 vectors.
 *
 * This is the honest version of a claim the page would otherwise be making
 * silently: that an int8 text tower is interchangeable with the fp32 one the
 * index was built from. This project already measured its own int8 CLIP text
 * export at 0.8830 cosine against that baseline -- the weakest number in its
 * benchmark table -- so "same checkpoint, so same vectors" is exactly the
 * assumption that has already been wrong once here.
 *
 * The comparison is free of network dependencies: both halves are on the page.
 * Runs once, after the first free-text query has already been answered.
 */
let parityMeasured = false;

async function measureParity() {
  if (parityMeasured || state.examples.length === 0) return;
  parityMeasured = true;

  try {
    const texts = state.examples.map((example) => example.text);
    const local = await encode(texts);

    // Both sides are unit-norm, so the inner product is the cosine directly.
    const cosines = local.map((vector, i) => {
      const reference = state.examples[i].vector;
      let sum = 0;
      for (let j = 0; j < vector.length; j++) sum += vector[j] * reference[j];
      return sum;
    });

    const mean = cosines.reduce((a, b) => a + b, 0) / cosines.length;
    const worst = Math.min(...cosines);
    el("parity").textContent =
      `Measured on this machine: mean ${mean.toFixed(4)} cosine against the fp32 vectors ` +
      `the index was built with, worst ${worst.toFixed(4)} ` +
      `(“${texts[cosines.indexOf(worst)]}”), over ${cosines.length} queries.`;
    el("parity").hidden = false;
  } catch {
    // A missing parity line is cosmetic; a broken search is not. The encoder
    // has already answered a real query by the time this runs.
    parityMeasured = false;
  }
}

function rankAndRender() {
  const k = Number(el("topk").value);

  // Two timers, because the page makes two different claims. The header stat
  // is what answering costs: the filter pass, then the ranking. The summary's
  // number is narrower -- the dot-product scan alone -- because that is what
  // the sentence around it says it is. Rendering sits outside both; its cost
  // scales with `k` rather than with the index.
  const startedFilter = performance.now();
  const rows = allowedRows();
  const filterMs = performance.now() - startedFilter;

  const startedScan = performance.now();
  const hits = state.queryVector
    ? topK(state.queryVector, rows, k)
    : // No query yet: show the head of the corpus so the grid is never empty.
      rows.slice(0, k).map((row) => ({ row, score: null }));
  const scanMs = performance.now() - startedScan;

  // Written on every path, browsing included. Confined to the ranked path it
  // stayed an em-dash until the first query succeeded, leaving one header stat
  // looking broken beside three that were populated -- and on a page where the
  // encoder can fail, "until the first success" can mean forever.
  const answerMs = filterMs + scanMs;
  el("stat-latency").textContent = `${answerMs < 1 ? "<1" : Math.round(answerMs)} ms`;
  render(hits);

  if (!state.queryVector) {
    el("summary").textContent =
      `Browsing ${rows.length.toLocaleString()} of ${state.corpus.count.toLocaleString()} images. ` +
      `Search or pick an example to rank them.`;
    return;
  }

  const scope = rows.length === state.corpus.count ? "" : ` within ${rows.length} filtered`;
  // From state, not from an argument. A re-rank does no encoding, so a
  // parameter defaulting to 0 made every filter change report a freshly
  // encoded vector as precomputed.
  const provenance =
    state.querySource === "encoded"
      ? `, ${Math.round(state.embedMs)} ms to embed`
      : ", vector precomputed";
  el("summary").textContent =
    `Top ${hits.length} for “${state.queryText}”${scope} — ` +
    `${scanMs < 1 ? "<1" : scanMs.toFixed(1)} ms to scan ${rows.length.toLocaleString()} vectors${provenance}.`;
}

/* -- Rendering ------------------------------------------------------------ */

function render(hits) {
  const list = el("results");
  list.replaceChildren();

  if (hits.length === 0) {
    // `li`, not `p`. The host is an `ol`, whose only permitted children are
    // `li`, `script` and `template`. A `p` rendered anyway, because `.empty`
    // spans the grid explicitly -- but it is invalid markup, and what a screen
    // reader announces is a list with a stray child inside it.
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No images match those filters.";
    list.append(empty);
    return;
  }

  const template = el("card-template");
  const fragment = document.createDocumentFragment();

  for (const { row, score } of hits) {
    const item = state.corpus.items[row];
    const card = template.content.cloneNode(true);
    const image = card.querySelector("img");

    if (item.image) {
      // Paths in corpus.json are relative to the bundle, not to this page, so
      // the bundle stays movable and only its mount point is known here.
      image.src = `${DATA}/${item.image}`;
      image.alt = item.title;
    } else {
      image.remove();
    }

    card.querySelector(".title").textContent = item.title;
    card.querySelector(".facets").textContent = FACET_FIELDS.map((f) => item.payload[f])
      .filter(Boolean)
      .join(" · ");

    const badge = card.querySelector(".score");
    if (score === null) badge.remove();
    else badge.textContent = score.toFixed(3);

    fragment.append(card);
  }
  list.append(fragment);
}

function notice(message) {
  const host = el("notice");
  host.hidden = message === null;
  host.textContent = message ?? "";
}

/** Show download progress, or hide the bar entirely when passed null. */
function progress(fraction, label = "") {
  const host = el("progress");
  host.hidden = fraction === null;
  if (fraction === null) return;
  el("progress-bar").style.width = `${Math.round(fraction * 100)}%`;
  el("progress-label").textContent = label;
}

/**
 * Aggregate transformers.js per-file progress into one bar.
 *
 * The library reports each file separately and the sizes are wildly uneven --
 * a 62 MB tensor file next to a 2 MB tokenizer -- so a per-file bar would sit
 * at "1 of 4" for the entire wait. Summing bytes tracks the actual delay.
 */
const downloads = new Map();

function reportProgress(event) {
  if (event.status !== "progress" || !event.total) return;
  downloads.set(event.file, { loaded: event.loaded, total: event.total });

  let loaded = 0;
  let total = 0;
  for (const entry of downloads.values()) {
    loaded += entry.loaded;
    total += entry.total;
  }
  const mb = (bytes) => (bytes / 1024 / 1024).toFixed(0);
  progress(loaded / total, `Fetching the CLIP text encoder — ${mb(loaded)} / ${mb(total)} MB`);
}

/* -- Wiring --------------------------------------------------------------- */

el("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = el("query").value.trim();
  if (text) run(text);
});

// Filters and top-k re-rank from the cached vector — no request, no re-embed.
for (const id of ["topk", ...FACET_FIELDS.map((f) => `facet-${f}`)]) {
  el(id)?.addEventListener("change", () => rankAndRender());
}

/** Whatever the markup marks `selected`, so Reset has no second copy of it. */
const DEFAULT_TOPK = el("topk").value;

el("reset").addEventListener("click", () => {
  for (const field of FACET_FIELDS) {
    const select = el(`facet-${field}`);
    if (select) select.value = "";
  }
  el("topk").value = DEFAULT_TOPK;
  el("query").value = "";
  state.queryVector = null;
  state.queryText = "";
  state.querySource = null;
  state.embedMs = 0;
  notice(null);
  rankAndRender();
});

load()
  .then(() => rankAndRender())
  .catch((error) => {
    el("summary").textContent = `Could not load the index: ${error.message}`;
  });
