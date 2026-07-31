/**
 * Client-side retrieval.
 *
 * The index is a `count x dim` block of unit-norm float32 written straight out
 * of the FAISS artifact, so scoring is a dot product and nothing here needs a
 * library. Two consequences shape the code below:
 *
 *   - A query is embedded once and kept. Changing a facet or the result count
 *     re-ranks from the cached vector, so filters cost no network round trip.
 *   - Example queries arrive with their vectors already computed, so they run
 *     the identical path with `fetch` skipped entirely. If the Worker is down
 *     or its quota is spent, they still work.
 */

const DATA = "data";
const state = {
  corpus: null,
  /** Float32Array, count * dim, row-major. */
  embeddings: null,
  examples: [],
  /** Cached vector of the last successful query, so filters re-rank locally. */
  queryVector: null,
  queryText: "",
  remoteEmbeddingBroken: false,
};

const el = (id) => document.getElementById(id);
const FACET_FIELDS = ["articleType", "baseColour", "gender"];

/* -- Loading -------------------------------------------------------------- */

async function load() {
  const [corpus, buffer, examples] = await Promise.all([
    fetch(`${DATA}/corpus.json`).then((r) => r.json()),
    fetch(`${DATA}/embeddings.bin`).then((r) => r.arrayBuffer()),
    fetch(`${DATA}/examples.json`)
      .then((r) => r.json())
      // Examples are a convenience, not a dependency; a bundle exported with
      // --skip-examples has no such file and the page is still usable.
      .catch(() => ({ examples: [] })),
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

/** Ask the Worker to embed free text. Returns null when it cannot. */
async function embedRemotely(text) {
  const response = await fetch("/api/embed", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.error ?? `embedding service returned ${response.status}`);
  }
  const { vector, dim } = await response.json();
  if (dim !== state.corpus.dim) {
    throw new Error(`query is ${dim}-d but the index is ${state.corpus.dim}-d`);
  }
  return Float32Array.from(vector);
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
      vector = await embedRemotely(text);
      embedMs = performance.now() - started;
      state.remoteEmbeddingBroken = false;
    }

    state.queryVector = vector;
    state.queryText = text;
    rankAndRender(embedMs);
  } catch (error) {
    state.remoteEmbeddingBroken = true;
    notice(
      `Live text encoding is unavailable (${error.message}). ` +
        `The example queries below still work — they ship with their vectors precomputed.`
    );
  } finally {
    button.disabled = false;
  }
}

function rankAndRender(embedMs = 0) {
  const k = Number(el("topk").value);
  const rows = allowedRows();

  if (!state.queryVector) {
    // No query yet: show the head of the corpus so the grid is never empty.
    render(rows.slice(0, k).map((row) => ({ row, score: null })));
    el("summary").textContent =
      `Browsing ${rows.length.toLocaleString()} of ${state.corpus.count.toLocaleString()} images. ` +
      `Search or pick an example to rank them.`;
    return;
  }

  const started = performance.now();
  const hits = topK(state.queryVector, rows, k);
  const searchMs = performance.now() - started;

  el("stat-latency").textContent = `${searchMs < 1 ? "<1" : Math.round(searchMs)} ms`;
  render(hits);

  const scope = rows.length === state.corpus.count ? "" : ` within ${rows.length} filtered`;
  const remote = embedMs ? `, ${Math.round(embedMs)} ms to embed` : ", vector precomputed";
  el("summary").textContent =
    `Top ${hits.length} for “${state.queryText}”${scope} — ` +
    `${searchMs < 1 ? "<1" : searchMs.toFixed(1)} ms to scan ${rows.length.toLocaleString()} vectors${remote}.`;
}

/* -- Rendering ------------------------------------------------------------ */

function render(hits) {
  const list = el("results");
  list.replaceChildren();

  if (hits.length === 0) {
    const empty = document.createElement("p");
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

el("reset").addEventListener("click", () => {
  for (const field of FACET_FIELDS) {
    const select = el(`facet-${field}`);
    if (select) select.value = "";
  }
  el("query").value = "";
  state.queryVector = null;
  state.queryText = "";
  notice(null);
  rankAndRender();
});

load()
  .then(() => rankAndRender())
  .catch((error) => {
    el("summary").textContent = `Could not load the index: ${error.message}`;
  });
