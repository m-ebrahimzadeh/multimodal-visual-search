#!/usr/bin/env node
/**
 * Move the text encoder's weights from the Hub into this deployment's bucket.
 *
 * A script rather than a paragraph of shell in the README, because the thing
 * most likely to go wrong here is the file *list*. `allowRemoteModels` is off
 * in the browser, so a file missing from the bucket is a hard failure rather
 * than a quiet fall back to huggingface.co -- which is the property that makes
 * the demo work behind a filtering proxy, and also means an incomplete upload
 * breaks free-text search outright. One list, used by both halves.
 *
 * Run once. The model is pinned by name, so the objects never change; a
 * different export would be a different key.
 *
 *   node scripts/models.mjs pull            # Hub  -> ./models
 *   node scripts/models.mjs push --local    # ./models -> local dev storage
 *   node scripts/models.mjs push            # ./models -> the real bucket
 */

import { spawnSync } from "node:child_process";
import { mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Wrangler's entry script, run under this Node rather than through `npx`.
 *
 * Node refuses to spawn a `.cmd` without a shell since the fix for
 * CVE-2024-27980, so `npx wrangler` fails here on Windows in a way that reports
 * no reason. The file is not in the package's `exports`, which stops `import`
 * but not a path handed to a subprocess. Resolved from this script's own
 * location so the cwd does not matter.
 */
const HERE = dirname(fileURLToPath(import.meta.url));
const WRANGLER = join(HERE, "..", "node_modules", "wrangler", "bin", "wrangler.js");

const REPO = "Xenova/clip-vit-base-patch32";
const BUCKET = "multimodal-visual-search-models";
const STAGING = "models";

/**
 * Exactly what `CLIPTextModelWithProjection` and `AutoTokenizer` ask for at
 * dtype q8, and nothing else. The repo also holds a vision tower and seven
 * other quantisations of the text one -- 2.5 GB in total that this demo never
 * touches, since it only ever embeds text.
 */
const FILES = [
  "config.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  // 61.5 MB, and the whole reason for the bucket: a Cloudflare static asset is
  // capped at 25 MiB. Most of it is not the transformer -- CLIP's token
  // embedding table is 49,408 x 512, which does not quantise away.
  "onnx/text_model_quantized.onnx",
];

const MIME = { ".json": "application/json", ".onnx": "application/octet-stream" };

async function pull() {
  for (const file of FILES) {
    const target = join(STAGING, REPO, file);
    await mkdir(dirname(target), { recursive: true });

    const url = `https://huggingface.co/${REPO}/resolve/main/${file}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);

    const bytes = Buffer.from(await response.arrayBuffer());
    await writeFile(target, bytes);
    console.log(`  pulled ${file} (${(bytes.length / 1048576).toFixed(2)} MB)`);
  }
}

async function push(remote) {
  for (const file of FILES) {
    const source = join(STAGING, REPO, file);
    const size = (await stat(source)).size;

    const result = spawnSync(
      process.execPath,
      [
        WRANGLER,
        "r2",
        "object",
        "put",
        `${BUCKET}/${REPO}/${file}`,
        `--file=${source}`,
        `--content-type=${MIME[file.slice(file.lastIndexOf("."))]}`,
        remote ? "--remote" : "--local",
      ],
      { stdio: "inherit", cwd: join(HERE, "..") }
    );
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`upload of ${file} exited ${result.status}`);
    console.log(`  pushed ${file} (${(size / 1048576).toFixed(2)} MB)`);
  }
}

const [command, ...flags] = process.argv.slice(2);
if (command === "pull") {
  await pull();
} else if (command === "push") {
  await push(!flags.includes("--local"));
} else {
  console.error("usage: models.mjs pull | push [--local]");
  process.exit(2);
}
