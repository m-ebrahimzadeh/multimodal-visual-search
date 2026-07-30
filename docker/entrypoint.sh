#!/usr/bin/env bash
# Container entrypoint: fetch a prebuilt index, then serve.
#
# The container never embeds a corpus. Ingestion is a GPU-bound batch job that
# runs elsewhere and publishes its output to the Hub; this process pulls that
# artifact and serves queries from it. A Space that re-embedded 44k images on
# every cold start would take an hour to answer its first request.
set -euo pipefail

PORT="${PORT:-7860}"
ARTIFACT_REPO="${VSEARCH_ARTIFACT_REPO:-}"
ARTIFACTS_DIR="${VSEARCH_ARTIFACTS_DIR:-/app/artifacts}"

if [[ -n "${ARTIFACT_REPO}" ]]; then
  if [[ -f "${ARTIFACTS_DIR}/.pulled" ]]; then
    echo "[entrypoint] Index already present in ${ARTIFACTS_DIR}; skipping pull."
  else
    echo "[entrypoint] Pulling index artifacts from ${ARTIFACT_REPO}..."
    # Non-fatal: the app starts in a degraded state and reports why on
    # /health, which is far easier to diagnose than a crash-looping container.
    if vsearch pull --repo "${ARTIFACT_REPO}" --destination "${ARTIFACTS_DIR}"; then
      touch "${ARTIFACTS_DIR}/.pulled"
      echo "[entrypoint] Index ready."
    else
      echo "[entrypoint] WARNING: pull failed; starting without an index." >&2
    fi
  fi
else
  echo "[entrypoint] VSEARCH_ARTIFACT_REPO unset; serving whatever is in ${ARTIFACTS_DIR}."
fi

echo "[entrypoint] Starting on 0.0.0.0:${PORT}"
# exec so uvicorn becomes PID 1 and receives SIGTERM directly; otherwise the
# shell swallows it and the platform has to wait out the kill timeout.
exec uvicorn vsearch.serve:app --host 0.0.0.0 --port "${PORT}"
