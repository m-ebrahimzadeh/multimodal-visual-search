"""Combined API + UI entry point.

    uvicorn vsearch.serve:app --port 7860

This is what the Docker image and the Hugging Face Space run: one process
serving the Gradio UI at "/" and the JSON API alongside it, with the encoders
loaded exactly once.

For the API alone (no Gradio import, no UI routes), run
``vsearch.api.main:app`` instead.
"""

from __future__ import annotations

from vsearch.api.main import attach_ui

app = attach_ui()

__all__ = ["app"]
