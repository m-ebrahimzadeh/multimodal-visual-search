"""Static-bundle export for the Cloudflare Worker demo."""

from vsearch.web.export import DEFAULT_EXAMPLES, WebBundle, export_bundle
from vsearch.web.parity import Parity, measure_parity

__all__ = ["DEFAULT_EXAMPLES", "Parity", "WebBundle", "export_bundle", "measure_parity"]
