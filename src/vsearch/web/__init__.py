"""Static-bundle export and deployment verification for the web demo."""

from vsearch.web.export import DEFAULT_EXAMPLES, WebBundle, export_bundle
from vsearch.web.verify import Deployment, describe, verify_deployment

__all__ = [
    "DEFAULT_EXAMPLES",
    "Deployment",
    "WebBundle",
    "describe",
    "export_bundle",
    "verify_deployment",
]
