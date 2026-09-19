"""
AeroCortex mission dashboard.

Streamlit has been replaced by a Minimalist Monochrome SPA hosted by
`dashboard.server` (FastAPI + static files). Launch with:

    python main.py --dashboard
"""

from dashboard.server import app

__all__ = ["app"]
