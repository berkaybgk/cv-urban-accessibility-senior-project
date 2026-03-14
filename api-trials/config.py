"""
Centralised configuration: environment variables and default constants.

Required env vars (set in .env at the project root or api-trials/.env):
    GOOGLE_MAPS_API_KEY
    GCS_BUCKET_NAME
    GCP_PROJECT_ID
"""

import os
import warnings
from dotenv import load_dotenv

# Silence noisy Google Cloud / auth / urllib3 warnings
warnings.filterwarnings("ignore", module=r"google\.auth")
warnings.filterwarnings("ignore", module=r"google\.cloud")
warnings.filterwarnings("ignore", module=r"urllib3")

# Load .env from this directory (api-trials/) or the repo root
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Google Cloud / Maps credentials ─────────────────────────────────────────
GOOGLE_MAPS_API_KEY: str = os.environ.get("GOOGLE_MAPS_API_KEY", "")
GCS_BUCKET_NAME: str = os.environ.get("GCS_BUCKET_NAME", "")
GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID", "")

# ── Street sampling defaults ────────────────────────────────────────────────
DEFAULT_NETWORK_TYPE = "all_public"       # OSM network type for pedestrian/drive/all_public/all/bike etc. 
DEFAULT_SAMPLE_INTERVAL_M = 10      # metres between sample points
DEFAULT_EDGE_INDEX = 0              # 0 = longest edge in the network

# ── Street View API defaults ────────────────────────────────────────────────
SV_SIZE = "640x640"
SV_FOV = 120
SV_PITCH = -30

STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
STREETVIEW_META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

# 4 camera headings per sample point (label, offset from street bearing)
CAMERA_DIRECTIONS: list[tuple[str, int]] = [
    ("forward-right",  60),
    ("forward-left",    300),
]
