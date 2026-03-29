"""
Centralised configuration: environment variables and default constants.

Required env vars (set in .env at the project root or s1-streetview-sampler/.env):
    GOOGLE_MAPS_API_KEY
    GCS_BUCKET_NAME
    GCP_PROJECT_ID
"""

import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", module=r"google\.auth")
warnings.filterwarnings("ignore", module=r"google\.cloud")
warnings.filterwarnings("ignore", module=r"urllib3")

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Google Cloud / Maps credentials ─────────────────────────────────────────
GOOGLE_MAPS_API_KEY: str = os.environ.get("GOOGLE_MAPS_API_KEY", "")
GCS_BUCKET_NAME: str = os.environ.get("GCS_BUCKET_NAME", "")
GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID", "")

# ── Street sampling defaults ────────────────────────────────────────────────
DEFAULT_NETWORK_TYPE = "drive"
DEFAULT_SAMPLE_INTERVAL_M = 30

# ── Street View API defaults ────────────────────────────────────────────────
SV_SIZE = "640x640"
SV_FOV = 90
SV_PITCH = -10

STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
STREETVIEW_META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

# 2 camera headings per sample point (label, offset from street bearing)
CAMERA_DIRECTIONS: list[tuple[str, int]] = [
    ("forward",  0),
    ("backward", 180),
]
