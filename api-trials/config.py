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
DEFAULT_NETWORK_TYPE = "all_public"
DEFAULT_SAMPLE_INTERVAL_M = 15      # metres between sample points
DEFAULT_EDGE_INDEX = 0              # 0 = longest edge in the network

# ── Zone classification ─────────────────────────────────────────────────────
JUNCTION_ZONE_M = 15.0              # metres from a junction node; the edge
                                    # portion within this distance is NOT
                                    # sampled for midblock — instead it gets
                                    # one deliberate junction_approach point.

# ── Street View API defaults ────────────────────────────────────────────────
SV_SIZE = "640x640"
SV_FOV = 105
SV_PITCH = -30
SV_MAX_SNAP_DISTANCE_M = 25.0      # max metres the actual panorama may be
                                    # from the requested coordinate

STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
STREETVIEW_META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

# ── Zone-based camera presets ───────────────────────────────────────────────
# Each zone type → list of (label, bearing_offset) pairs.
#
# midblock bearings follow the canonical edge direction (west→east),
# so +40° looks ahead-right and −40° looks ahead-left consistently.
#
# junction_approach bearings already point TOWARD the junction, so the
# single camera uses offset 0 — the image looks straight at the crossing.
CAMERA_PRESETS: dict[str, list[tuple[str, int]]] = {
    "midblock": [
        ("forward-right",  40),     # right sidewalk
        ("forward-left",  -40),     # left sidewalk
    ],
    "junction_approach": [
        ("toward-junction", 0),     # single image aimed at the intersection
    ],
}

DEFAULT_CAMERA_PRESET: list[tuple[str, int]] = CAMERA_PRESETS["midblock"]
