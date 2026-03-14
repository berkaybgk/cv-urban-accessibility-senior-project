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

# ── Zone classification thresholds ──────────────────────────────────────────
JUNCTION_ZONE_M = 15.0              # metres from a junction node to classify
                                    # a sample as "junction_approach"
CORNER_THRESHOLD_DEG = 30.0         # bearing change (°) within one edge that
                                    # triggers a "bend" classification

# ── Street View API defaults ────────────────────────────────────────────────
SV_SIZE = "640x640"
SV_FOV = 105
SV_PITCH = -30
SV_MAX_SNAP_DISTANCE_M = 25.0      # max metres the returned panorama may be
                                    # from the requested coordinate before we
                                    # consider it unreliable and skip the point

STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
STREETVIEW_META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

# ── Zone-based camera presets ───────────────────────────────────────────────
# Each zone type maps to a list of (label, bearing_offset) pairs.
# Offsets are added to the smoothed street bearing at each sample point.
#
# Edge geometries are canonicalised west→east / south→north, so +90° is
# always the *right* side (south side on a W→E street) and −90° is always
# the *left* side.
CAMERA_PRESETS: dict[str, list[tuple[str, int]]] = {
    "midblock": [
        ("forward-right",  40),     # right sidewalk, looking ahead-right
        ("forward-left",  -40),     # left sidewalk, looking ahead-left
    ],
    "junction_approach": [
        ("crossing-ahead",   0),    # looking straight at the junction
        ("crossing-right",  90),    # right side of the junction
        ("crossing-left",  -90),    # left side of the junction
    ],
    "bend": [
        ("forward-right",  40),     # same as midblock – sidewalks still exist
        ("forward-left",  -40),     # through bends, just with smoothed bearing
    ],
}

# Fallback for any zone not listed above (should not happen)
DEFAULT_CAMERA_PRESET: list[tuple[str, int]] = CAMERA_PRESETS["midblock"]
