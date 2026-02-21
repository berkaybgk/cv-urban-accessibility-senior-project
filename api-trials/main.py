import os
import requests
from dotenv import load_dotenv
from gcs_utils import upload_image_from_bytes, upload_image, download_image, list_images

load_dotenv(".env")

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

response = requests.get(f"https://maps.googleapis.com/maps/api/streetview"
                        f"?size=640x640"
                        f"&location=41.0082,28.9784"
                        f"&fov=90"
                        f"&heading=0"
                        f"&pitch=0"
                        f"&key={GOOGLE_MAPS_API_KEY}")

# Save locally and upload to GCS
if response.status_code == 200:
    # Option 1: Save locally first, then upload
    local_path = "street_view_image.jpg"
    with open(local_path, "wb") as f:
        f.write(response.content)
    print(f"Street view image saved locally as '{local_path}'")
    upload_image(local_path, "streetview/41.0082_28.9784.jpg")

    # Option 2: Upload directly from response bytes (no local file needed)
    # upload_image_from_bytes(response.content, "streetview/41.0082_28.9784.jpg")
else:
    print(f"Error fetching street view image: {response.status_code}")




