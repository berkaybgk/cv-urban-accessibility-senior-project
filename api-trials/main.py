import os
import requests
from dotenv import load_dotenv

load_dotenv(".env")

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

response = requests.get(f"https://maps.googleapis.com/maps/api/streetview"
                        f"?size=640x640"
                        f"&location=41.0082,28.9784"
                        f"&fov=90"
                        f"&heading=0"
                        f"&pitch=0"
                        f"&key={GOOGLE_MAPS_API_KEY}")

# Display the image
if response.status_code == 200:
    with open("street_view_image.jpg", "wb") as f:
        f.write(response.content)
    print("Street view image saved as 'street_view_image.jpg'")
else:
    print(f"Error fetching street view image: {response.status_code}")




