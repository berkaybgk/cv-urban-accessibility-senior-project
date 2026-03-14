import os
import requests
from dotenv import load_dotenv

load_dotenv("../.env")

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

location = "40.960312,29.076770"
headings = [150, 270]  # Two directions to get 2 images
fov = 120
pitch = -30  # Slight negative pitch

for heading in headings:
    response = requests.get(
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size=640x640"
        f"&location={location}"
        f"&fov={fov}"
        f"&heading={heading}"
        f"&pitch={pitch}"
        f"&key={GOOGLE_MAPS_API_KEY}"
    )

    # Display the image
    if response.status_code == 200:
        filename = f"street_view_image_{heading}.jpg"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Street view image saved as '{filename}'")
    else:
        print(f"Error fetching street view image for heading {heading}: {response.status_code}")


#210