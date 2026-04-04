import requests
import dotenv

dotenv.load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

LATITUDE, LONGITUDE = 40.968857, 29.071699

PITCH = -30.0
HEADING = 27.0 + 90.0

def fetch_side_image(longitude: float, latitude: float, pitch: float, heading: float):
    url = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={latitude},{longitude}&heading={heading}&fov=90&pitch={pitch}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url)
    print(response.status_code)
    if response.status_code == 200:
        return response.content
    else:
        print(response.text)
        return None

def save_image(image: bytes, path: str) -> None:
    with open(path, "wb") as f:
        f.write(image)

if __name__ == "__main__":
    image = fetch_side_image(LONGITUDE, LATITUDE, PITCH, HEADING)
    if image is not None:
        print("Image fetched successfully")
        save_image(image, "side_image.jpg")
    else:
        print("Failed to fetch image")
