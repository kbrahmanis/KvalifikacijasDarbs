
import io
import json
import zipfile
from pathlib import Path

import requests
from PIL import Image


# 1. Configuration
TOTAL_IMAGES = 50
OUTPUT_DIR = Path("photos")

BASE_URL = "http://images.cocodataset.org/test2017/"
METADATA_URL = (
    "http://images.cocodataset.org/annotations/"
    "image_info_test2017.zip"
)


# 2. Get image filenames from COCO metadata
def get_image_filenames():
    response = requests.get(METADATA_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:

        # Find the JSON file regardless of its directory
        json_filename = next(
            name for name in archive.namelist()
            if name.endswith("image_info_test2017.json")
        )

        with archive.open(json_filename) as file:
            metadata = json.load(file)

    return [image["file_name"] for image in metadata["images"]]


# 3. Download and save images
def download_images():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filenames = get_image_filenames()
    downloaded = 0

    for filename in filenames:
        if downloaded >= TOTAL_IMAGES:
            break

        url = BASE_URL + filename

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            image = image.convert("RGB")

            output_path = OUTPUT_DIR / f"image_{downloaded + 1}.jpg"
            image.save(output_path, quality=95)

            downloaded += 1
            print(f"[{downloaded}/{TOTAL_IMAGES}] Saved {output_path}")

        except (requests.RequestException, OSError) as error:
            print(f"Failed: {url} - {error}")

    print(f"\nDownloaded {downloaded} images into '{OUTPUT_DIR}'.")


# 4. Run
if __name__ == "__main__":
    download_images()
