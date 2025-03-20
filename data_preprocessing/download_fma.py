import os
import requests
from zipfile import ZipFile

FMA_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
DATASET_DIR = "data/raw/"

os.makedirs(DATASET_DIR, exist_ok=True)

zip_path = os.path.join(DATASET_DIR, "fma_small.zip")
if not os.path.exists(zip_path):
    print("Downloading FMA dataset...")
    response = requests.get(FMA_URL, stream=True)
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print("Download complete!")

print("Extracting files...")
with ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(DATASET_DIR)
print("Extraction complete!")
