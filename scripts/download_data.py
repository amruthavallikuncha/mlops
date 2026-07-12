import hashlib

import requests

from heart_risk.config import PROCESSED_DATA_PATH, RAW_DATA_PATH
from heart_risk.data import clean_data, load_raw_data, save_processed_data

SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_DATA_PATH.exists():
        print(f"Downloading UCI data from {SOURCE_URL}")
        response = requests.get(SOURCE_URL, timeout=60)
        response.raise_for_status()
        RAW_DATA_PATH.write_bytes(response.content)
    digest = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
    cleaned = clean_data(load_raw_data(RAW_DATA_PATH))
    save_processed_data(cleaned, PROCESSED_DATA_PATH)
    print(f"Prepared {len(cleaned)} records at {PROCESSED_DATA_PATH}")
    print(f"Raw file SHA-256: {digest}")


if __name__ == "__main__":
    main()
