"""Download the UCI Heart Disease dataset (Cleveland subset) and write raw + cleaned CSVs."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import clean, load_raw

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "heart_clean.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "processed.cleveland.data"
    if not target.exists():
        print("Downloading", URL)
        with urllib.request.urlopen(URL, timeout=60) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extract("processed.cleveland.data", RAW_DIR)
    df = clean(load_raw(target))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} shape={df.shape}")


if __name__ == "__main__":
    main()
