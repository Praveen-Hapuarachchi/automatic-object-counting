"""
utils/io_utils.py
=================
Image I/O helpers: loading, saving, ground-truth CSV reading.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from config import IO


def load_image(path: str) -> np.ndarray:
    """Load an image from disk. Raises FileNotFoundError if missing."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(image: np.ndarray, path: str):
    """Save a BGR image to disk, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def collect_images(directory: str = None) -> list[Path]:
    """Return sorted list of image paths in directory."""
    directory = Path(directory or IO["input_dir"])
    exts = IO["supported_formats"]
    images = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in exts
    )
    return images


def load_ground_truth(csv_path: str) -> dict[str, int]:
    """
    Load a CSV with columns [filename, ground_truth_count].
    Returns {filename_stem: count}.
    """
    try:
        df = pd.read_csv(csv_path)
        return {
            Path(row["filename"]).stem: int(row["ground_truth_count"])
            for _, row in df.iterrows()
        }
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[IO] Warning reading ground truth CSV: {e}")
        return {}


def make_sample_ground_truth(image_paths: list[Path],
                              counts: list[int],
                              save_path: str = "data/ground_truth.csv"):
    """Helper: write a ground-truth CSV from a list of paths and counts."""
    df = pd.DataFrame({
        "filename": [p.name for p in image_paths],
        "ground_truth_count": counts,
    })
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[IO] Ground truth CSV written → {save_path}")
