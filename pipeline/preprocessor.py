"""
pipeline/preprocessor.py
========================
Step 1 of the pipeline: Image Preprocessing.

Operations (in order):
  1. Convert to grayscale
  2. Noise reduction  — Gaussian or Median filter
  3. Contrast enhancement — CLAHE or global Histogram Equalisation

Lecture references:
  - L02: Grayscale images, image acquisition
  - L05: Image enhancement, spatial domain, histogram analysis
  - L08: Spatial filtering (Gaussian / Median kernels)
"""

import cv2
import numpy as np
from config import PREPROCESSING


class Preprocessor:
    """Converts a raw BGR image into a clean, contrast-enhanced grayscale image."""

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or PREPROCESSING

    # ------------------------------------------------------------------
    def process(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Parameters
        ----------
        image : np.ndarray
            Raw image loaded by cv2.imread (BGR or grayscale).

        Returns
        -------
        enhanced : np.ndarray
            Preprocessed grayscale image ready for segmentation.
        steps : dict
            Intermediate images keyed by step name (for visualisation).
        """
        steps = {}

        # ── 1. Grayscale conversion ────────────────────────────────────
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        steps["1_grayscale"] = gray.copy()

        # ── 2. Noise reduction ─────────────────────────────────────────
        filtered = self._denoise(gray)
        steps["2_denoised"] = filtered.copy()

        # ── 3. Contrast enhancement ────────────────────────────────────
        enhanced = self._enhance_contrast(filtered)
        steps["3_enhanced"] = enhanced.copy()

        return enhanced, steps

    # ------------------------------------------------------------------
    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        ft = self.cfg["filter_type"]
        if ft == "gaussian":
            return cv2.GaussianBlur(
                gray,
                self.cfg["gaussian_kernel"],
                self.cfg["gaussian_sigma"],
            )
        elif ft == "median":
            return cv2.medianBlur(gray, self.cfg["median_kernel"])
        else:
            raise ValueError(f"Unknown filter_type: '{ft}'. Use 'gaussian' or 'median'.")

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        if self.cfg["use_clahe"]:
            clahe = cv2.createCLAHE(
                clipLimit=self.cfg["clahe_clip_limit"],
                tileGridSize=self.cfg["clahe_tile_grid"],
            )
            return clahe.apply(gray)
        else:
            return cv2.equalizeHist(gray)
