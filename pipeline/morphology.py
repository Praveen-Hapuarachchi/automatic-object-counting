"""
pipeline/morphology.py
======================
Step 3 of the pipeline: Morphological Image Processing.

Applied in order (matching proposal Table 1):
  1. Erosion  (3×3, 1 iter)  — shrinks blobs, removes tiny noise pixels
  2. Dilation (3×3, 1 iter)  — restores object size after erosion
  3. Opening  (5×5, 2 iter)  — erosion then dilation; removes salt noise
  4. Closing  (5×5, 2 iter)  — dilation then erosion; fills small holes

Lecture references:
  - L07: Morphological Image Processing
         Dilation | Erosion | Opening | Closing (binary & grayscale)
"""

import cv2
import numpy as np
from config import MORPHOLOGY


class MorphologyProcessor:
    """Refines a binary segmented image using morphological operations."""

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or MORPHOLOGY

    # ------------------------------------------------------------------
    def process(self, binary: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Parameters
        ----------
        binary : np.ndarray  (uint8, 0 or 255)
            Binary image from Segmentor.

        Returns
        -------
        refined : np.ndarray
            Morphologically refined binary image.
        steps : dict
            Intermediate images after each operation.
        """
        steps = {}
        img = binary.copy()

        # ── Erosion ────────────────────────────────────────────────────
        img = self._apply(img, cv2.MORPH_ERODE, "erosion")
        steps["6_eroded"] = img.copy()

        # ── Dilation ───────────────────────────────────────────────────
        img = self._apply(img, cv2.MORPH_DILATE, "dilation")
        steps["7_dilated"] = img.copy()

        # ── Opening (removes small noise) ──────────────────────────────
        img = self._apply(img, cv2.MORPH_OPEN, "opening")
        steps["8_opened"] = img.copy()

        # ── Closing (fills holes, smooths contours) ────────────────────
        img = self._apply(img, cv2.MORPH_CLOSE, "closing")
        steps["9_closed"] = img.copy()

        return img, steps

    # ------------------------------------------------------------------
    def _apply(self, img: np.ndarray,
               morph_type: int,
               key: str) -> np.ndarray:
        cfg = self.cfg[key]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,       # elliptical SE fits round objects best
            cfg["kernel"]
        )
        return cv2.morphologyEx(
            img, morph_type, kernel,
            iterations=cfg["iterations"]
        )
