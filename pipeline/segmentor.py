"""
pipeline/segmentor.py
=====================
Step 2: Image Segmentation (Otsu / Adaptive + optional Watershed).

Auto-inversion: objects (dark buttons) on light background → inverted
so objects become white foreground in the binary mask.

Lecture references:
  - L06: Thresholding (Otsu), similarity-based segmentation
  - L07: Watershed, region-based segmentation
"""

import cv2
import numpy as np
from config import SEGMENTATION, WATERSHED


class Segmentor:
    def __init__(self, seg_cfg=None, ws_cfg=None):
        self.cfg    = seg_cfg or SEGMENTATION
        self.ws_cfg = ws_cfg  or WATERSHED

    def process(self, enhanced):
        steps = {}

        binary = self._threshold(enhanced)

        # Inversion: objects are darker than background in our case
        if self.cfg.get("invert", True):
            binary = cv2.bitwise_not(binary)
        else:
            # Auto-detect: if >70% pixels are white, bg is white → invert
            if np.sum(binary == 255) / binary.size > 0.70:
                binary = cv2.bitwise_not(binary)

        steps["4_threshold"] = binary.copy()

        if self.cfg.get("use_watershed", True):
            binary = self._watershed(enhanced, binary)
            steps["5_watershed"] = binary.copy()

        return binary, steps

    def _threshold(self, gray):
        method = self.cfg.get("method", "otsu")
        if method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.cfg["adaptive_block_size"],
                self.cfg["adaptive_c"])
        else:
            raise ValueError(f"Unknown method: {method}")
        return binary

    def _watershed(self, gray, binary):
        wc = self.ws_cfg
        kernel_bg = np.ones(wc["bg_dilation_kernel"], np.uint8)
        sure_bg   = cv2.dilate(binary, kernel_bg,
                               iterations=wc["bg_dilation_iterations"])

        dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_n = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        thr    = wc["dist_threshold"] * dist_n.max()
        _, sure_fg = cv2.threshold(dist_n, thr, 255, cv2.THRESH_BINARY)
        sure_fg    = sure_fg.astype(np.uint8)

        unknown    = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers    = markers + 1
        markers[unknown == 255] = 0

        color   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(color, markers)

        refined = np.zeros_like(binary)
        refined[markers > 1] = 255
        return refined
