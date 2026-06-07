"""
pipeline/counter.py
===================
Step 4 of the pipeline: Object Counting via Connected Component Analysis.

Each spatially connected white region in the refined binary image is
labelled as a distinct object candidate. Per-component statistics
(area, bounding box, centroid, perimeter, circularity, solidity) are
extracted for downstream feature filtering.

Lecture references:
  - L04: Pixel connectivity, connected components, neighborhood
  - L06: Region-based segmentation (connected-component labelling)
"""

import cv2
import numpy as np


class ObjectCounter:
    """Labels connected components and extracts geometric properties."""

    # ------------------------------------------------------------------
    def process(self, refined: np.ndarray) -> tuple[int, list[dict], np.ndarray]:
        """
        Parameters
        ----------
        refined : np.ndarray  (uint8, 0 or 255)
            Morphologically refined binary image.

        Returns
        -------
        raw_count : int
            Number of connected components before feature filtering.
        components : list[dict]
            One dict per component with keys:
              label, area, bbox (x,y,w,h), centroid (cx,cy),
              perimeter, circularity, solidity, contour
        label_map : np.ndarray
            Integer label image (0 = background).
        """
        # cv2.connectedComponentsWithStats returns:
        #   num_labels, label_map, stats, centroids
        num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
            refined, connectivity=8
        )

        components = []

        for label in range(1, num_labels):   # 0 is background
            # Basic stats from connectedComponentsWithStats
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            cx, cy = centroids[label]

            # Extract the isolated component mask for contour analysis
            mask = np.zeros(refined.shape, dtype=np.uint8)
            mask[label_map == label] = 255

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, closed=True)

            # Circularity: 4π·area / perimeter²  (1 = perfect circle)
            circularity = 0.0
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                circularity = min(circularity, 1.0)   # clamp floating-point noise

            # Solidity: area / convex hull area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0

            components.append({
                "label":       label,
                "area":        int(area),
                "bbox":        (int(x), int(y), int(w), int(h)),
                "centroid":    (float(cx), float(cy)),
                "perimeter":   float(perimeter),
                "circularity": float(circularity),
                "solidity":    float(solidity),
                "contour":     contour,
            })

        raw_count = len(components)
        return raw_count, components, label_map
