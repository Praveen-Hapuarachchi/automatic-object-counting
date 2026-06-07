"""
pipeline/feature_filter.py
===========================
Step 5 of the pipeline: Feature-Based Filtering.

Discards object candidates that fail geometric criteria so that noise,
shadows, and image artefacts do not inflate the count.

Filter criteria (all configurable in config.py):
  • Minimum area        — removes tiny noise blobs
  • Maximum area        — removes huge background artefacts
  • Minimum circularity — removes elongated edge fragments
  • Minimum solidity    — removes crescent / concave noise shapes

Lecture references:
  - L06: Segmentation, feature extraction from regions
  - L07: Morphological refinement context
"""

import numpy as np
from config import FEATURE_FILTER


class FeatureFilter:
    """Filters connected components by geometric properties."""

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or FEATURE_FILTER

    # ------------------------------------------------------------------
    def filter(self, components: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Parameters
        ----------
        components : list[dict]
            Raw component list from ObjectCounter.

        Returns
        -------
        accepted : list[dict]
            Components that pass all criteria — these are counted as objects.
        rejected : list[dict]
            Components that failed — kept for diagnostic visualisation.
        """
        min_area   = self.cfg.get("min_area", 100)
        max_area   = self.cfg.get("max_area", None)
        min_circ   = self.cfg.get("min_circularity", 0.0)
        min_solid  = self.cfg.get("min_solidity", 0.3)

        accepted, rejected = [], []

        for comp in components:
            reason = self._reject_reason(comp, min_area, max_area,
                                         min_circ, min_solid)
            if reason is None:
                accepted.append(comp)
            else:
                comp["reject_reason"] = reason
                rejected.append(comp)

        return accepted, rejected

    # ------------------------------------------------------------------
    def _reject_reason(self, comp, min_area, max_area,
                       min_circ, min_solid) -> str | None:
        if comp["area"] < min_area:
            return f"area {comp['area']} < min {min_area}"
        if max_area is not None and comp["area"] > max_area:
            return f"area {comp['area']} > max {max_area}"
        if comp["circularity"] < min_circ:
            return f"circularity {comp['circularity']:.3f} < min {min_circ}"
        if comp["solidity"] < min_solid:
            return f"solidity {comp['solidity']:.3f} < min {min_solid}"
        return None
