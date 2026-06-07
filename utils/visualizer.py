"""
utils/visualizer.py
===================
Visualisation utilities for the object counting pipeline.

Produces:
  1. Annotated output image — bounding boxes + centroid dots + count label
  2. Pipeline step strip    — side-by-side view of every intermediate image
  3. Evaluation bar chart   — predicted vs ground-truth counts per image

Lecture references:
  - L06: Expected Outcomes — visual identification via bounding boxes / labels
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from pathlib import Path
from config import VISUALISATION


class Visualizer:

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or VISUALISATION

    # ------------------------------------------------------------------
    def annotate(self, image: np.ndarray,
                 accepted: list[dict],
                 rejected: list[dict],
                 final_count: int) -> np.ndarray:
        """
        Draw bounding boxes, centroids and a count banner on the image.

        Green boxes  = accepted (counted) objects
        Red  boxes   = rejected objects (noise / artefacts)
        """
        out = image.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        c = self.cfg

        # Accepted objects — green
        for obj in accepted:
            x, y, w, h = obj["bbox"]
            cx, cy = int(obj["centroid"][0]), int(obj["centroid"][1])
            cv2.rectangle(out, (x, y), (x+w, y+h),
                          c["bbox_color"], c["bbox_thickness"])
            cv2.circle(out, (cx, cy),
                       c["centroid_radius"], c["centroid_color"], -1)
            cv2.putText(out, str(obj["label"]),
                        (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        c["font_scale"], c["font_color"],
                        c["font_thickness"], cv2.LINE_AA)

        # Rejected objects — red (semi-transparent overlay)
        for obj in rejected:
            x, y, w, h = obj["bbox"]
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 180), 1)

        # Count banner at top
        banner = f"Count: {final_count}"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_DUPLEX,
                                      1.0, 2)
        cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (30, 30, 30), -1)
        cv2.putText(out, banner, (10, th + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    def save_pipeline_strip(self, original: np.ndarray,
                            steps: dict,
                            annotated: np.ndarray,
                            save_path: str):
        """
        Save a horizontal strip showing every pipeline step.
        """
        images = {"Original": original}
        images.update(steps)
        images["Final"] = annotated

        n = len(images)
        fig, axes = plt.subplots(2, (n + 1) // 2,
                                 figsize=(4 * ((n + 1) // 2), 8))
        axes = axes.flatten()

        for ax, (title, img) in zip(axes, images.items()):
            if img.ndim == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        # Hide unused axes
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.suptitle("Pipeline Steps", fontsize=11, fontweight="bold")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    def save_evaluation_chart(self, records: list[dict], save_path: str):
        """
        Bar chart comparing predicted vs ground-truth counts.
        """
        evaluated = [r for r in records if r.get("ground_truth") is not None]
        if not evaluated:
            return

        names  = [Path(r["filename"]).stem for r in evaluated]
        pred   = [r["predicted"]    for r in evaluated]
        gt     = [r["ground_truth"] for r in evaluated]

        x = np.arange(len(names))
        w = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
        ax.bar(x - w/2, gt,   w, label="Ground Truth", color="#2196F3", alpha=0.85)
        ax.bar(x + w/2, pred, w, label="Predicted",    color="#4CAF50", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Object Count")
        ax.set_title("Predicted vs Ground Truth Counts")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Evaluation chart saved → {save_path}")
