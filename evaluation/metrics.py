"""
evaluation/metrics.py
=====================
Performance evaluation for the object counting system.

Metrics computed:
  • Per-image:
      - predicted count, ground-truth count, absolute error, relative error (%)
      - correct / incorrect flag

  • Aggregate (over full dataset):
      - Counting Accuracy (%)  — fraction of images with exact count match
      - Mean Absolute Error (MAE)
      - Root Mean Square Error (RMSE)
      - Mean Relative Error (%)
      - Total processing time and mean per-image time

Outputs:
  • Pandas DataFrame  (for CSV export)
  • Printed / file-written evaluation report

Lecture references:
  - L06 Performance Evaluation section of the proposal
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from config import EVALUATION


class EvaluationMetrics:
    """Accumulates per-image results and computes aggregate statistics."""

    def __init__(self):
        self.records: list[dict] = []

    # ------------------------------------------------------------------
    def add(self, filename: str,
            predicted: int,
            ground_truth: int | None,
            processing_time_s: float):
        """Record one image's result."""
        rec = {
            "filename":        filename,
            "predicted":       predicted,
            "ground_truth":    ground_truth,
            "processing_time": round(processing_time_s, 4),
        }
        if ground_truth is not None:
            err = abs(predicted - ground_truth)
            rel = err / ground_truth * 100 if ground_truth > 0 else 0.0
            rec["absolute_error"]   = err
            rec["relative_error_%"] = round(rel, 2)
            rec["correct"]          = (predicted == ground_truth)
        else:
            rec["absolute_error"]   = None
            rec["relative_error_%"] = None
            rec["correct"]          = None
        self.records.append(rec)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Compute aggregate statistics over all added records."""
        df = self.to_dataframe()
        evaluated = df.dropna(subset=["ground_truth"])

        stats = {
            "total_images":    len(df),
            "evaluated":       len(evaluated),
            "mean_time_s":     round(df["processing_time"].mean(), 4),
            "total_time_s":    round(df["processing_time"].sum(), 4),
        }

        if len(evaluated) > 0:
            errors = evaluated["absolute_error"].astype(float)
            correct = evaluated["correct"].astype(bool)
            stats["counting_accuracy_%"] = round(correct.mean() * 100, 2)
            stats["MAE"]                 = round(errors.mean(), 4)
            stats["RMSE"]                = round(float(np.sqrt((errors**2).mean())), 4)
            stats["mean_relative_error_%"] = round(
                evaluated["relative_error_%"].astype(float).mean(), 2
            )
        return stats

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    # ------------------------------------------------------------------
    def save_csv(self, path: str = None):
        path = path or EVALUATION["results_csv"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        print(f"[Evaluation] Results saved → {path}")

    # ------------------------------------------------------------------
    def print_report(self, save_path: str = None):
        stats = self.summary()
        lines = [
            "=" * 60,
            "  OBJECT COUNTING SYSTEM — EVALUATION REPORT",
            "  EC7205 / EE7204  |  University of Ruhuna",
            "=" * 60,
            f"  Total images processed : {stats['total_images']}",
            f"  Images with ground truth: {stats['evaluated']}",
            f"  Mean processing time   : {stats['mean_time_s']} s / image",
            f"  Total processing time  : {stats['total_time_s']} s",
        ]
        if "counting_accuracy_%" in stats:
            lines += [
                "-" * 60,
                f"  Counting Accuracy      : {stats['counting_accuracy_%']} %",
                f"  Mean Absolute Error    : {stats['MAE']}",
                f"  RMSE                   : {stats['RMSE']}",
                f"  Mean Relative Error    : {stats['mean_relative_error_%']} %",
            ]
        lines += ["=" * 60]

        report = "\n".join(lines)
        print(report)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report + "\n\n")
                f.write(self.to_dataframe().to_string(index=False))
            print(f"[Evaluation] Report saved → {save_path}")

        return stats


# ──────────────────────────────────────────────────────────────────────
def load_ground_truth(csv_path: str) -> dict[str, int]:
    """
    Load ground truth CSV.
    Expected columns: filename, ground_truth_count
    Returns dict  {filename_stem: count}
    """
    try:
        df = pd.read_csv(csv_path)
        return {
            Path(row["filename"]).stem: int(row["ground_truth_count"])
            for _, row in df.iterrows()
        }
    except FileNotFoundError:
        print(f"[Evaluation] Ground truth CSV not found: {csv_path}")
        return {}
