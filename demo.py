"""
demo.py
=======
Generates synthetic test images (simulating industrial buttons/small parts)
and runs the full pipeline on them so you can verify everything works
without needing a real dataset.

Run:
    python demo.py

What it does:
  1. Creates 6 synthetic images with known object counts (12–20 objects)
     saved to data/images/
  2. Writes a matching ground_truth.csv to data/
  3. Runs the full pipeline on all images
  4. Prints and saves the evaluation report

Images simulate:
  - Varying illumination (normal, dark, bright, uneven vignette)
  - Different sized circular objects (buttons)
  - Partial occlusion / touching objects
  - Background noise / texture

EC7205 / EE7204 | University of Ruhuna
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.preprocessor   import Preprocessor
from pipeline.segmentor      import Segmentor
from pipeline.morphology     import MorphologyProcessor
from pipeline.counter        import ObjectCounter
from pipeline.feature_filter import FeatureFilter
from evaluation.metrics      import EvaluationMetrics
from utils.visualizer        import Visualizer
from utils.io_utils          import save_image, load_image
from config                  import IO, EVALUATION, VISUALISATION


# ══════════════════════════════════════════════════════════════════════
#  SYNTHETIC IMAGE GENERATION
# ══════════════════════════════════════════════════════════════════════

def make_button_image(n_objects: int,
                      size: tuple = (480, 640),
                      illumination: str = "normal",
                      seed: int = 42) -> np.ndarray:
    """
    Create a synthetic BGR image with n_objects circular 'buttons'
    on a light fabric-textured background.

    illumination : 'normal' | 'dark' | 'bright' | 'uneven'
    """
    rng = np.random.default_rng(seed)
    h, w = size

    # ── Background: light fabric texture ─────────────────────────────
    bg_val  = 200
    bg      = np.full((h, w), bg_val, dtype=np.uint8)
    noise   = rng.integers(-12, 12, (h, w), dtype=np.int16)
    bg      = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ── Illuminate ────────────────────────────────────────────────────
    if illumination == "dark":
        bg = (bg * 0.55).astype(np.uint8)
    elif illumination == "bright":
        bg = np.clip(bg.astype(np.int16) + 55, 0, 255).astype(np.uint8)
    elif illumination == "uneven":
        # Vignette: darker at edges
        Y, X = np.mgrid[0:h, 0:w]
        cx, cy = w // 2, h // 2
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        dist_norm = dist / dist.max()
        factor = 1.0 - 0.45 * dist_norm
        bg = np.clip(bg * factor, 0, 255).astype(np.uint8)

    canvas = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    # ── Place buttons ─────────────────────────────────────────────────
    min_r, max_r   = 16, 30
    placed         = []   # list of (cx, cy, r)
    max_attempts   = 5000

    button_colors = [
        (40,  40,  40),    # black
        (60,  60, 140),    # navy
        (30, 100,  30),    # dark green
        (130, 60,  30),    # brown
        (100, 30,  80),    # maroon
    ]

    attempts = 0
    while len(placed) < n_objects and attempts < max_attempts:
        attempts += 1
        r  = int(rng.integers(min_r, max_r + 1))
        cx = int(rng.integers(r + 5, w - r - 5))
        cy = int(rng.integers(r + 5, h - r - 5))

        # Check overlap (allow slight touch to simulate real scenarios)
        overlap = False
        for px, py, pr in placed:
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < (r + pr) * 0.82:     # 18% overlap allowed
                overlap = True
                break
        if overlap:
            continue

        placed.append((cx, cy, r))

        # Draw button
        color = button_colors[len(placed) % len(button_colors)]
        cv2.circle(canvas, (cx, cy), r, color, -1)

        # Button hole pattern (2–4 holes)
        n_holes = int(rng.integers(2, 5))
        hole_r  = max(2, r // 6)
        for i in range(n_holes):
            angle  = 2 * np.pi * i / n_holes
            hx     = int(cx + (r // 3) * np.cos(angle))
            hy     = int(cy + (r // 3) * np.sin(angle))
            cv2.circle(canvas, (hx, hy), hole_r, (220, 210, 200), -1)

        # Subtle rim highlight
        cv2.circle(canvas, (cx, cy), r, (180, 180, 180), 1)

    actual_count = len(placed)
    return canvas, actual_count


def generate_dataset(out_dir: str = "data/images",
                     gt_csv:  str = "data/ground_truth.csv"):
    """Generate 6 synthetic test images with varying conditions."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    specs = [
        # (filename,          n_objects, illumination, seed)
        ("img_01_normal.jpg",    15,   "normal",  1),
        ("img_02_dark.jpg",      12,   "dark",    2),
        ("img_03_bright.jpg",    18,   "bright",  3),
        ("img_04_uneven.jpg",    14,   "uneven",  4),
        ("img_05_crowded.jpg",   20,   "normal",  5),
        ("img_06_sparse.jpg",     8,   "normal",  6),
    ]

    records = []
    print("\n[Demo] Generating synthetic dataset ...")
    for fname, n, illum, seed in specs:
        img, actual = make_button_image(n, illumination=illum, seed=seed)
        save_path   = str(Path(out_dir) / fname)
        cv2.imwrite(save_path, img)
        records.append({"filename": fname, "ground_truth_count": actual})
        print(f"  {fname}  →  requested={n}, placed={actual}, illum={illum}")

    df = pd.DataFrame(records)
    Path(gt_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(gt_csv, index=False)
    print(f"[Demo] Ground truth CSV → {gt_csv}\n")
    return records


# ══════════════════════════════════════════════════════════════════════
#  PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_demo():
    import time

    # ── Generate synthetic data ───────────────────────────────────────
    records  = generate_dataset()
    gt_lookup = {
        Path(r["filename"]).stem: r["ground_truth_count"]
        for r in records
    }

    # ── Build pipeline ────────────────────────────────────────────────
    preprocessor   = Preprocessor()
    segmentor      = Segmentor()
    morphology     = MorphologyProcessor()
    counter        = ObjectCounter()
    feature_filter = FeatureFilter()
    visualizer     = Visualizer()
    metrics        = EvaluationMetrics()

    Path(IO["output_dir"]).mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  EC7205 — Automatic Object Counting DEMO")
    print(f"  University of Ruhuna")
    print(f"{'='*60}")

    from utils.io_utils import collect_images
    image_paths = collect_images("data/images")

    all_steps_store = {}

    for img_path in image_paths:
        t_start = time.perf_counter()
        all_steps = {}

        orig = load_image(str(img_path))

        # Step 1
        enhanced, pre_steps = preprocessor.process(orig)
        all_steps.update(pre_steps)

        # Step 2
        binary, seg_steps = segmentor.process(enhanced)
        all_steps.update(seg_steps)

        # Step 3
        refined, morph_steps = morphology.process(binary)
        all_steps.update(morph_steps)

        # Step 4
        raw_count, components, label_map = counter.process(refined)

        # Step 5
        accepted, rejected = feature_filter.filter(components)
        final_count = len(accepted)

        elapsed = time.perf_counter() - t_start

        # Annotate & save
        annotated = visualizer.annotate(orig, accepted, rejected, final_count)
        stem      = img_path.stem

        out_img   = str(Path(IO["output_dir"]) / f"{stem}_result.jpg")
        save_image(annotated, out_img)

        strip_path = str(Path(IO["output_dir"]) / f"{stem}_pipeline.jpg")
        visualizer.save_pipeline_strip(orig, all_steps, annotated, strip_path)

        gt = gt_lookup.get(stem)
        metrics.add(img_path.name, final_count, gt, elapsed)

        match = " ✓" if final_count == gt else " ✗"
        print(f"  {img_path.name:<28}  "
              f"predicted={final_count:>3}  GT={gt:>3}  "
              f"{elapsed*1000:>6.1f} ms  {match}")

    # ── Report ────────────────────────────────────────────────────────
    print()
    stats = metrics.print_report(save_path=EVALUATION["report_path"])
    metrics.save_csv(EVALUATION["results_csv"])

    visualizer.save_evaluation_chart(
        metrics.records,
        "outputs/evaluation_chart.png"
    )

    print("\n[Demo] Output files:")
    for f in sorted(Path("outputs").rglob("*")):
        if f.is_file():
            print(f"  {f}")

    return stats


if __name__ == "__main__":
    run_demo()
