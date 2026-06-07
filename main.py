"""
main.py
=======
Automatic Object Counting System — Main Entry Point
EC7205 / EE7204 | University of Ruhuna

Usage
-----
# Count objects in all images inside data/images/
python main.py

# Single image
python main.py --image data/images/sample.jpg

# Single image with known ground truth
python main.py --image data/images/sample.jpg --gt 12

# Batch with ground truth CSV
python main.py --gt_csv data/ground_truth.csv

# Use adaptive thresholding (better for uneven illumination)
python main.py --method adaptive

# Disable watershed (for well-separated objects)
python main.py --no_watershed
"""

import argparse
import time
import sys
from pathlib import Path

import cv2

# ── Project imports ────────────────────────────────────────────────────
from pipeline.preprocessor   import Preprocessor
from pipeline.segmentor      import Segmentor
from pipeline.morphology     import MorphologyProcessor
from pipeline.counter        import ObjectCounter
from pipeline.feature_filter import FeatureFilter
from evaluation.metrics      import EvaluationMetrics
from utils.visualizer        import Visualizer
from utils.io_utils          import (load_image, save_image,
                                     collect_images, load_ground_truth)
from config                  import IO, VISUALISATION, EVALUATION, SEGMENTATION


# ══════════════════════════════════════════════════════════════════════
def run_pipeline(image_path: str,
                 preprocessor:    Preprocessor,
                 segmentor:       Segmentor,
                 morphology:      MorphologyProcessor,
                 counter:         ObjectCounter,
                 feature_filter:  FeatureFilter,
                 visualizer:      Visualizer,
                 save_output:     bool = True) -> tuple[int, dict]:
    """
    Execute the full image processing pipeline on one image.

    Returns
    -------
    final_count : int
    timing : dict  (step-by-step timings in seconds)
    """
    all_steps = {}
    timings   = {}

    # ── Load ──────────────────────────────────────────────────────────
    t0   = time.perf_counter()
    orig = load_image(image_path)
    timings["load"] = time.perf_counter() - t0

    # ── Step 1 : Preprocessing ────────────────────────────────────────
    t0 = time.perf_counter()
    enhanced, pre_steps = preprocessor.process(orig)
    timings["preprocess"] = time.perf_counter() - t0
    all_steps.update(pre_steps)

    # ── Step 2 : Segmentation ─────────────────────────────────────────
    t0 = time.perf_counter()
    binary, seg_steps = segmentor.process(enhanced)
    timings["segment"] = time.perf_counter() - t0
    all_steps.update(seg_steps)

    # ── Step 3 : Morphological Processing ────────────────────────────
    t0 = time.perf_counter()
    refined, morph_steps = morphology.process(binary)
    timings["morphology"] = time.perf_counter() - t0
    all_steps.update(morph_steps)

    # ── Step 4 : Connected Component Counting ────────────────────────
    t0 = time.perf_counter()
    raw_count, components, label_map = counter.process(refined)
    timings["counting"] = time.perf_counter() - t0

    # ── Step 5 : Feature Filtering ────────────────────────────────────
    t0 = time.perf_counter()
    accepted, rejected = feature_filter.filter(components)
    final_count = len(accepted)
    timings["filtering"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())

    # ── Visualisation ─────────────────────────────────────────────────
    annotated = visualizer.annotate(orig, accepted, rejected, final_count)

    stem = Path(image_path).stem
    if save_output:
        out_img = str(Path(IO["output_dir"]) / f"{stem}_result.jpg")
        save_image(annotated, out_img)

        if VISUALISATION.get("save_steps", True):
            strip_path = str(Path(IO["output_dir"]) / f"{stem}_pipeline.jpg")
            visualizer.save_pipeline_strip(orig, all_steps, annotated, strip_path)

    return final_count, timings


# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Automatic Object Counting — EC7205 Mini Project"
    )
    parser.add_argument("--image",        type=str,  default=None,
                        help="Single image path (skips batch mode)")
    parser.add_argument("--gt",           type=int,  default=None,
                        help="Ground truth count for single image")
    parser.add_argument("--gt_csv",       type=str,
                        default=EVALUATION["ground_truth_csv"],
                        help="CSV with ground truth counts")
    parser.add_argument("--method",       type=str,
                        default=SEGMENTATION["method"],
                        choices=["otsu", "adaptive"],
                        help="Thresholding method")
    parser.add_argument("--no_watershed", action="store_true",
                        help="Disable watershed post-processing")
    parser.add_argument("--no_save",      action="store_true",
                        help="Do not save output images")
    args = parser.parse_args()

    # ── Build pipeline components ─────────────────────────────────────
    import config
    config.SEGMENTATION["method"]        = args.method
    config.SEGMENTATION["use_watershed"] = not args.no_watershed

    preprocessor   = Preprocessor()
    segmentor      = Segmentor()
    morphology     = MorphologyProcessor()
    counter        = ObjectCounter()
    feature_filter = FeatureFilter()
    visualizer     = Visualizer()
    metrics        = EvaluationMetrics()

    # ── Determine image list ──────────────────────────────────────────
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = collect_images()
        if not image_paths:
            print(f"[Main] No images found in '{IO['input_dir']}'. "
                  "Run demo.py first to generate test images.")
            sys.exit(0)

    # ── Load ground truth ─────────────────────────────────────────────
    gt_lookup = load_ground_truth(args.gt_csv)
    if args.gt is not None and args.image:
        gt_lookup[Path(args.image).stem] = args.gt

    # ── Run pipeline ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  EC7205 Automatic Object Counting")
    print(f"  Processing {len(image_paths)} image(s) ...")
    print(f"{'='*55}")

    for img_path in image_paths:
        try:
            t_start = time.perf_counter()
            count, timings = run_pipeline(
                str(img_path),
                preprocessor, segmentor, morphology,
                counter, feature_filter, visualizer,
                save_output=not args.no_save,
            )
            elapsed = time.perf_counter() - t_start

            gt = gt_lookup.get(img_path.stem)
            metrics.add(img_path.name, count, gt, elapsed)

            gt_str = f"  GT={gt}" if gt is not None else ""
            match  = " ✓" if (gt is not None and count == gt) else \
                     " ✗" if gt is not None else ""
            print(f"  {img_path.name:<30}  count={count:>4}{gt_str}  "
                  f"{elapsed*1000:.1f} ms{match}")

        except Exception as exc:
            print(f"  [ERROR] {img_path.name}: {exc}")

    # ── Report ────────────────────────────────────────────────────────
    print()
    metrics.print_report(save_path=EVALUATION["report_path"])
    metrics.save_csv(EVALUATION["results_csv"])

    # Evaluation chart
    if gt_lookup:
        visualizer.save_evaluation_chart(
            metrics.records,
            "outputs/evaluation_chart.png"
        )


if __name__ == "__main__":
    main()
