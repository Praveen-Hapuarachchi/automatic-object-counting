"""
config.py
=========
Central configuration for the Automatic Object Counting System.
EC7205 / EE7204 - Image Processing and Computer Vision
University of Ruhuna
"""

# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
PREPROCESSING = {
    "filter_type":      "gaussian",
    "gaussian_kernel":  (5, 5),
    "gaussian_sigma":   1.0,
    "median_kernel":    5,
    # CLAHE handles uneven illumination better than global HE
    "use_clahe":        True,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid":  (8, 8),
}

# ─────────────────────────────────────────────
#  SEGMENTATION
# ─────────────────────────────────────────────
SEGMENTATION = {
    "method":          "otsu",      # 'otsu' | 'adaptive'
    "use_watershed":   True,
    "adaptive_block_size": 31,
    "adaptive_c":      5,
    # True = objects are darker than background (buttons on light fabric)
    "invert":          True,
}

# ─────────────────────────────────────────────
#  MORPHOLOGICAL OPERATIONS
# Tuned via grid-search: small 3x3 kernels avoid merging touching objects
# ─────────────────────────────────────────────
MORPHOLOGY = {
    "erosion":  {"kernel": (3, 3), "iterations": 1},
    "dilation": {"kernel": (3, 3), "iterations": 1},
    "opening":  {"kernel": (3, 3), "iterations": 1},   # small = no merging
    "closing":  {"kernel": (3, 3), "iterations": 1},   # fills button holes
}

# ─────────────────────────────────────────────
#  FEATURE FILTERING
# ─────────────────────────────────────────────
FEATURE_FILTER = {
    "min_area":        300,     # tuned: removes noise, keeps all buttons
    "max_area":        None,
    "min_circularity": 0.0,
    "min_solidity":    0.3,
}

# ─────────────────────────────────────────────
#  WATERSHED
# ─────────────────────────────────────────────
WATERSHED = {
    "dist_threshold":          0.30,   # tuned: best split for touching objects
    "bg_dilation_kernel":      (3, 3),
    "bg_dilation_iterations":  3,
}

# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────
VISUALISATION = {
    "bbox_color":       (0, 255, 0),
    "bbox_thickness":   2,
    "centroid_color":   (0, 0, 255),
    "centroid_radius":  4,
    "font_scale":       0.5,
    "font_thickness":   1,
    "font_color":       (255, 255, 0),
    "save_steps":       True,
}

# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
EVALUATION = {
    "ground_truth_csv": "data/ground_truth.csv",
    "results_csv":      "outputs/results.csv",
    "report_path":      "outputs/evaluation_report.txt",
}

# ─────────────────────────────────────────────
#  I/O
# ─────────────────────────────────────────────
IO = {
    "input_dir":         "data/images",
    "output_dir":        "outputs/visualizations",
    "supported_formats": (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"),
}
