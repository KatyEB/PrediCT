# common_task/config.py

from pathlib import Path

# All paths relative to this file's location (common_task/)
BASE_DIR = Path(__file__).parent

ground_truth_config = {
    "INPUT_CSV":       str(BASE_DIR / "train_raw.csv"),
    "OUTPUT_CSV":      str(BASE_DIR / "train.csv"),
    "TOTALSEG_RAW":    str(BASE_DIR / "totalseg_outputs"),
    "HEART_MASKS_DIR": str("ground_truth"), #used for saving so relative path is fine
    "STATS_JSON":      str(BASE_DIR / "ground_truth" / "stats.json"),
    "N_SCANS":         50,                      # ✅ meets 30-50 requirement
    "DEVICE":          "gpu",
    "TASK":            "heartchambers_highres", # ✅ correct task
}

preprocessing_config = {
    "TRAIN_CSV":   str(BASE_DIR / "train.csv"),
    "SPLITS_JSON": str(BASE_DIR / "splits.json"),
    "STATS_JSON":  str(BASE_DIR / "dataset_stats.json"),
    "VAL_SIZE":    0.15,
    "TEST_SIZE":   0.15,
    "RANDOM_SEED": 42,

    # NEW:
    "TASK": "binary" #can also be "multi"
    "HEART_MASK": False # if True, use heart seg model to get heart masks.

}

dataloader_config = {
    "BATCH_SIZE":  1,               # safe for <12GB VRAM
    "ROI_SIZE":    (96, 128, 96),    # 96^3 patches
    "CACHE_RATE":  1.0,             # auto-overridden by get_cache_rate()
    "NUM_WORKERS": 0,               # must be 0 on Windows
}

HU_CONFIG = {
    "WINDOW_LEVEL": 100,
    "WINDOW_WIDTH": 500,
    "A_MIN":        -150,           # WL - WW/2
    "A_MAX":        350,            # WL + WW/2
}