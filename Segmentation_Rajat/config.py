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
    "DATASET_CSV":   str("E:\MyProjects\Gsoc_2026_Official\data_resampled\dataset_resampled.csv"), # provdies address for resamplaed dataset path
    "DATASET_FOR_DOB_SCV": str("E:/MyProjects/Gsoc_2026_Official/data_canonical/tables/scan_index.csv"), # provides address of data from where we intake feature to do DOB SCV
    "SPLITS_JSON": str(BASE_DIR / "MetaData" / "splits.json"), # path to save the splits json file
    "STATS_JSON":  str(BASE_DIR / "MetaData" / "dataset_stats.json"), # path to save the dataset statistics json file
    "VAL_SIZE":    0.15, # Val Percentage
    "TEST_SIZE":   0.15, # Test Percentage
    "RANDOM_SEED": 42, # for reproducibility
    "TASK": "binary", #can also be "multi", multi -> RCA, LCA, LCX, LADX
}

do_heart_roi_masking = True
add_heart_mask_channel = False

dataloader_config = {
    "BATCH_SIZE":  1,               # safe for <12GB VRAM
    "ROI_SIZE":    (128, 128, 35),    # 96^3 patches
    "CACHE_RATE":  1.0,             # auto-overridden by get_cache_rate()
    "NUM_WORKERS": 0,               # must be 0 on Windows
    "HEART_MASK_FLAG": do_heart_roi_masking, #     if True, dataloader will load heart masks and return as additional channel
    "ADD_HEART_MASK_CHANNEL": add_heart_mask_channel, # if True, dataloader will load heart masks and return as additional channel
    "HEART_MODEL_PATH": str(BASE_DIR / "LW_UNET_TVERSKY" / "best_model.pth"), # path to the pretrained heart segmentation model,
    "ADD_COORD_CHANNELS": True, # if True, dataloader will add coordinate convolution channels to the input
    "COORD_MODE": "normalized", # "normalized" or "absolute", only relevant if ADD_COORD_CHANNELS is True
    "DUAL_HU_WINDOWING": True, # if True, then do DUAL HU WINDOWING
}

HU_CONFIG = {
    "WINDOW_LEVEL": 100,
    "WINDOW_WIDTH": 500,
    "A_MIN":        -150,           # WL - WW/2
    "A_MAX":        350,            # WL + WW/2
}