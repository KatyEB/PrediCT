# PrediCT — CAC scoring

Ten files. Each one runs on its own.

    load.py         folder in  ->  Volume (raw HU + spacing + patient id)
    preprocess.py   orient, resample, locate heart, crop
    models.py       read YAML, verify sha256, gate, window, infer
    geometry.py     exact polygon area
    scoring.py      Agatston, lesions, volume, risk
    pipeline.py     the spine: load -> preprocess -> gate -> infer -> score -> write
    run.py          command line
    evaluate.py     Dice / MAE / risk agreement — only when ground truth exists
    server.py       HTTP wrapper around pipeline.py
    ui/index.html   viewer

## Install

    pip install numpy scipy SimpleITK pydicom pyyaml
    pip install fastapi uvicorn              # for server.py
    pip install torch monai                  # for real models
    pip install TotalSegmentator             # for roi: heart

The dummy model runs without torch, so the whole pipeline is testable before
a checkpoint exists.

## Use

    python load.py <folder>                  see what is in a folder
    python run.py --list                     available models
    python run.py <folder>                   score with every model
    python run.py <folder> --legacy          reproduce the old numbers
    python server.py                         then http://127.0.0.1:8000
    python evaluate.py results/results.csv a1-roi a3-coverage-v2
    pytest -q                                70 tests

## Where checkpoints live

Outside the repo. A YAML in models/ points at one:

    weights: approach3_coverage_v2/best_model.pth    # relative to models_root
    sha256:  4e9c...

models_root resolves: PREDICT_MODELS_ROOT, then models_root in
~/.predict/settings.yaml, then ./runs.

Once sha256 is filled in, a manifest that says v2 can never load v1.

## Resumable

Each patient-model pair writes its row the moment it finishes. Interrupt a
66-patient run at 60 and rerunning scores the remaining 6. --restart forces
a full rescore.

## Two settings that change the number

    --legacy      no minimum-lesion rule, no thickness correction.
                  Reproduces agatston_scoring_a{1,3}.py exactly.
    --lesions 3d  3D connected components. DIFFERENT numbers, not more
                  precise ones. Stamped on output; never mix with 2d.
