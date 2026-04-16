# VisDrone Tracking-by-Detection Pipeline

## Overview
This repository implements an end-to-end tracking-by-detection pipeline for VisDrone aerial data.
The system uses Faster R-CNN for frame-level detection and a simplified ByteTrack-style IoU tracker for temporal association.
The implementation is aligned with the final assignment notebook behavior and exported artifact metrics.

## Features
- End-to-end detection and tracking pipeline
- Faster R-CNN detector training and COCO evaluation
- ByteTrack-style IoU association (no ReID)
- Detection and tracking metric export
- Stage-based CLI execution
- Reproducible artifact outputs

## Project Structure
```
visdrone_pipeline/
  dataset.py         # COCO dataset loading utilities
  preprocess.py      # VisDrone -> COCO conversion and split resolution
  detector.py        # Faster R-CNN build/train/eval/checkpoint helpers
  tracking.py        # Simplified ByteTrack-style IoU tracker + MOT inference
  tracking_eval.py   # MOTA/HOTA/IDSW/precision/recall evaluation

run_pipeline.py      # Stage-based CLI pipeline entrypoint
artifacts/           # Generated outputs (COCO json, checkpoints, metrics, reports)
```

## Setup
- Python: 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Run from the repository root.

- Train detector:

```bash
python run_pipeline.py --stage TRAIN --epochs 4 --batch-size 2
```

- Run full pipeline:

```bash
python run_pipeline.py --stage ALL --epochs 4
```

- Run tracking inference only:

```bash
python run_pipeline.py --stage MOT_INFER
```

- Run tracking evaluation only:

```bash
python run_pipeline.py --stage TRACK_EVAL
```

You can override dataset paths when needed:

```bash
python run_pipeline.py --stage ALL \
  --det-train-dir VisDrone2019-DET-train \
  --det-val-dir VisDrone2019-DET-val \
  --mot-val-dir VisDrone2019-MOT-val
```

## Results
Final reported metrics from the aligned artifact summary:

Detection:
- mAP@0.50 = 0.3285
- mAP@0.50:0.95 = 0.1964
- mAR@0.50 = 0.4026
- mAR@0.50:0.95 = 0.2603

Tracking:
- MOTA = 0.1157
- HOTA = 0.3641
- IDSW = 264

## Method Summary
- Detector: Faster R-CNN (ResNet-50 FPN), trained on VisDrone DET-train and evaluated on DET-val.
- Tracker: Simplified ByteTrack-style association with IoU matching, score splitting, and track lifecycle rules.
- ReID: Not used.

## Limitations
- No ReID branch, which increases identity switches in occlusion-heavy scenes.
- IoU-only association is less robust to long-term occlusion and abrupt motion.
- Moderate MOTA reflects accumulated detector misses and identity fragmentation.

## Future Work
- Integrate appearance-based ReID for stronger identity continuity.
- Evaluate stronger detectors for improved small-object localization and recall.
- Increase training budget and perform targeted threshold/hyperparameter tuning.
