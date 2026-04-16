from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from visdrone_pipeline import (
    ByteTrackParams,
    VisDroneDataset,
    collate_fn,
    configure_model_for_runtime,
    convert_visdrone_to_coco,
    evaluate_detector,
    evaluate_tracking,
    find_latest_checkpoint,
    get_model,
    load_checkpoint,
    resolve_images_dir,
    run_inference_on_mot,
    train_detector,
)


STAGES = ("ALL", "TRAIN", "DET_EVAL", "MOT_INFER", "TRACK_EVAL")


def parse_args() -> argparse.Namespace:
    """
    Description:
    Parse CLI arguments for stage-based pipeline execution.
    Inputs:
    - Command-line arguments.
    Outputs:
    - argparse.Namespace with pipeline runtime settings.
    """
    parser = argparse.ArgumentParser(description="VisDrone tracking-by-detection pipeline")

    parser.add_argument("--stage", type=str, default="ALL", choices=STAGES)
    parser.add_argument("--det-train-dir", type=str, default="VisDrone2019-DET-train")
    parser.add_argument("--det-val-dir", type=str, default="VisDrone2019-DET-val")
    parser.add_argument("--mot-val-dir", type=str, default="VisDrone2019-MOT-val")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--scheduler-step-size", type=int, default=3)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)

    parser.add_argument("--det-score-thresh", type=float, default=0.3)
    parser.add_argument("--mot-score-thresh", type=float, default=0.5)
    parser.add_argument("--track-assoc-min-score", type=float, default=0.4)
    parser.add_argument("--max-frames", type=int, default=400)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-detections-per-img", type=int, default=100)

    parser.add_argument("--resume-if-available", dest="resume_if_available", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume_if_available", action="store_false")
    parser.add_argument("--skip-if-exists", dest="skip_if_exists", action="store_true", default=True)
    parser.add_argument("--no-skip", dest="skip_if_exists", action="store_false")

    return parser.parse_args()


def build_paths(artifacts_dir: Path) -> Dict[str, Path]:
    """
    Description:
    Build standard artifact paths used by the pipeline.
    Inputs:
    - artifacts_dir: Root artifacts directory.
    Outputs:
    - Dictionary of key artifact paths.
    """
    return {
        "artifacts_dir": artifacts_dir,
        "coco_dir": artifacts_dir / "coco",
        "detector_dir": artifacts_dir / "detector",
        "mot_dir": artifacts_dir / "mot",
        "reports_dir": artifacts_dir / "reports",
        "run_config_json": artifacts_dir / "run_config.json",
        "metrics_summary_json": artifacts_dir / "metrics_summary.json",
        "manifest_json": artifacts_dir / "artifact_manifest.json",
        "train_json": artifacts_dir / "coco" / "train.json",
        "val_json": artifacts_dir / "coco" / "val.json",
        "train_history_json": artifacts_dir / "detector" / "train_history.json",
        "train_history_csv": artifacts_dir / "detector" / "train_history.csv",
        "latest_ckpt": artifacts_dir / "detector" / "detector_latest.pth",
        "mot_summary_json": artifacts_dir / "mot" / "mot_inference_summary.json",
        "tracking_metrics_json": artifacts_dir / "mot" / "tracking_metrics.json",
    }


def ensure_artifact_dirs(paths: Dict[str, Path]) -> None:
    """
    Description:
    Ensure required artifact directories exist.
    Inputs:
    - paths: Path dictionary from build_paths.
    Outputs:
    - None.
    """
    for key in ("artifacts_dir", "coco_dir", "detector_dir", "mot_dir", "reports_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    (paths["mot_dir"] / "detections").mkdir(parents=True, exist_ok=True)
    (paths["mot_dir"] / "tracks").mkdir(parents=True, exist_ok=True)


def ensure_coco_annotations(
    det_train_dir: Path,
    det_val_dir: Path,
    coco_dir: Path,
    skip_if_exists: bool,
) -> Tuple[Path, Path]:
    """
    Description:
    Ensure train/val COCO annotations are available.
    Inputs:
    - det_train_dir: DET train split path.
    - det_val_dir: DET val split path.
    - coco_dir: COCO output directory.
    - skip_if_exists: Whether to reuse existing outputs.
    Outputs:
    - Tuple of train_json and val_json paths.
    """
    train_json = coco_dir / "train.json"
    val_json = coco_dir / "val.json"

    if skip_if_exists and train_json.exists() and val_json.exists():
        print("[convert] Skipping conversion (COCO files already exist).")
        return train_json, val_json

    print("[convert] Converting VisDrone DET annotations to COCO JSON...")
    stats = convert_visdrone_to_coco(
        det_train_dir=det_train_dir,
        det_val_dir=det_val_dir,
        output_dir=coco_dir,
    )
    print(json.dumps(stats, indent=2))
    return train_json, val_json


def build_data_loaders(
    det_train_dir: Path,
    det_val_dir: Path,
    train_json: Path,
    val_json: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Description:
    Build train and validation dataloaders for detector training/evaluation.
    Inputs:
    - det_train_dir: DET train split path.
    - det_val_dir: DET val split path.
    - train_json: COCO train annotation path.
    - val_json: COCO val annotation path.
    - batch_size: Dataloader batch size.
    - num_workers: Dataloader workers.
    Outputs:
    - Tuple of train_loader and val_loader.
    """
    train_images = resolve_images_dir(det_train_dir)
    val_images = resolve_images_dir(det_val_dir)

    train_dataset = VisDroneDataset(images_dir=train_images, annotation_json=train_json)
    val_dataset = VisDroneDataset(images_dir=val_images, annotation_json=val_json)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def write_train_history_csv(history: List[Dict[str, object]], output_csv: Path) -> None:
    """
    Description:
    Write compact detector training history to CSV.
    Inputs:
    - history: Training history list.
    - output_csv: CSV file path.
    Outputs:
    - None.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "mAP@0.50",
                "mAP@0.50:0.95",
                "mAR@0.50",
                "mAR@0.50:0.95",
                "precision",
                "recall",
            ]
        )
        for row in history:
            metrics = row.get("val_metrics", {}) if isinstance(row, dict) else {}
            writer.writerow(
                [
                    row.get("epoch"),
                    row.get("lr"),
                    row.get("train_loss"),
                    metrics.get("mAP@0.50"),
                    metrics.get("mAP@0.50:0.95"),
                    metrics.get("mAR@0.50"),
                    metrics.get("mAR@0.50:0.95"),
                    metrics.get("precision"),
                    metrics.get("recall"),
                ]
            )


def run_train_stage(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    device: torch.device,
    det_train_dir: Path,
    det_val_dir: Path,
) -> Path:
    """
    Description:
    Execute detector training stage with resume support.
    Inputs:
    - args: Parsed CLI arguments.
    - paths: Artifact path dictionary.
    - device: Execution device.
    - det_train_dir: DET train split path.
    - det_val_dir: DET val split path.
    Outputs:
    - Path to latest checkpoint after stage.
    """
    train_json, val_json = ensure_coco_annotations(
        det_train_dir=det_train_dir,
        det_val_dir=det_val_dir,
        coco_dir=paths["coco_dir"],
        skip_if_exists=args.skip_if_exists,
    )
    train_loader, val_loader = build_data_loaders(
        det_train_dir=det_train_dir,
        det_val_dir=det_val_dir,
        train_json=train_json,
        val_json=val_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    latest_ckpt, latest_epoch = find_latest_checkpoint(paths["detector_dir"])
    resume = bool(args.resume_if_available and latest_ckpt is not None and latest_epoch < args.epochs)

    model = get_model(num_object_classes=10, pretrained=not resume).to(device)
    model = configure_model_for_runtime(model=model, detections_per_img=args.model_detections_per_img)

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    start_epoch = 1
    if resume and latest_ckpt is not None:
        checkpoint_payload = load_checkpoint(model=model, checkpoint_path=str(latest_ckpt), device=device)
        if "optimizer_state_dict" in checkpoint_payload:
            optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
        start_epoch = int(checkpoint_payload.get("epoch", latest_epoch)) + 1

    if latest_ckpt is not None and latest_epoch >= args.epochs:
        print(f"[train] Skipping training (latest checkpoint already at epoch {latest_epoch}).")
        return latest_ckpt

    print(f"[train] Training detector from epoch {start_epoch} to {args.epochs}...")
    latest_output_ckpt, history = train_detector(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=paths["detector_dir"],
        epochs=args.epochs,
        start_epoch=start_epoch,
        det_score_thresh=args.det_score_thresh,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        grad_clip_norm=args.grad_clip_norm,
        print_freq=20,
    )

    if latest_output_ckpt is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    shutil.copy2(latest_output_ckpt, paths["latest_ckpt"])
    write_train_history_csv(history=history, output_csv=paths["train_history_csv"])
    return latest_output_ckpt


def run_detection_eval_stage(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    device: torch.device,
    det_train_dir: Path,
    det_val_dir: Path,
) -> Dict[str, float]:
    """
    Description:
    Execute detector validation stage on DET-val.
    Inputs:
    - args: Parsed CLI arguments.
    - paths: Artifact path dictionary.
    - device: Execution device.
    - det_train_dir: DET train split path.
    - det_val_dir: DET val split path.
    Outputs:
    - Detection metric dictionary.
    """
    _, val_json = ensure_coco_annotations(
        det_train_dir=det_train_dir,
        det_val_dir=det_val_dir,
        coco_dir=paths["coco_dir"],
        skip_if_exists=True,
    )

    _, val_loader = build_data_loaders(
        det_train_dir=det_train_dir,
        det_val_dir=det_val_dir,
        train_json=paths["train_json"],
        val_json=val_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    latest_ckpt, _ = find_latest_checkpoint(paths["detector_dir"])
    if latest_ckpt is None:
        raise FileNotFoundError("Detection evaluation requires a detector checkpoint, but none was found.")

    model = get_model(num_object_classes=10, pretrained=False).to(device)
    model = configure_model_for_runtime(model=model, detections_per_img=args.model_detections_per_img)
    load_checkpoint(model=model, checkpoint_path=str(latest_ckpt), device=device)

    print("[det_eval] Evaluating detector on DET-val...")
    metrics = evaluate_detector(
        model=model,
        data_loader=val_loader,
        device=device,
        score_thresh=args.det_score_thresh,
    )
    return metrics


def run_mot_infer_stage(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    device: torch.device,
    mot_val_dir: Path,
) -> Dict[str, Dict[str, object]]:
    """
    Description:
    Execute MOT inference and simplified ByteTrack-style association.
    Inputs:
    - args: Parsed CLI arguments.
    - paths: Artifact path dictionary.
    - device: Execution device.
    - mot_val_dir: MOT validation split path.
    Outputs:
    - Sequence-level MOT inference summary dictionary.
    """
    latest_ckpt, _ = find_latest_checkpoint(paths["detector_dir"])
    if latest_ckpt is None:
        raise FileNotFoundError("MOT inference requires a detector checkpoint, but none was found.")

    model = get_model(num_object_classes=10, pretrained=False).to(device)
    model = configure_model_for_runtime(model=model, detections_per_img=args.model_detections_per_img)
    load_checkpoint(model=model, checkpoint_path=str(latest_ckpt), device=device)

    print("[mot_infer] Running MOT inference...")
    summary = run_inference_on_mot(
        model=model,
        mot_root=str(mot_val_dir),
        device=device,
        output_dir=str(paths["mot_dir"]),
        score_thresh=args.mot_score_thresh,
        assoc_min_score=args.track_assoc_min_score,
        max_frames=args.max_frames,
        tracker_params=ByteTrackParams(),
        use_tracking=True,
    )

    paths["mot_summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_tracking_eval_stage(paths: Dict[str, Path], mot_val_dir: Path) -> Dict[str, object]:
    """
    Description:
    Execute tracking evaluation against MOT ground truth.
    Inputs:
    - paths: Artifact path dictionary.
    - mot_val_dir: MOT validation split path.
    Outputs:
    - Tracking metrics dictionary containing overall and per-sequence metrics.
    """
    tracks_dir = paths["mot_dir"] / "tracks"
    if not tracks_dir.exists() or not any(tracks_dir.glob("*.txt")):
        raise FileNotFoundError(f"No track files found under: {tracks_dir}")

    print("[track_eval] Evaluating tracking outputs...")
    metrics = evaluate_tracking(
        gt_root=str(mot_val_dir),
        pred_dir=str(tracks_dir),
        iou_threshold=0.5,
        class_aware=True,
    )
    paths["tracking_metrics_json"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_run_config(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    """
    Description:
    Persist run configuration used by the CLI.
    Inputs:
    - args: Parsed CLI arguments.
    - paths: Artifact path dictionary.
    Outputs:
    - None.
    """
    payload = {
        "stage": args.stage,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "grad_clip_norm": args.grad_clip_norm,
        "det_score_thresh": args.det_score_thresh,
        "mot_score_thresh": args.mot_score_thresh,
        "track_assoc_min_score": args.track_assoc_min_score,
        "max_frames": args.max_frames,
        "model_detections_per_img": args.model_detections_per_img,
        "resume_if_available": args.resume_if_available,
        "skip_if_exists": args.skip_if_exists,
        "artifacts_dir": str(paths["artifacts_dir"]),
    }
    paths["run_config_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_metrics_summary(
    paths: Dict[str, Path],
    det_metrics: Optional[Dict[str, float]],
    track_metrics: Optional[Dict[str, object]],
) -> None:
    """
    Description:
    Persist final metrics summary in notebook-compatible schema.
    Inputs:
    - paths: Artifact path dictionary.
    - det_metrics: Detection metric dictionary.
    - track_metrics: Tracking metric dictionary.
    Outputs:
    - None.
    """
    payload = {
        "detection": det_metrics or {},
        "tracking_overall": (track_metrics or {}).get("overall", {}),
    }
    paths["metrics_summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest(paths: Dict[str, Path]) -> None:
    """
    Description:
    Write compact artifact manifest for reproducibility.
    Inputs:
    - paths: Artifact path dictionary.
    Outputs:
    - None.
    """
    payload = {
        "artifacts_dir": str(paths["artifacts_dir"]),
        "coco_train_json": str(paths["train_json"]),
        "coco_val_json": str(paths["val_json"]),
        "latest_checkpoint": str(paths["latest_ckpt"]),
        "train_history_json": str(paths["train_history_json"]),
        "train_history_csv": str(paths["train_history_csv"]),
        "mot_summary_json": str(paths["mot_summary_json"]),
        "tracking_metrics_json": str(paths["tracking_metrics_json"]),
        "run_config_json": str(paths["run_config_json"]),
        "metrics_summary_json": str(paths["metrics_summary_json"]),
    }
    paths["manifest_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_existing_detection_metrics(paths: Dict[str, Path]) -> Dict[str, float]:
    """
    Description:
    Load best available detection metrics from existing artifacts.
    Inputs:
    - paths: Artifact path dictionary.
    Outputs:
    - Detection metric dictionary (or empty dict).
    """
    if paths["metrics_summary_json"].exists():
        payload = json.loads(paths["metrics_summary_json"].read_text(encoding="utf-8"))
        if isinstance(payload.get("detection"), dict):
            return payload["detection"]

    if paths["train_history_json"].exists():
        history = json.loads(paths["train_history_json"].read_text(encoding="utf-8"))
        if history and isinstance(history[-1], dict):
            return history[-1].get("val_metrics", {})

    return {}


def load_existing_tracking_metrics(paths: Dict[str, Path]) -> Dict[str, object]:
    """
    Description:
    Load best available tracking metrics from existing artifacts.
    Inputs:
    - paths: Artifact path dictionary.
    Outputs:
    - Tracking metrics dictionary (or empty overall object).
    """
    if paths["tracking_metrics_json"].exists():
        payload = json.loads(paths["tracking_metrics_json"].read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "overall" in payload:
            return payload

    if paths["metrics_summary_json"].exists():
        payload = json.loads(paths["metrics_summary_json"].read_text(encoding="utf-8"))
        if isinstance(payload.get("tracking_overall"), dict):
            return {"overall": payload["tracking_overall"]}

    return {"overall": {}}


def print_final_results(det_metrics: Dict[str, float], track_metrics: Dict[str, object]) -> None:
    """
    Description:
    Print standardized final result block for assignment reporting.
    Inputs:
    - det_metrics: Detection metric dictionary.
    - track_metrics: Tracking metric dictionary.
    Outputs:
    - None.
    """
    overall = track_metrics.get("overall", {}) if isinstance(track_metrics, dict) else {}

    print("================ FINAL RESULTS ================")
    print()
    print("Detection:")
    print(f"mAP@0.50: {float(det_metrics.get('mAP@0.50', 0.0)):.4f}")
    print(f"mAP@0.50:0.95: {float(det_metrics.get('mAP@0.50:0.95', 0.0)):.4f}")
    print(f"mAR@0.50: {float(det_metrics.get('mAR@0.50', 0.0)):.4f}")
    print(f"mAR@0.50:0.95: {float(det_metrics.get('mAR@0.50:0.95', 0.0)):.4f}")
    print(f"Precision: {float(det_metrics.get('precision', 0.0)):.4f}")
    print(f"Recall: {float(det_metrics.get('recall', 0.0)):.4f}")
    print()
    print("Tracking:")
    print(f"MOTA: {float(overall.get('MOTA', 0.0)):.4f}")
    print(f"HOTA: {float(overall.get('HOTA', 0.0)):.4f}")
    print(f"IDSW: {float(overall.get('IDSW', 0.0)):.0f}")
    print(f"Precision: {float(overall.get('precision', 0.0)):.4f}")
    print(f"Recall: {float(overall.get('recall', 0.0)):.4f}")


def main() -> None:
    """
    Description:
    Entry point for stage-based CLI pipeline execution.
    Inputs:
    - None.
    Outputs:
    - None.
    """
    args = parse_args()
    device = torch.device(args.device)

    det_train_dir = Path(args.det_train_dir)
    det_val_dir = Path(args.det_val_dir)
    mot_val_dir = Path(args.mot_val_dir)

    paths = build_paths(Path(args.artifacts_dir))
    ensure_artifact_dirs(paths)
    write_run_config(args, paths)

    stage = str(args.stage).strip().upper()
    det_metrics: Optional[Dict[str, float]] = None
    track_metrics: Optional[Dict[str, object]] = None

    if stage == "ALL":
        run_train_stage(args=args, paths=paths, device=device, det_train_dir=det_train_dir, det_val_dir=det_val_dir)
        det_metrics = run_detection_eval_stage(args=args, paths=paths, device=device, det_train_dir=det_train_dir, det_val_dir=det_val_dir)
        run_mot_infer_stage(args=args, paths=paths, device=device, mot_val_dir=mot_val_dir)
        track_metrics = run_tracking_eval_stage(paths=paths, mot_val_dir=mot_val_dir)
    elif stage == "TRAIN":
        run_train_stage(args=args, paths=paths, device=device, det_train_dir=det_train_dir, det_val_dir=det_val_dir)
    elif stage == "DET_EVAL":
        det_metrics = run_detection_eval_stage(args=args, paths=paths, device=device, det_train_dir=det_train_dir, det_val_dir=det_val_dir)
    elif stage == "MOT_INFER":
        run_mot_infer_stage(args=args, paths=paths, device=device, mot_val_dir=mot_val_dir)
    elif stage == "TRACK_EVAL":
        track_metrics = run_tracking_eval_stage(paths=paths, mot_val_dir=mot_val_dir)
    else:
        raise ValueError(f"Invalid stage: {stage}")

    if det_metrics is None:
        det_metrics = load_existing_detection_metrics(paths)
    if track_metrics is None:
        track_metrics = load_existing_tracking_metrics(paths)

    write_metrics_summary(paths=paths, det_metrics=det_metrics, track_metrics=track_metrics)
    write_manifest(paths)
    print_final_results(det_metrics=det_metrics, track_metrics=track_metrics)


if __name__ == "__main__":
    main()
