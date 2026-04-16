import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from pycocotools.cocoeval import COCOeval
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_object_classes: int = 10, pretrained: bool = True) -> torch.nn.Module:
    """
    Description:
    Build a Faster R-CNN model with a VisDrone-specific classifier head.
    Inputs:
    - num_object_classes: Number of VisDrone foreground classes.
    - pretrained: Whether to use torchvision default pretrained weights.
    Outputs:
    - Configured Faster R-CNN model.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # torchvision includes background as class 0.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_object_classes + 1)
    return model


def configure_model_for_runtime(model: torch.nn.Module, detections_per_img: Optional[int] = None) -> torch.nn.Module:
    """
    Description:
    Apply optional runtime inference limits without changing model architecture.
    Inputs:
    - model: Detection model instance.
    - detections_per_img: Optional cap on detections per image.
    Outputs:
    - Runtime-configured model.
    """
    if detections_per_img is None:
        return model
    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "detections_per_img"):
        model.roi_heads.detections_per_img = int(detections_per_img)
    return model


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 20,
    grad_clip_norm: Optional[float] = None,
) -> float:
    """
    Description:
    Train detector for one epoch.
    Inputs:
    - model: Detection model.
    - optimizer: Optimizer instance.
    - data_loader: Training dataloader.
    - device: Execution device.
    - epoch: 1-based epoch index.
    - print_freq: Batch logging frequency.
    - grad_clip_norm: Optional gradient clipping norm.
    Outputs:
    - Average epoch training loss.
    """
    model.train()
    running_loss = 0.0
    valid_steps = 0

    for step, (images, targets) in enumerate(data_loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = float(losses.item())

        if not math.isfinite(loss_value):
            print(f"[Epoch {epoch}] Non-finite loss at step {step}: {loss_value:.5f}. Skipping.")
            continue

        optimizer.zero_grad()
        losses.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()

        running_loss += loss_value
        valid_steps += 1

        if step % print_freq == 0:
            loss_log = ", ".join(f"{name}: {value.item():.4f}" for name, value in loss_dict.items())
            print(f"[Epoch {epoch}] step={step}/{len(data_loader)} total_loss={loss_value:.4f} ({loss_log})")

    avg_loss = running_loss / max(1, valid_steps)
    print(f"[Epoch {epoch}] average_loss={avg_loss:.4f}")
    return avg_loss


def train_detector(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Union[str, Path],
    epochs: int,
    start_epoch: int = 1,
    det_score_thresh: float = 0.30,
    scheduler_step_size: int = 3,
    scheduler_gamma: float = 0.1,
    grad_clip_norm: Optional[float] = 5.0,
    print_freq: int = 20,
) -> Tuple[Optional[Path], List[Dict[str, object]]]:
    """
    Description:
    Run multi-epoch detector training with StepLR scheduling, clipping, checkpoints, and validation.
    Inputs:
    - model: Detection model.
    - optimizer: Optimizer instance.
    - train_loader: Training dataloader.
    - val_loader: Validation dataloader.
    - device: Execution device.
    - output_dir: Directory for checkpoints and training history.
    - epochs: Final epoch index to train up to.
    - start_epoch: Epoch index to start from.
    - det_score_thresh: Validation score threshold.
    - scheduler_step_size: StepLR step size.
    - scheduler_gamma: StepLR gamma.
    - grad_clip_norm: Optional gradient clipping norm.
    - print_freq: Batch logging frequency.
    Outputs:
    - Tuple of latest checkpoint path and history list.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history_path = output_path / "train_history.json"

    history: List[Dict[str, object]] = []
    if history_path.exists():
        try:
            import json

            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []

    scheduler = StepLR(optimizer, step_size=int(scheduler_step_size), gamma=float(scheduler_gamma))
    if start_epoch > 1:
        for _ in range(1, start_epoch):
            scheduler.step()

    latest_checkpoint: Optional[Path] = None
    for epoch in range(start_epoch, epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        avg_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=print_freq,
            grad_clip_norm=grad_clip_norm,
        )
        checkpoint_path = save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, output_dir=output_path)
        metrics = evaluate_detector(model=model, data_loader=val_loader, device=device, score_thresh=det_score_thresh)

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": avg_loss,
                "checkpoint": str(checkpoint_path),
                "val_metrics": metrics,
            }
        )

        scheduler.step()
        latest_checkpoint = checkpoint_path

    import json

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return latest_checkpoint, history


def _to_coco_results(
    outputs: Iterable[Dict[str, torch.Tensor]],
    targets: Iterable[Dict[str, torch.Tensor]],
    score_thresh: float,
) -> List[Dict[str, object]]:
    """
    Description:
    Convert detector outputs to COCO result rows.
    Inputs:
    - outputs: Model outputs for a batch.
    - targets: Batch targets containing image ids.
    - score_thresh: Minimum confidence score.
    Outputs:
    - List of COCO-style detection dictionaries.
    """
    coco_results: List[Dict[str, object]] = []

    for target, output in zip(targets, outputs):
        image_id = int(target["image_id"].item())
        boxes = output["boxes"].detach().cpu()
        scores = output["scores"].detach().cpu()
        labels = output["labels"].detach().cpu()

        for box, score, label in zip(boxes, scores, labels):
            score_value = float(score.item())
            if score_value < score_thresh:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width <= 0 or height <= 0:
                continue

            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [x1, y1, width, height],
                    "score": score_value,
                }
            )

    return coco_results


def evaluate_detector(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.05,
    coco_gt=None,
) -> Dict[str, float]:
    """
    Description:
    Evaluate detector using COCOeval and return notebook-aligned metrics.
    Inputs:
    - model: Detection model.
    - data_loader: Validation dataloader.
    - device: Execution device.
    - score_thresh: Minimum score threshold for predictions.
    - coco_gt: Optional preloaded COCO ground-truth object.
    Outputs:
    - Dictionary containing mAP, mAR, precision, and recall metrics.
    """
    model.eval()

    if coco_gt is None:
        if not hasattr(data_loader.dataset, "coco"):
            raise ValueError("Dataset must expose COCO ground truth as dataset.coco")
        coco_gt = data_loader.dataset.coco

    predictions: List[Dict[str, object]] = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            predictions.extend(_to_coco_results(outputs, targets, score_thresh=score_thresh))

    if not predictions:
        print("No predictions above threshold; returning zero metrics.")
        return {
            "mAP@0.50:0.95": 0.0,
            "mAP@0.50": 0.0,
            "mAR@0.50": 0.0,
            "mAR@0.50:0.95": 0.0,
            "mAR@max=1": 0.0,
            "mAR@max=10": 0.0,
            "mAR@max=100": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    precision = evaluator.eval["precision"][:, :, :, 0, 2]
    precision = precision[precision > -1]
    mean_precision = float(precision.mean()) if precision.size > 0 else 0.0

    recall = evaluator.eval["recall"][:, :, 0, 2]
    recall = recall[recall > -1]
    mean_recall = float(recall.mean()) if recall.size > 0 else 0.0

    iou_thresholds = list(evaluator.params.iouThrs)
    idx_iou50 = min(range(len(iou_thresholds)), key=lambda idx: abs(float(iou_thresholds[idx]) - 0.5))
    recall_iou50 = evaluator.eval["recall"][idx_iou50, :, 0, 2]
    recall_iou50 = recall_iou50[recall_iou50 > -1]
    mean_recall_iou50 = float(recall_iou50.mean()) if recall_iou50.size > 0 else 0.0

    metrics = {
        "mAP@0.50:0.95": float(evaluator.stats[0]),
        "mAP@0.50": float(evaluator.stats[1]),
        "mAR@0.50": mean_recall_iou50,
        "mAR@0.50:0.95": float(evaluator.stats[8]),
        "mAR@max=1": float(evaluator.stats[6]),
        "mAR@max=10": float(evaluator.stats[7]),
        "mAR@max=100": float(evaluator.stats[8]),
        "precision": mean_precision,
        "recall": mean_recall,
    }

    print(
        "Detector metrics: "
        f"mAP@0.50:0.95={metrics['mAP@0.50:0.95']:.4f}, "
        f"mAP@0.50={metrics['mAP@0.50']:.4f}, "
        f"mAR@0.50={metrics['mAR@0.50']:.4f}, "
        f"mAR@0.50:0.95={metrics['mAR@0.50:0.95']:.4f}, "
        f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}"
    )
    return metrics


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: Union[str, Path],
) -> Path:
    """
    Description:
    Save detector checkpoint for a training epoch.
    Inputs:
    - model: Detection model.
    - optimizer: Optimizer instance.
    - epoch: Epoch index.
    - output_dir: Directory where checkpoint is stored.
    Outputs:
    - Path to saved checkpoint file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path / f"detector_epoch_{epoch:03d}.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Union[str, Path], device: torch.device) -> Dict[str, object]:
    """
    Description:
    Load detector checkpoint into model weights.
    Inputs:
    - model: Detection model.
    - checkpoint_path: Checkpoint file path.
    - device: Target device for loading.
    Outputs:
    - Checkpoint payload dictionary.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Tuple[Optional[Path], int]:
    """
    Description:
    Locate the latest epoch checkpoint in a detector checkpoint directory.
    Inputs:
    - checkpoint_dir: Directory containing detector_epoch_XXX.pth files.
    Outputs:
    - Tuple of latest checkpoint path (or None) and its epoch number.
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_path.glob("detector_epoch_*.pth"))
    if not checkpoints:
        return None, 0

    latest = checkpoints[-1]
    try:
        payload = torch.load(latest, map_location="cpu")
        epoch = int(payload.get("epoch", 0))
    except Exception:
        epoch = 0
    return latest, epoch
