from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

from .preprocess import resolve_visdrone_root


def evaluate_tracking(
    gt_root: str,
    pred_dir: str,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> Dict[str, object]:
    """
    Description:
    Evaluate MOT predictions using MOTA, HOTA, IDSW, precision, and recall.
    Inputs:
    - gt_root: MOT ground-truth root path.
    - pred_dir: Directory with predicted track txt files.
    - iou_threshold: Minimum IoU for matching.
    - class_aware: Whether to enforce class-consistent matching.
    Outputs:
    - Dictionary with overall and per-sequence metrics.
    """
    gt_annotations_dir = _resolve_gt_annotations_dir(gt_root)
    pred_path = Path(pred_dir)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_path}")

    gt_files = sorted(gt_annotations_dir.glob("*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"No annotation files found under: {gt_annotations_dir}")

    per_sequence: Dict[str, Dict[str, float]] = {}

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_idsw = 0
    global_gt = 0

    global_pair_matches: DefaultDict[Tuple[str, int, int], int] = defaultdict(int)
    global_gt_matches: DefaultDict[Tuple[str, int], int] = defaultdict(int)
    global_pred_matches: DefaultDict[Tuple[str, int], int] = defaultdict(int)

    for gt_file in gt_files:
        sequence_name = gt_file.stem
        pred_file = pred_path / f"{sequence_name}.txt"

        gt_by_frame = _read_mot_file(gt_file, is_gt=True)
        pred_by_frame = _read_mot_file(pred_file, is_gt=False) if pred_file.exists() else defaultdict(list)

        tp, fp, fn, idsw, total_gt, pair_matches, gt_matches, pred_matches = _evaluate_sequence(
            sequence_name=sequence_name,
            gt_by_frame=gt_by_frame,
            pred_by_frame=pred_by_frame,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
        )

        metrics = _compute_metrics(tp=tp, fp=fp, fn=fn, idsw=idsw, total_gt=total_gt, pair_matches=pair_matches, gt_matches=gt_matches, pred_matches=pred_matches)
        per_sequence[sequence_name] = metrics

        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_idsw += idsw
        global_gt += total_gt

        for key, value in pair_matches.items():
            global_pair_matches[(sequence_name, key[0], key[1])] += value
        for key, value in gt_matches.items():
            global_gt_matches[(sequence_name, key)] += value
        for key, value in pred_matches.items():
            global_pred_matches[(sequence_name, key)] += value

    overall = _compute_metrics(
        tp=global_tp,
        fp=global_fp,
        fn=global_fn,
        idsw=global_idsw,
        total_gt=global_gt,
        pair_matches=global_pair_matches,
        gt_matches=global_gt_matches,
        pred_matches=global_pred_matches,
    )

    return {
        "overall": overall,
        "per_sequence": per_sequence,
    }


def _resolve_gt_annotations_dir(gt_root: str) -> Path:
    """
    Description:
    Resolve the annotations directory from MOT root variants.
    Inputs:
    - gt_root: Ground-truth root or annotations path.
    Outputs:
    - Path to annotations directory.
    """
    root_path = Path(gt_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Ground-truth path not found: {root_path}")

    if (root_path / "annotations").exists():
        return root_path / "annotations"

    try:
        resolved = resolve_visdrone_root(gt_root, required_subdirs=("annotations", "sequences"))
        return resolved / "annotations"
    except Exception:
        pass

    if root_path.name == "annotations":
        return root_path

    nested_annotations = root_path / root_path.name / "annotations"
    if nested_annotations.exists():
        return nested_annotations

    raise FileNotFoundError(f"Could not locate annotations directory from: {root_path}")


def _read_mot_file(file_path: Path, is_gt: bool) -> DefaultDict[int, List[Dict[str, float]]]:
    """
    Description:
    Read MOT-style txt annotations/predictions into frame-indexed records.
    Inputs:
    - file_path: MOT txt file path.
    - is_gt: Whether the file contains ground-truth entries.
    Outputs:
    - Frame-indexed dictionary of object dictionaries.
    """
    records: DefaultDict[int, List[Dict[str, float]]] = defaultdict(list)
    if not file_path.exists():
        return records

    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [token.strip() for token in line.split(",")]
        if len(parts) < 6:
            continue

        frame = int(float(parts[0]))
        obj_id = int(float(parts[1]))
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])

        if w <= 0 or h <= 0:
            continue

        score = float(parts[6]) if len(parts) > 6 else 1.0
        category = int(float(parts[7])) if len(parts) > 7 else -1

        if is_gt:
            if category == 0 or not (1 <= category <= 10):
                continue
            if score <= 0.0:
                continue

        bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
        records[frame].append(
            {
                "id": obj_id,
                "bbox": bbox,
                "score": score,
                "label": category,
            }
        )

    return records


def _evaluate_sequence(
    sequence_name: str,
    gt_by_frame: DefaultDict[int, List[Dict[str, float]]],
    pred_by_frame: DefaultDict[int, List[Dict[str, float]]],
    iou_threshold: float,
    class_aware: bool,
):
    """
    Description:
    Evaluate one MOT sequence and accumulate matching statistics.
    Inputs:
    - sequence_name: Sequence identifier.
    - gt_by_frame: Ground-truth frame dictionary.
    - pred_by_frame: Prediction frame dictionary.
    - iou_threshold: Minimum IoU for matching.
    - class_aware: Whether to enforce class-consistent matching.
    Outputs:
    - Tuple of TP/FP/FN/IDSW counts and association bookkeeping.
    """
    frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    tp = 0
    fp = 0
    fn = 0
    idsw = 0
    total_gt = 0

    pair_matches: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    gt_matches: DefaultDict[int, int] = defaultdict(int)
    pred_matches: DefaultDict[int, int] = defaultdict(int)

    prev_gt_to_pred: Dict[int, int] = {}
    prev_frame_for_gt: Dict[int, int] = {}

    for frame in frames:
        gt_objects = gt_by_frame.get(frame, [])
        pred_objects = pred_by_frame.get(frame, [])

        total_gt += len(gt_objects)

        matches, unmatched_gt, unmatched_pred = _match_objects(
            gt_objects,
            pred_objects,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
        )

        tp += len(matches)
        fn += len(unmatched_gt)
        fp += len(unmatched_pred)

        for gt_index, pred_index in matches:
            gt_obj = gt_objects[gt_index]
            pred_obj = pred_objects[pred_index]

            gt_id = int(gt_obj["id"])
            pred_id = int(pred_obj["id"])

            if gt_id in prev_gt_to_pred:
                prev_pred_id = prev_gt_to_pred[gt_id]
                prev_frame = prev_frame_for_gt.get(gt_id, frame - 1)
                if prev_pred_id != pred_id and (frame - prev_frame) <= 1:
                    idsw += 1

            prev_gt_to_pred[gt_id] = pred_id
            prev_frame_for_gt[gt_id] = frame

            pair_matches[(gt_id, pred_id)] += 1
            gt_matches[gt_id] += 1
            pred_matches[pred_id] += 1

    _ = sequence_name
    return tp, fp, fn, idsw, total_gt, pair_matches, gt_matches, pred_matches


def _match_objects(
    gt_objects: Sequence[Dict[str, float]],
    pred_objects: Sequence[Dict[str, float]],
    iou_threshold: float,
    class_aware: bool,
):
    """
    Description:
    Match GT and predicted objects in one frame.
    Inputs:
    - gt_objects: Ground-truth objects for a frame.
    - pred_objects: Predicted objects for a frame.
    - iou_threshold: Minimum IoU threshold.
    - class_aware: Whether to enforce class consistency.
    Outputs:
    - Tuple of match pairs, unmatched GT indices, unmatched pred indices.
    """
    if len(gt_objects) == 0:
        return [], [], list(range(len(pred_objects)))
    if len(pred_objects) == 0:
        return [], list(range(len(gt_objects))), []

    iou_matrix = _pairwise_iou(gt_objects, pred_objects)

    if class_aware:
        for gt_idx, gt_obj in enumerate(gt_objects):
            for pred_idx, pred_obj in enumerate(pred_objects):
                gt_label = int(gt_obj.get("label", -1))
                pred_label = int(pred_obj.get("label", -1))
                if gt_label != -1 and pred_label != -1 and gt_label != pred_label:
                    iou_matrix[gt_idx, pred_idx] = 0.0

    matches: List[Tuple[int, int]] = []
    unmatched_gt = set(range(len(gt_objects)))
    unmatched_pred = set(range(len(pred_objects)))

    if linear_sum_assignment is not None:
        row_indices, col_indices = linear_sum_assignment(1.0 - iou_matrix)
        for gt_idx, pred_idx in zip(row_indices, col_indices):
            if iou_matrix[gt_idx, pred_idx] < iou_threshold:
                continue
            matches.append((int(gt_idx), int(pred_idx)))
            unmatched_gt.discard(int(gt_idx))
            unmatched_pred.discard(int(pred_idx))
    else:
        while True:
            if iou_matrix.size == 0:
                break
            gt_idx, pred_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            best_iou = iou_matrix[gt_idx, pred_idx]
            if best_iou < iou_threshold:
                break
            matches.append((int(gt_idx), int(pred_idx)))
            unmatched_gt.discard(int(gt_idx))
            unmatched_pred.discard(int(pred_idx))
            iou_matrix[gt_idx, :] = -1.0
            iou_matrix[:, pred_idx] = -1.0

    return matches, sorted(unmatched_gt), sorted(unmatched_pred)


def _pairwise_iou(gt_objects: Sequence[Dict[str, float]], pred_objects: Sequence[Dict[str, float]]) -> np.ndarray:
    """
    Description:
    Compute pairwise IoU matrix between GT and predicted objects.
    Inputs:
    - gt_objects: Ground-truth objects.
    - pred_objects: Predicted objects.
    Outputs:
    - IoU matrix.
    """
    gt_boxes = np.asarray([obj["bbox"] for obj in gt_objects], dtype=np.float32)
    pred_boxes = np.asarray([obj["bbox"] for obj in pred_objects], dtype=np.float32)

    x1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    y1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    x2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    y2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])

    union = gt_area[:, None] + pred_area[None, :] - inter_area
    union = np.maximum(union, 1e-6)

    return inter_area / union


def _compute_metrics(
    tp: int,
    fp: int,
    fn: int,
    idsw: int,
    total_gt: int,
    pair_matches,
    gt_matches,
    pred_matches,
) -> Dict[str, float]:
    """
    Description:
    Compute sequence or global MOT metrics from aggregated counts.
    Inputs:
    - tp, fp, fn, idsw: Global matching error counts.
    - total_gt: Total ground-truth object count.
    - pair_matches, gt_matches, pred_matches: Association bookkeeping maps.
    Outputs:
    - Dictionary with MOTA, HOTA, IDSW, precision, recall, and raw counts.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    mota = 1.0 - ((fn + fp + idsw) / total_gt) if total_gt > 0 else 0.0

    det_a = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    ass_a = _association_accuracy(pair_matches=pair_matches, gt_matches=gt_matches, pred_matches=pred_matches)
    hota = math.sqrt(max(0.0, det_a * ass_a))

    return {
        "MOTA": float(mota),
        "HOTA": float(hota),
        "IDSW": float(idsw),
        "precision": float(precision),
        "recall": float(recall),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _association_accuracy(pair_matches, gt_matches, pred_matches) -> float:
    """
    Description:
    Compute association accuracy term used in HOTA approximation.
    Inputs:
    - pair_matches: Pair-wise match duration counts.
    - gt_matches: Match counts per GT id.
    - pred_matches: Match counts per predicted id.
    Outputs:
    - Association accuracy scalar.
    """
    weighted_sum = 0.0
    total_weight = 0.0

    for pair_key, pair_tp in pair_matches.items():
        if len(pair_key) == 3:
            _, gt_id, pred_id = pair_key
            gt_key = (pair_key[0], gt_id)
            pred_key = (pair_key[0], pred_id)
        else:
            gt_id, pred_id = pair_key
            gt_key = gt_id
            pred_key = pred_id

        fpa = pred_matches[pred_key] - pair_tp
        fna = gt_matches[gt_key] - pair_tp

        denom = pair_tp + fpa + fna
        ass_i = (pair_tp / denom) if denom > 0 else 0.0

        weighted_sum += ass_i * pair_tp
        total_weight += pair_tp

    if total_weight <= 0:
        return 0.0

    return weighted_sum / total_weight
