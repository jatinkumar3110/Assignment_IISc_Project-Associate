from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from .preprocess import resolve_visdrone_root

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


@dataclass
class ByteTrackParams:
    """
    Description:
    Runtime parameters for simplified ByteTrack-style IoU tracking.
    Inputs:
    - Threshold and lifecycle fields supplied at construction.
    Outputs:
    - Parameter object consumed by ByteTrackTracker.
    """

    high_thresh: float = 0.6
    low_thresh: float = 0.2
    match_thresh: float = 0.7
    new_track_thresh: float = 0.7
    max_time_lost: int = 30
    min_hits: int = 3


class ByteTrackTracker:
    """
    Description:
    Simplified ByteTrack-style tracker using score-split IoU association.
    Inputs:
    - params: Optional ByteTrackParams object.
    Outputs:
    - Stateful tracker that emits active tracks per frame.
    """

    def __init__(self, params: Optional[ByteTrackParams] = None) -> None:
        """
        Description:
        Initialize tracker state.
        Inputs:
        - params: Optional tracking configuration.
        Outputs:
        - None.
        """
        self.params = params or ByteTrackParams()
        self.tracks: List[Dict[str, object]] = []
        self.next_track_id = 1
        self.frame_id = 0

    def reset(self) -> None:
        """
        Description:
        Reset all active tracks and frame state.
        Inputs:
        - None.
        Outputs:
        - None.
        """
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_id = 0

    def update(self, detections: Sequence[Dict[str, object]], frame_id: Optional[int] = None) -> List[Dict[str, object]]:
        """
        Description:
        Update tracker with one frame of detections.
        Inputs:
        - detections: Detection records containing bbox, score, and label.
        - frame_id: Optional explicit frame id.
        Outputs:
        - List of active tracks emitted for this frame.
        """
        if frame_id is None:
            self.frame_id += 1
        else:
            self.frame_id = frame_id

        high_dets = [det for det in detections if float(det["score"]) >= self.params.high_thresh]
        low_dets = [det for det in detections if self.params.low_thresh <= float(det["score"]) < self.params.high_thresh]

        for track in self.tracks:
            track["age"] = int(track["age"]) + 1
            track["time_since_update"] = int(track["time_since_update"]) + 1

        first_matches, unmatched_track_indices, unmatched_high_indices = _associate(
            self.tracks,
            high_dets,
            match_thresh=self.params.match_thresh,
        )
        self._apply_matches(first_matches, high_dets)

        remaining_tracks = [self.tracks[index] for index in unmatched_track_indices]
        second_matches_local, remaining_unmatched_track_local, _ = _associate(
            remaining_tracks,
            low_dets,
            match_thresh=self.params.match_thresh,
        )

        second_matches = [(unmatched_track_indices[t_idx], d_idx) for t_idx, d_idx in second_matches_local]
        self._apply_matches(second_matches, low_dets)

        unmatched_after_second = [unmatched_track_indices[idx] for idx in remaining_unmatched_track_local]
        self._prune_lost_tracks(unmatched_after_second)

        for det_index in unmatched_high_indices:
            det = high_dets[det_index]
            if float(det["score"]) < self.params.new_track_thresh:
                continue
            self._start_track(det)

        active_outputs = []
        for track in self.tracks:
            if int(track["time_since_update"]) == 0:
                enough_hits = int(track["hits"]) >= self.params.min_hits or self.frame_id <= self.params.min_hits
                if enough_hits:
                    active_outputs.append(
                        {
                            "frame": self.frame_id,
                            "track_id": int(track["track_id"]),
                            "bbox": track["bbox"],
                            "score": float(track["score"]),
                            "label": int(track["label"]),
                        }
                    )

        return active_outputs

    def _apply_matches(self, matches: Sequence[Tuple[int, int]], detections: Sequence[Dict[str, object]]) -> None:
        """
        Description:
        Apply matched detections to track state.
        Inputs:
        - matches: Matched (track_idx, det_idx) pairs.
        - detections: Detection list used in matching.
        Outputs:
        - None.
        """
        for track_idx, det_idx in matches:
            detection = detections[det_idx]
            track = self.tracks[track_idx]
            track["bbox"] = np.asarray(detection["bbox"], dtype=np.float32)
            track["score"] = float(detection["score"])
            track["label"] = int(detection["label"])
            track["hits"] = int(track["hits"]) + 1
            track["time_since_update"] = 0

    def _prune_lost_tracks(self, unmatched_track_indices: Sequence[int]) -> None:
        """
        Description:
        Remove tracks that exceeded max_time_lost.
        Inputs:
        - unmatched_track_indices: Unmatched track indices for current frame.
        Outputs:
        - None.
        """
        _ = unmatched_track_indices
        self.tracks = [
            track
            for track in self.tracks
            if int(track["time_since_update"]) <= self.params.max_time_lost
        ]

    def _start_track(self, detection: Dict[str, object]) -> None:
        """
        Description:
        Start a new track from an unmatched high-confidence detection.
        Inputs:
        - detection: Detection dictionary with bbox, score, and label.
        Outputs:
        - None.
        """
        self.tracks.append(
            {
                "track_id": self.next_track_id,
                "bbox": np.asarray(detection["bbox"], dtype=np.float32),
                "score": float(detection["score"]),
                "label": int(detection["label"]),
                "hits": 1,
                "age": 1,
                "time_since_update": 0,
            }
        )
        self.next_track_id += 1


def _associate(
    tracks: Sequence[Dict[str, object]],
    detections: Sequence[Dict[str, object]],
    match_thresh: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Description:
    Match detections to tracks by IoU and class consistency.
    Inputs:
    - tracks: Active track records.
    - detections: Candidate detections.
    - match_thresh: Minimum IoU threshold for a match.
    Outputs:
    - Tuple of matches, unmatched track indices, unmatched detection indices.
    """
    if len(tracks) == 0:
        return [], [], list(range(len(detections)))
    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    iou_matrix = _compute_iou_matrix(tracks, detections)
    cost_matrix = 1.0 - iou_matrix

    matches: List[Tuple[int, int]] = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))

    if linear_sum_assignment is not None:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        for track_idx, det_idx in zip(row_indices, col_indices):
            iou = iou_matrix[track_idx, det_idx]
            if iou < match_thresh:
                continue
            if int(tracks[track_idx]["label"]) != int(detections[det_idx]["label"]):
                continue
            matches.append((track_idx, det_idx))
            unmatched_tracks.discard(track_idx)
            unmatched_detections.discard(det_idx)
    else:
        # Greedy fallback if scipy is unavailable.
        while True:
            if iou_matrix.size == 0:
                break
            track_idx, det_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            best_iou = iou_matrix[track_idx, det_idx]
            if best_iou < match_thresh:
                break
            if int(tracks[track_idx]["label"]) != int(detections[det_idx]["label"]):
                iou_matrix[track_idx, det_idx] = -1.0
                continue
            matches.append((int(track_idx), int(det_idx)))
            unmatched_tracks.discard(int(track_idx))
            unmatched_detections.discard(int(det_idx))
            iou_matrix[track_idx, :] = -1.0
            iou_matrix[:, det_idx] = -1.0

    return matches, sorted(unmatched_tracks), sorted(unmatched_detections)


def _compute_iou_matrix(
    tracks: Sequence[Dict[str, object]],
    detections: Sequence[Dict[str, object]],
) -> np.ndarray:
    """
    Description:
    Compute pairwise IoU matrix between tracks and detections.
    Inputs:
    - tracks: Active track records with bbox fields.
    - detections: Detection records with bbox fields.
    Outputs:
    - IoU matrix of shape [num_tracks, num_detections].
    """
    track_boxes = np.asarray([track["bbox"] for track in tracks], dtype=np.float32)
    det_boxes = np.asarray([det["bbox"] for det in detections], dtype=np.float32)

    if track_boxes.size == 0 or det_boxes.size == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    x1 = np.maximum(track_boxes[:, None, 0], det_boxes[None, :, 0])
    y1 = np.maximum(track_boxes[:, None, 1], det_boxes[None, :, 1])
    x2 = np.minimum(track_boxes[:, None, 2], det_boxes[None, :, 2])
    y2 = np.minimum(track_boxes[:, None, 3], det_boxes[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    track_area = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
    det_area = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])

    union = track_area[:, None] + det_area[None, :] - intersection
    union = np.maximum(union, 1e-6)

    return intersection / union


def bytetrack_update(
    tracker: ByteTrackTracker,
    detections: Sequence[Dict[str, object]],
    frame_id: Optional[int] = None,
) -> List[Dict[str, object]]:
    """
    Description:
    Stateless wrapper for ByteTrackTracker.update.
    Inputs:
    - tracker: ByteTrackTracker instance.
    - detections: Current frame detections.
    - frame_id: Optional frame id.
    Outputs:
    - Active tracks for the frame.
    """
    return tracker.update(detections=detections, frame_id=frame_id)


def run_inference_on_mot(
    model: torch.nn.Module,
    mot_root: str,
    device: torch.device,
    output_dir: Optional[str] = None,
    score_thresh: float = 0.3,
    assoc_min_score: float = 0.4,
    max_frames: Optional[int] = None,
    tracker_params: Optional[ByteTrackParams] = None,
    use_tracking: bool = True,
) -> Dict[str, Dict[str, object]]:
    """
    Description:
    Run frame-wise detector inference on MOT sequences and optional ByteTrack-style association.
    Inputs:
    - model: Detector model.
    - mot_root: MOT split root path.
    - device: Execution device.
    - output_dir: Optional output directory for detection/track files.
    - score_thresh: Detection confidence threshold for exported detections.
    - assoc_min_score: Minimum confidence threshold used for tracker association.
    - max_frames: Optional frame cap per sequence.
    - tracker_params: Optional tracker configuration.
    - use_tracking: Whether to generate track files.
    Outputs:
    - Per-sequence summary dictionary.
    """
    model.eval()
    resolved_mot_root = resolve_visdrone_root(mot_root, required_subdirs=("sequences", "annotations"))

    sequences_dir = resolved_mot_root / "sequences"
    output_path = Path(output_dir) if output_dir is not None else None

    if output_path is not None:
        (output_path / "detections").mkdir(parents=True, exist_ok=True)
        if use_tracking:
            (output_path / "tracks").mkdir(parents=True, exist_ok=True)

    sequence_outputs: Dict[str, Dict[str, object]] = {}

    sequence_dirs = sorted(path for path in sequences_dir.iterdir() if path.is_dir())

    with torch.no_grad():
        for sequence_dir in sequence_dirs:
            sequence_name = sequence_dir.name
            frame_dir = sequence_dir / "img1"
            if not frame_dir.exists():
                frame_dir = sequence_dir

            frame_paths = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if max_frames is not None:
                frame_paths = frame_paths[:max_frames]

            tracker = ByteTrackTracker(params=tracker_params) if use_tracking else None

            sequence_detections_count = 0
            sequence_tracks: List[str] = []
            sequence_det_lines: List[str] = []
            total_tracks_this_seq = 0

            for frame_path in frame_paths:
                frame_id = _parse_frame_id(frame_path.stem)

                image = Image.open(frame_path).convert("RGB")
                image_tensor = F.to_tensor(image).to(device)
                output = model([image_tensor])[0]

                detections = _extract_frame_detections(output, score_thresh=score_thresh)
                sequence_detections_count += len(detections)
                association_detections = [det for det in detections if float(det["score"]) >= assoc_min_score]

                for det in detections:
                    x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    sequence_det_lines.append(
                        f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(det['score']):.6f},{int(det['label'])},-1,-1"
                    )

                if tracker is not None:
                    active_tracks = bytetrack_update(tracker, detections=association_detections, frame_id=frame_id)
                    total_tracks_this_seq += len(active_tracks)
                    for track in active_tracks:
                        x1, y1, x2, y2 = [float(v) for v in track["bbox"]]
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        sequence_tracks.append(
                            f"{frame_id},{int(track['track_id'])},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(track['score']):.6f},{int(track['label'])},-1,-1"
                        )

            sequence_outputs[sequence_name] = {
                "num_frames": len(frame_paths),
                "num_detection_rows": len(sequence_det_lines),
                "num_track_rows": len(sequence_tracks),
                "tracks_generated": total_tracks_this_seq,
            }

            if output_path is not None:
                det_file = output_path / "detections" / f"{sequence_name}.txt"
                det_file.write_text("\n".join(sequence_det_lines), encoding="utf-8")

                if use_tracking:
                    track_file = output_path / "tracks" / f"{sequence_name}.txt"
                    track_file.write_text("\n".join(sequence_tracks), encoding="utf-8")

            print(
                f"[MOT inference] seq={sequence_name} frames={len(frame_paths)} "
                f"detections={sequence_detections_count} tracks={total_tracks_this_seq}"
            )

    return sequence_outputs


def _extract_frame_detections(output: Dict[str, torch.Tensor], score_thresh: float) -> List[Dict[str, object]]:
    """
    Description:
    Convert one detector output into tracking-ready detection dictionaries.
    Inputs:
    - output: Model output dictionary for one frame.
    - score_thresh: Minimum confidence score.
    Outputs:
    - List of detection dictionaries.
    """
    detections: List[Dict[str, object]] = []
    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if float(score) < score_thresh:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "label": int(label),
            }
        )
    return detections


def _parse_frame_id(stem: str) -> int:
    """
    Description:
    Parse frame id integer from an MOT frame filename stem.
    Inputs:
    - stem: Filename stem.
    Outputs:
    - Parsed frame id.
    """
    if stem.isdigit():
        return int(stem)

    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits)

    raise ValueError(f"Cannot parse frame id from name: {stem}")
