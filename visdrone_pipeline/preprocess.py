import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

from PIL import Image

VISDRONE_CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "people"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
]


class ConversionError(RuntimeError):
    """
    Description:
    Raised when VisDrone annotation conversion cannot proceed safely.
    Inputs:
    - Runtime message string.
    Outputs:
    - ConversionError exception instance.
    """

    pass


def _resolve_visdrone_root(dataset_dir: Union[str, Path], required_subdirs: Iterable[str]) -> Path:
    """
    Description:
    Resolve a VisDrone split root even when it is nested one level deeper.
    Inputs:
    - dataset_dir: Candidate split directory path.
    - required_subdirs: Subdirectories that must exist under resolved root.
    Outputs:
    - Resolved split root Path.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    required_subdirs = tuple(required_subdirs)

    def has_required(path: Path) -> bool:
        return all((path / subdir).exists() for subdir in required_subdirs)

    if has_required(dataset_path):
        return dataset_path

    nested = dataset_path / dataset_path.name
    if nested.exists() and has_required(nested):
        return nested

    candidates = [child for child in dataset_path.iterdir() if child.is_dir() and has_required(child)]
    if len(candidates) == 1:
        return candidates[0]

    required_txt = ", ".join(required_subdirs)
    raise ConversionError(
        f"Could not resolve VisDrone root under {dataset_path}. "
        f"Expected subfolders: {required_txt}."
    )


def _list_images(images_dir: Path) -> List[Path]:
    """
    Description:
    Collect image files from a VisDrone images directory.
    Inputs:
    - images_dir: Directory containing image files.
    Outputs:
    - Sorted list of image paths.
    """
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in valid_ext)


def _parse_visdrone_line(line: str) -> Tuple[float, float, float, float, int, int, int]:
    """
    Description:
    Parse one VisDrone DET annotation row.
    Inputs:
    - line: Raw comma-separated annotation row.
    Outputs:
    - Parsed tuple (x, y, w, h, score, cls, truncation, occlusion).
    """
    parts = [token.strip() for token in line.split(",")]
    if len(parts) < 8:
        raise ValueError("Invalid VisDrone annotation line (expected at least 8 fields)")

    x = float(parts[0])
    y = float(parts[1])
    w = float(parts[2])
    h = float(parts[3])
    score = int(float(parts[4]))
    cls = int(float(parts[5]))
    truncation = int(float(parts[6]))
    occlusion = int(float(parts[7]))
    return x, y, w, h, score, cls, truncation, occlusion


def _clamp_bbox(x: float, y: float, w: float, h: float, image_w: int, image_h: int) -> Tuple[float, float, float, float]:
    """
    Description:
    Clamp a bounding box to valid image bounds.
    Inputs:
    - x, y, w, h: Box values in xywh format.
    - image_w, image_h: Image width and height.
    Outputs:
    - Clamped xywh tuple.
    """
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_w), x + w)
    y2 = min(float(image_h), y + h)
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def _convert_split(split_dir: Path, output_json: Path) -> Dict[str, int]:
    """
    Description:
    Convert one VisDrone DET split to COCO JSON format.
    Inputs:
    - split_dir: Resolved split directory containing images and annotations.
    - output_json: Output JSON file path.
    Outputs:
    - Conversion summary counts for the split.
    """
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"

    image_files = _list_images(images_dir)
    if not image_files:
        raise ConversionError(f"No images found in {images_dir}")

    coco = {
        "images": [],
        "annotations": [],
        "categories": VISDRONE_CATEGORIES,
    }

    image_id_map: Dict[str, int] = {}
    annotation_id = 1
    ignored_missing_ann = 0
    ignored_invalid_bbox = 0
    ignored_class_or_region = 0

    for image_id, image_path in enumerate(image_files, start=1):
        image_name = image_path.name
        stem = image_path.stem

        with Image.open(image_path) as image:
            width, height = image.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        )
        image_id_map[stem] = image_id

        ann_path = annotations_dir / f"{stem}.txt"
        if not ann_path.exists():
            ignored_missing_ann += 1
            continue

        for raw_line in ann_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue

            try:
                x, y, w, h, _, cls, truncation, occlusion = _parse_visdrone_line(raw_line)
            except ValueError:
                ignored_invalid_bbox += 1
                continue

            # Ignore regions (class 0) and keep only classes 1..10.
            if cls == 0 or not (1 <= cls <= 10):
                ignored_class_or_region += 1
                continue

            if w <= 0 or h <= 0:
                ignored_invalid_bbox += 1
                continue

            x, y, w, h = _clamp_bbox(x, y, w, h, width, height)
            if w <= 0 or h <= 0:
                ignored_invalid_bbox += 1
                continue

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "truncation": truncation,
                    "occlusion": occlusion,
                }
            )
            annotation_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco), encoding="utf-8")

    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "missing_annotation_files": ignored_missing_ann,
        "ignored_invalid_bbox": ignored_invalid_bbox,
        "ignored_class_or_region": ignored_class_or_region,
    }


def convert_visdrone_to_coco(
    det_train_dir: Union[str, Path],
    det_val_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Dict[str, int]]:
    """
    Description:
    Convert VisDrone DET train and val annotations into COCO JSON files.
    Inputs:
    - det_train_dir: DET train split path.
    - det_val_dir: DET val split path.
    - output_dir: Destination directory for train.json and val.json.
    Outputs:
    - Dictionary containing conversion statistics and output JSON paths.
    """
    output_path = Path(output_dir)

    train_root = _resolve_visdrone_root(det_train_dir, required_subdirs=("images", "annotations"))
    val_root = _resolve_visdrone_root(det_val_dir, required_subdirs=("images", "annotations"))

    train_json = output_path / "train.json"
    val_json = output_path / "val.json"

    train_stats = _convert_split(train_root, train_json)
    val_stats = _convert_split(val_root, val_json)

    return {
        "train": train_stats,
        "val": val_stats,
        "train_json": {"path": str(train_json)},
        "val_json": {"path": str(val_json)},
    }


def resolve_visdrone_root(dataset_dir: Union[str, Path], required_subdirs: Iterable[str]) -> Path:
    """
    Description:
    Public wrapper for robust VisDrone split root resolution.
    Inputs:
    - dataset_dir: Candidate split directory path.
    - required_subdirs: Required child directories under resolved root.
    Outputs:
    - Resolved split root Path.
    """
    return _resolve_visdrone_root(dataset_dir, required_subdirs=required_subdirs)
