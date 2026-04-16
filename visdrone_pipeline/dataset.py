from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .preprocess import resolve_visdrone_root

try:
    from pycocotools.coco import COCO
except ImportError as exc:  # pragma: no cover
    raise ImportError("pycocotools is required for VisDroneDataset.") from exc


class VisDroneDataset(Dataset):
    """
    Description:
    PyTorch dataset that loads VisDrone images and COCO-format annotations.
    Inputs:
    - images_dir: Directory containing split images.
    - annotation_json: COCO JSON path for the same split.
    - transforms: Optional callable that transforms image and target.
    Outputs:
    - Dataset instance yielding (image_tensor, target_dict).
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        annotation_json: Union[str, Path],
        transforms: Optional[Callable[[Image.Image, Dict[str, torch.Tensor]], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]] = None,
    ) -> None:
        """
        Description:
        Initialize dataset state and COCO annotation index.
        Inputs:
        - images_dir: Images directory path.
        - annotation_json: COCO annotation file path.
        - transforms: Optional transform callable.
        Outputs:
        - None.
        """
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")

        annotation_path = Path(annotation_json)
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation JSON does not exist: {annotation_path}")

        self.coco = COCO(str(annotation_path))
        self.image_ids = sorted(self.coco.getImgIds())
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Description:
        Return number of indexed images.
        Inputs:
        - None.
        Outputs:
        - Number of samples.
        """
        return len(self.image_ids)

    def __getitem__(self, index: int):
        """
        Description:
        Load one image and its COCO targets in torchvision detection format.
        Inputs:
        - index: Dataset sample index.
        Outputs:
        - Tuple[Tensor, Dict[str, Tensor]] for model consumption.
        """
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.images_dir / image_info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image file missing for id={image_id}: {image_path}")

        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        area = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            x1 = float(x)
            y1 = float(y)
            x2 = x1 + float(w)
            y2 = y1 + float(h)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]))
            area.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area_tensor = torch.tensor(area, dtype=torch.float32)
            iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target


def collate_fn(batch):
    """
    Description:
    Collate function for detection batches with variable annotation counts.
    Inputs:
    - batch: Sequence of dataset items.
    Outputs:
    - Tuple of image list and target list.
    """
    return tuple(zip(*batch))


def resolve_images_dir(split_dir: Union[str, Path]) -> Path:
    """
    Description:
    Resolve a VisDrone split root and return its images directory.
    Inputs:
    - split_dir: Path to split root, possibly nested.
    Outputs:
    - Path to resolved images directory.
    """
    split_root = resolve_visdrone_root(split_dir, required_subdirs=("images", "annotations"))
    images_dir = split_root / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory missing under resolved split root: {split_root}")
    return images_dir
