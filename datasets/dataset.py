import os
import cv2
import torch

from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple

from utils import parse_annotation, to_tensor


class RoadDamageDataset(Dataset):
    def __init__(self, dir: str, split: str = "train"):
        assert split in ["train", "val", "test"]
        self.dir = dir
        self.image_dir = os.path.join(dir, split, "images")
        self.annotation_dir = (
            os.path.join(dir, split, "annotations/xmls")
            if split == "train" or split == "val"
            else None
        )
        self.images = os.listdir(self.image_dir)
        self.annotations = {}
        for image_filename in self.images:
            annotation_path = os.path.join(
                self.annotation_dir, image_filename.replace(".jpg", ".xml")
            )
            annotation = parse_annotation(annotation_path)
            self.annotations[image_filename] = annotation

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image_filename = self.images[index]
        image_path = os.path.join(self.image_dir, image_filename)

        image = cv2.imread(image_path)

        if self.annotation_dir is None:
            return to_tensor(image, None)

        return to_tensor(image, self.annotations[image_filename])

    def __len__(self) -> int:
        return len(self.images)
