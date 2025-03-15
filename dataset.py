import os
import cv2
import torch
import torchvision.transforms as T
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class Size:
    width: int
    height: int
    depth: int


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class AnnotationObject:
    name: str
    pose: str
    truncated: bool
    difficult: bool
    bndbox: BoundingBox


@dataclass
class Annotation:
    filename: str
    size: Size
    segmented: bool
    objects: list[AnnotationObject]


class RoadDamageDataset(Dataset):
    def __init__(self, dir: str, split: str = "train"):
        assert split in ["train", "test"]
        self.dir = dir
        self.image_dir = os.path.join(dir, split, "images")
        self.annotation_dir = (
            os.path.join(dir, split, "annotations/xmls") if split == "train" else None
        )
        self.images = os.listdir(self.image_dir)

    def parse_annotation(self, path: str) -> Annotation:
        tree = ET.parse(path)
        root = tree.getroot()

        filename = root.find("filename").text

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        depth = int(size.find("depth").text)
        size = Size(width, height, depth)

        segmented = False if root.find("segmented").text == "0" else True

        objects = []
        for o in root.findall("object"):
            name = o.find("name").text
            pose = o.find("pose").text
            truncated = False if o.find("truncated").text == "0" else True
            difficult = False if o.find("difficult").text == "0" else True

            bndbox = o.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            bndbox = BoundingBox(xmin, ymin, xmax, ymax)

            objects.append(AnnotationObject(name, pose, truncated, difficult, bndbox))

        annotation = Annotation(filename, size, segmented, objects)

        return annotation

    def class_to_label(self, class_name: str) -> int:
        class_map = {
            "D00": 1,
            "D10": 2,
            "D20": 3,
            "D40": 4,
        }
        return class_map.get(class_name, 0)

    def to_tensor(self, image, annotation: Annotation) -> tuple[torch.Tensor, dict]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = T.Compose([T.ToPILImage(), T.ToTensor()])
        image = transform(image)

        if annotation is None:
            return image, None

        boxes = []
        labels = []

        for o in annotation.objects:
            boxes.append([o.bndbox.xmin, o.bndbox.ymin, o.bndbox.xmax, o.bndbox.ymax])
            labels.append(self.class_to_label(o.name))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return image, target

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        image_filename = self.images[index]
        image_path = os.path.join(self.image_dir, image_filename)

        image = cv2.imread(image_path)

        if self.annotation_dir is None:
            return self.to_tensor(image, None)

        annotation_path = os.path.join(
            self.annotation_dir, image_filename.replace(".jpg", ".xml")
        )
        annotation = self.parse_annotation(annotation_path)

        return self.to_tensor(image, annotation)

    def __len__(self):
        return len(self.images)
