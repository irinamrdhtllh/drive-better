import os
import random
import shutil
import cv2
import torch
import torchvision.transforms as T
import xml.etree.ElementTree as ET

from typing import Optional, Dict, Tuple

from datasets.annotation import Annotation, Size, BoundingBox, AnnotationObject


def collate_fn(batch):
    return tuple(zip(*batch))


def class_to_label(class_name: str) -> int:
    class_map = {
        "D00": 0,  # Longitudinal crack
        "D10": 1,  # Transverse crack
        "D20": 2,  # Alligator crack
        "D40": 3,  # Pothole
    }
    return class_map.get(class_name, 4)


def label_to_class(label: int) -> Optional[str]:
    class_map = {
        0: "D00",
        1: "D10",
        2: "D20",
        3: "D40",
    }
    return class_map.get(label)


def train_val_split(dir: str, val_ratio: int = 0.2):
    train_images_dir = os.path.join(dir, "train/images")
    train_annotations_dir = os.path.join(dir, "train/annotations/xmls")
    val_images_dir = os.path.join(dir, "val/images")
    val_annotations_dir = os.path.join(dir, "val/annotations/xmls")

    os.makedirs(val_images_dir)
    os.makedirs(val_annotations_dir)

    images = [file for file in os.listdir(train_images_dir)]
    num_val_images = int(len(images) * val_ratio)
    val_images = random.sample(images, num_val_images)

    for image in val_images:
        image_path = os.path.join(train_images_dir, image)
        annotation_path = os.path.join(
            train_annotations_dir, image.replace(".jpg", ".xml")
        )
        shutil.move(
            image_path,
            os.path.join(val_images_dir, image),
        )
        shutil.move(
            annotation_path,
            os.path.join(val_annotations_dir, os.path.basename(annotation_path)),
        )


def parse_xml_node(root: ET.Element, node_name: str) -> Optional[str]:
    element = root.find(node_name)
    if element:
        return element.text
    return None


def parse_annotation(path: str) -> Annotation:
    tree = ET.parse(path)
    root = tree.getroot()

    filename = root.find("filename").text

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    depth = int(size.find("depth").text) if size.find("depth") else None
    size = Size(width, height, depth)

    segmented = False if parse_xml_node(root, "segmented") == "0" else True

    objects = []
    for o in root.findall("object"):
        name = o.find("name").text
        pose = parse_xml_node(root, "pose")
        truncated = False if parse_xml_node(root, "truncated") == "0" else True
        difficult = False if parse_xml_node(root, "difficult") == "0" else True

        bndbox = o.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        if xmax - xmin < 1e-9 or ymax - ymin < 1e-9:
            continue
        bndbox = BoundingBox(xmin, ymin, xmax, ymax)

        objects.append(AnnotationObject(name, pose, truncated, difficult, bndbox))

    annotation = Annotation(filename, size, segmented, objects)

    return annotation


def xml_to_yolotxt(dir: str):
    xmls_dir = os.path.join(dir, "annotations/xmls/")
    labels_dir = os.path.join(dir, "labels/")
    os.makedirs(labels_dir, exist_ok=True)

    for xml_file in os.listdir(xmls_dir):
        if not xml_file.endswith(".xml"):
            continue

        labels = []

        annotation = parse_annotation(os.path.join(xmls_dir, xml_file))

        image_width = annotation.size.width
        image_height = annotation.size.height

        for o in annotation.objects:
            name = class_to_label(o.name)

            xmin = o.bndbox.xmin
            ymin = o.bndbox.ymin
            xmax = o.bndbox.xmax
            ymax = o.bndbox.ymax

            x_center = (xmin + xmax) / (2 * image_width)
            y_center = (ymin + ymax) / (2 * image_height)
            box_width = (xmax - xmin) / image_width
            box_height = (ymax - ymin) / image_height

            labels.append(f"{name} {x_center} {y_center} {box_width} {box_height}")

        txt_filename = os.path.join(labels_dir, xml_file.replace(".xml", ".txt"))
        with open(txt_filename, "w") as f:
            f.write("\n".join(labels))


def to_tensor(
    image, annotation: Optional[Annotation]
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.ToTensor(),
        ]
    )
    image = transform(image)

    if annotation is None:
        return image, None

    boxes = []
    labels = []

    if annotation.objects:
        for o in annotation.objects:
            boxes.append([o.bndbox.xmin, o.bndbox.ymin, o.bndbox.xmax, o.bndbox.ymax])
            labels.append(class_to_label(o.name))
    boxes = (
        torch.tensor(boxes, dtype=torch.float32)
        if boxes
        else torch.zeros((0, 4), dtype=torch.float32)
    )
    labels = (
        torch.tensor(labels, dtype=torch.int64)
        if labels
        else torch.zeros((0,), dtype=torch.int64)
    )

    target = {
        "boxes": boxes,
        "labels": labels,
    }

    return image, target


def visualize_boxes(
    image_path: str,
    target: Optional[Dict[str, torch.Tensor]] = None,
    prediction: Optional[Dict[str, torch.Tensor]] = None,
):
    if target is None and prediction is None:
        raise ValueError("At least one of 'target' or 'prediction' must not be None.")

    image = cv2.imread(image_path)
    if target is not None:
        for i, box in enumerate(target["boxes"]):
            cv2.rectangle(
                img=image,
                pt1=(int(box[0]), int(box[1])),  # xmin, ymin
                pt2=(int(box[2]), int(box[3])),  # xmax, ymax
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.putText(
                img=image,
                text=label_to_class(target["labels"][i].item()),
                org=(int(box[0]), int(box[1]) - 5),
                color=(0, 255, 0),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=2,
            )

    if prediction is not None:
        for i, box in enumerate(prediction["boxes"]):
            cv2.rectangle(
                img=image,
                pt1=(int(box[0]), int(box[1])),  # xmin, ymin
                pt2=(int(box[2]), int(box[3])),  # xmax, ymax
                color=(225, 0, 0),
                thickness=2,
            )
            cv2.putText(
                img=image,
                text=label_to_class(prediction["labels"][i].item()),
                org=(int(box[0]), int(box[1]) - 5),
                color=(255, 0, 0),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=2,
            )

    cv2.imshow(os.path.basename(image_path), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
