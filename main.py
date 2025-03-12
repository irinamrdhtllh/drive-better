import cv2
import xml.etree.ElementTree as ET
from dataclasses import dataclass


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
    size: tuple[int, int, int]
    segmented: bool
    objects: list[AnnotationObject]


def parse(annotation_path: str) -> Annotation:
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    filename = root.find("filename").text

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    depth = int(size.find("depth").text)
    size = [width, height, depth]

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


def visualize_annotation(image_path: str, annotation: Annotation):
    image = cv2.imread(image_path)
    for o in annotation.objects:
        bndbox = o.bndbox
        cv2.rectangle(
            img=image,
            pt1=(bndbox.xmin, bndbox.ymin),
            pt2=(bndbox.xmax, bndbox.ymax),
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.putText(
            img=image,
            text=o.name,
            org=(bndbox.xmin, bndbox.ymin - 5),
            color=(0, 255, 0),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=2,
        )

    cv2.imshow(annotation.filename, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "./dataset/United_States/train/images/United_States_000001.jpg"
    annotation_path = (
        "./dataset/United_States/train/annotations/xmls/United_States_000001.xml"
    )

    annotation = parse(annotation_path)
    print(annotation)

    visualize_annotation(image_path, annotation)
