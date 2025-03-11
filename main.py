import xml.etree.ElementTree as ET
from dataclasses import dataclass

tree = ET.parse(
    ".\\dataset\\United_States\\train\\annotations\\xmls\\United_States_000000.xml"
)
root = tree.getroot()


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


def parse(root: ET.Element) -> Annotation:
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


annotation = parse(root)
print(annotation)
