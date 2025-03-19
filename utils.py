import cv2
import xml.etree.ElementTree as ET

from annotation import Annotation, Size, BoundingBox, AnnotationObject


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_xml_node(root: ET.Element, node_name: str):
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


def visualize_annotation(image_path: str, annotation: Annotation):
    image = cv2.imread(image_path)
    for o in annotation.objects:
        cv2.rectangle(
            img=image,
            pt1=(o.bndbox.xmin, o.bndbox.ymin),
            pt2=(o.bndbox.xmax, o.bndbox.ymax),
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.putText(
            img=image,
            text=o.name,
            org=(o.bndbox.xmin, o.bndbox.ymin - 5),
            color=(0, 255, 0),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=2,
        )

    cv2.imshow(annotation.filename, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
