from dataclasses import dataclass


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
