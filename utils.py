import cv2

from dataset import Annotation


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
