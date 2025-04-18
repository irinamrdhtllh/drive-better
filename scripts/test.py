import torch

from tqdm import tqdm
from typing import Optional, Dict, List, Tuple

from models.model import FasterRCNN_ResNet50, YOLO11


def test(
    model: torch.nn.Module,
    test_data: List[Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]],
    device: str = "cpu",
) -> Tuple[List[Dict[str, torch.Tensor]], Optional[List[Dict[str, torch.Tensor]]]]:
    if isinstance(model, FasterRCNN_ResNet50):
        model.load_state_dict(
            torch.load("./road_damage_detector.pth", map_location=device)
        )
        model.to(device)
        model.eval()

        predictions = []
        targets = []
        with torch.no_grad():
            for i in tqdm(range(len(test_data)), desc="Testing"):
                image, target = test_data[i]
                image = image.to(device)
                _, prediction = model([image])
                predictions.append(prediction[0])

                if target is not None:
                    targets.append(target)

        return predictions, targets if targets else None
    elif isinstance(model, YOLO11):
        results = model.predict(test_data)
        return results
