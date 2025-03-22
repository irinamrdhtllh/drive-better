import torch

from tqdm import tqdm


def test(model, test_data, device="cpu"):
    model.load_state_dict(torch.load("./road_damage_detector.pth", map_location=device))
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
