import torch
import torch.optim as optim

from torch.utils.data import random_split
from torch.utils.data import DataLoader

from dataset import RoadDamageDataset
from model import Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the train and validation dataset
# czech_data = RoadDamageDataset(dir="./dataset/czech", split="train")
# india_data = RoadDamageDataset(dir="./dataset/india", split="train")
# japan_data = RoadDamageDataset(dir="./dataset/japan", split="train")
# norway_data = RoadDamageDataset(dir="./dataset/norway", split="train")
# us_data = RoadDamageDataset(dir="./dataset/united_states", split="train")

data = RoadDamageDataset(dir="./dataset/united_states", split="train")
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_data, val_data = random_split(data, [train_size, val_size])

test_data = RoadDamageDataset(dir="./dataset/united_states", split="test")

train_loader = DataLoader(
    train_data,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

val_loader = DataLoader(
    val_data,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

# Define the model
model = Model(num_classes=5)  # 4 road damage classes and 1 background
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for images, targets in train_loader:
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        losses, detections = model(images, targets)
        loss = sum(l for l in losses.values())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    # Validate the model
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses, detections = model(images, targets)
            loss = sum(l for l in losses.values())

            total_val_loss += loss.item()

    lr_scheduler.step()

    print(
        f"Epoch: [{epoch}/{num_epochs}], Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}"
    )

    torch.save(model.state_dict(), f"road_damage_detector.pth")
