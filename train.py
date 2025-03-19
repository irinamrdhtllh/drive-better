import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from utils import collate_fn


def train(
    model,
    train_data,
    val_data,
    optimizer,
    lr_scheduler=None,
    num_epochs=100,
    device="cpu",
):
    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Define the model
    model.to(device)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for images, targets in train_loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            losses, _ = model(images, targets)
            train_loss = sum(l for l in losses.values())

            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validate the model
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [i.to(device) for i in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                losses, _ = model(images, targets)
                val_loss = sum(l for l in losses.values())

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if lr_scheduler:
            if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(total_val_loss)
            else:
                lr_scheduler.step()

        print(
            f"Epoch: [{epoch + 1}/{num_epochs}], Train Loss: {(avg_train_loss):.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"road_damage_detector.pth")
            print(
                f"Model saved at epoch {epoch + 1} (best validation loss: {best_val_loss:.4f})."
            )
