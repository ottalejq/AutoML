import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.dataset import TabularDataset


def train_model(
    model: nn.Module,
    numeric_data,
    categorical_data,
    target,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
):
    dataset = TabularDataset(
        numeric_data=numeric_data,
        categorical_data=categorical_data,
        target=target,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for batch in loader:
            numeric = batch["numeric"]
            categorical = batch["categorical"]
            target_batch = batch["target"]

            optimizer.zero_grad()

            predictions = model(
                numeric,
                categorical,
            )

            loss = loss_function(
                predictions,
                target_batch,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    return model