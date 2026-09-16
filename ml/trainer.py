import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.dataset import TabularDataset


def train_model(
    model: nn.Module,
    prepared_data: dict,
    task_type: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    dataset = TabularDataset(
        numeric_data=prepared_data["numeric_data"],
        categorical_data=prepared_data["categorical_data"],
        target=prepared_data["target"],
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    if task_type == "regression":
        loss_function = nn.MSELoss()

    elif task_type == "binary_classification":
        loss_function = nn.BCEWithLogitsLoss()

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in loader:
            numeric = batch["numeric"]
            categorical = batch["categorical"]
            target = batch["target"]

            optimizer.zero_grad()

            predictions = model(
                numeric,
                categorical,
            )

            loss = loss_function(
                predictions,
                target,
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