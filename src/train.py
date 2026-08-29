import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import get_model
import os


def load_config(config_path: str) -> dict:
    """
    Load training configuration from a YAML file.
    """
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Returns:
        average loss
        training accuracy
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item()
            * inputs.size(0)
        )

        predicted = outputs.argmax(dim=1)

        total += targets.size(0)

        correct += (
            predicted.eq(targets)
            .sum()
            .item()
        )

        # Log training progress every 100 batches
        if batch_idx % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "training_progress",
                        "batch": batch_idx,
                        "total_batches": len(loader),
                    }
                ),
                flush=True,
            )

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on the validation dataset.

    Returns:
        validation loss
        validation accuracy
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        total_loss += (
            loss.item()
            * inputs.size(0)
        )

        predicted = outputs.argmax(dim=1)

        total += targets.size(0)

        correct += (
            predicted.eq(targets)
            .sum()
            .item()
        )

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    # ---------------------------------------------------------
    # Locate configuration file
    # ---------------------------------------------------------

    torch_threads = int(os.getenv("TORCH_NUM_THREADS", "2"))

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(torch_threads)

    # Docker path
    config_path = Path(
        os.getenv(
            "TRAINING_CONFIG",
            "configs/training_config.yaml",
        )
    )

    # Local development path
    if not config_path.exists():
        config_path = Path(
            "configs/training_config.yaml"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Training configuration not found: {config_path}"
    )

    config = load_config(
        str(config_path)
    )

    # ---------------------------------------------------------
    # Read model configuration
    # ---------------------------------------------------------

    architecture = (
        config["model"]["architecture"]
    )

    num_classes = (
        config["model"]["num_classes"]
    )

    # ---------------------------------------------------------
    # Read dataset configuration
    # ---------------------------------------------------------

    data_dir = (
        config["data"]["data_dir"]
    )

    # ---------------------------------------------------------
    # Read ALL training hyperparameters from YAML
    # ---------------------------------------------------------

    batch_size = (
        config["training"]["batch_size"]
    )

    learning_rate = (
        config["training"]["learning_rate"]
    )

    epochs = (
        config["training"]["epochs"]
    )

    patience = (
        config["training"][
            "early_stopping_patience"
        ]
    )

    num_workers = (
        config["training"][
            "num_workers"
        ]
    )

    # ---------------------------------------------------------
    # Read output configuration
    # ---------------------------------------------------------

    checkpoint_dir = Path(
        config["output"][
            "checkpoint_dir"
        ]
    )

    model_name = (
        config["output"]["model_name"]
    )

    checkpoint_path = (
        checkpoint_dir / model_name
    )

    # ---------------------------------------------------------
    # Select device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Structured JSON log
    print(
        json.dumps(
            {
                "event": "training_started",
                "device": str(device),
                "architecture": architecture,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "learning_rate": (
                    learning_rate
                ),
                "epochs": epochs,
                "early_stopping_patience": (
                    patience
                ),
                "num_workers": (
                    num_workers
                ),
            }
        ),
        flush=True,
    )

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    ).to(device)

    # ---------------------------------------------------------
    # Create DataLoaders
    # ---------------------------------------------------------

    train_loader, val_loader = (
        get_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    # ---------------------------------------------------------
    # Loss and optimizer
    # ---------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # ---------------------------------------------------------
    # Early stopping configuration
    # ---------------------------------------------------------

    best_val_loss = float("inf")

    patience_counter = 0

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
        )

        val_loss, val_accuracy = (
            evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
        )

        # -----------------------------------------------------
        # JSON metrics logging
        # -----------------------------------------------------

        log_entry = {
            "event": "epoch_complete",
            "epoch": epoch,
            "train_loss": round(
                train_loss,
                4,
            ),
            "train_accuracy": round(
                train_accuracy,
                4,
            ),
            "val_loss": round(
                val_loss,
                4,
            ),
            "val_accuracy": round(
                val_accuracy,
                4,
            ),
        }

        print(
            json.dumps(log_entry),
            flush=True,
        )

        # -----------------------------------------------------
        # Checkpoint and early stopping
        # -----------------------------------------------------

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "architecture": (
                        architecture
                    ),
                    "num_classes": (
                        num_classes
                    ),
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "val_loss": (
                        val_loss
                    ),
                    "val_accuracy": (
                        val_accuracy
                    ),
                },
                checkpoint_path,
            )

            print(
                json.dumps(
                    {
                        "event": (
                            "checkpoint_saved"
                        ),
                        "epoch": epoch,
                        "path": str(
                            checkpoint_path
                        ),
                        "val_loss": round(
                            val_loss,
                            4,
                        ),
                        "val_accuracy": round(
                            val_accuracy,
                            4,
                        ),
                    }
                ),
                flush=True,
            )

        else:
            patience_counter += 1

            print(
                json.dumps(
                    {
                        "event": (
                            "no_improvement"
                        ),
                        "epoch": epoch,
                        "patience_counter": (
                            patience_counter
                        ),
                        "patience": (
                            patience
                        ),
                    }
                ),
                flush=True,
            )

            if (
                patience_counter
                >= patience
            ):
                print(
                    json.dumps(
                        {
                            "event": (
                                "early_stopping"
                            ),
                            "epoch": epoch,
                            "best_val_loss": (
                                round(
                                    best_val_loss,
                                    4,
                                )
                            ),
                        }
                    ),
                    flush=True,
                )

                break

    # ---------------------------------------------------------
    # Training complete
    # ---------------------------------------------------------

    print(
        json.dumps(
            {
                "event": (
                    "training_complete"
                ),
                "best_val_loss": round(
                    best_val_loss,
                    4,
                ),
                "checkpoint_path": str(
                    checkpoint_path
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()