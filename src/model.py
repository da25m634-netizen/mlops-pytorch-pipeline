import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(
    architecture: str = "simple_cnn",
    num_classes: int = 10,
) -> nn.Module:
    architecture = architecture.lower()

    if architecture == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if architecture == "resnet18":
        model = models.resnet18(weights=None)

        # Adapt ResNet-18 for CIFAR-10 32x32 images.
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        model.maxpool = nn.Identity()

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

        return model

    raise ValueError(
        f"Unsupported architecture: {architecture}"
    )