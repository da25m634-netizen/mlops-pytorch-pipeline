import pytest
import torch

from src.model import get_model


@pytest.mark.parametrize(
    "architecture",
    [
        "simple_cnn",
        "resnet18",
    ],
)
def test_model_output_shape(
    architecture,
):
    model = get_model(
        architecture=architecture,
        num_classes=10,
    )

    inputs = torch.randn(
        4,
        3,
        32,
        32,
    )

    outputs = model(inputs)

    assert outputs.shape == (4, 10)


def test_invalid_architecture():
    with pytest.raises(ValueError):
        get_model(
            architecture="invalid_model",
            num_classes=10,
        )