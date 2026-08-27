import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import get_model


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = [
    0.4914,
    0.4822,
    0.4465,
]

CIFAR10_STD = [
    0.2470,
    0.2435,
    0.2616,
]


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "checkpoints/model.pt",
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = None


inference_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
        ),
    ]
)


def load_trained_model():
    global model

    if not os.path.exists(MODEL_PATH):
        print(
            f"Model checkpoint not found: {MODEL_PATH}"
        )
        model = None
        return

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    architecture = checkpoint.get(
        "architecture",
        "simple_cnn",
    )

    num_classes = checkpoint.get(
        "num_classes",
        10,
    )

    loaded_model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    )

    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loaded_model.to(device)
    loaded_model.eval()

    model = loaded_model

    print(
        f"Model loaded successfully from {MODEL_PATH}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_trained_model()
    yield


app = FastAPI(
    title="PyTorch CIFAR-10 Classifier",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if not image.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type",
        )

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image",
        )

    image_bytes = await image.read()

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        ) from exc

    input_tensor = (
        inference_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(input_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    probabilities_list = (
        probabilities.cpu().tolist()
    )

    predicted_index = int(
        probabilities.argmax().item()
    )

    class_probabilities = {
        class_name: round(probability, 6)
        for class_name, probability in zip(
            CIFAR10_CLASSES,
            probabilities_list,
        )
    }

    return {
        "predicted_class": (
            CIFAR10_CLASSES[predicted_index]
        ),
        "predicted_class_index": (
            predicted_index
        ),
        "confidence": round(
            probabilities_list[
                predicted_index
            ],
            6,
        ),
        "probabilities": (
            class_probabilities
        ),
    }