import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image


from training.config import LAST_MODEL


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = LAST_MODEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():

    model = efficientnet_v2_s(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        1
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state)

    model.to(DEVICE)

    model.eval()

    return model


transform = transforms.Compose([

    transforms.Resize((384, 384)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

])


def predict(image_path):

    model = build_model()

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probability = torch.sigmoid(output).item()

    return probability


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Path to image"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():

        print("Image not found")

        return

    probability = predict(image_path)

    print()

    print("=" * 40)

    print("Prediction")

    print("=" * 40)

    print()

    print(f"Image : {image_path.name}")

    print()

    print(f"AI probability : {probability * 100:.2f}%")

    print()

    if probability > 0.5:

        print("Prediction : AI GENERATED")

    else:

        print("Prediction : REAL")

    print()

    confidence = abs(probability - 0.5) * 2 * 100

    print(f"Confidence : {confidence:.2f}%")

    print()

    print("=" * 40)


if __name__ == "__main__":

    main()