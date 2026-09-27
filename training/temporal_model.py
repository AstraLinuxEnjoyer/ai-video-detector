import torch.nn as nn

from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


def build_temporal_model(pretrained: bool = True):
    weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None

    model = r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
