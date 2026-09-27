import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


def build_model():

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(in_features, 2)

    return model