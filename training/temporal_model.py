import torch.nn as nn

from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


def build_temporal_model(pretrained: bool = True, input_mode: str = 'rgb'):
    if input_mode not in ('rgb', 'flow'):
        raise ValueError('Unknown temporal input mode')
    weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None

    model = r2plus1d_18(weights=weights)
    if input_mode == 'flow':
        import torch
        old = model.stem[0]
        stem = nn.Conv3d(2, old.out_channels, old.kernel_size, old.stride,
                         old.padding, bias=False)
        with torch.no_grad():
            stem.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, 2, 1, 1, 1)*1.5)
        model.stem[0] = stem
        # RGB running statistics are inappropriate for signed velocity fields.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
