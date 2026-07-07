from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


@register_model("tiny_unet")
class TinyUNet(nn.Module):
    """Small U-Net baseline for pipeline debugging.

    Replace this model with a stronger implementation later by registering a
    class with the same ``num_classes`` constructor argument.
    """

    def __init__(self, num_classes: int, in_channels: int = 3, base_channels: int = 24):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 4, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def load_checkpoint(path: str, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model_name = ckpt["model_name"]
    class_names = ckpt["class_names"]
    model_kwargs = ckpt.get("model_kwargs", {})
    model = build_model(model_name, num_classes=len(class_names), **model_kwargs)
    model.load_state_dict(ckpt["model_state"])
    return model, class_names, ckpt


def predict_large_image(model, image_tensor, tile_size=512, overlap=64, device="cpu"):
    """Sliding-window inference for one CxHxW image tensor in [0, 1]."""
    model.eval()
    _, h, w = image_tensor.shape
    stride = tile_size - overlap
    logits_sum = None
    count = torch.zeros((1, h, w), dtype=torch.float32)

    with torch.no_grad():
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                y0_eff = max(0, y1 - tile_size)
                x0_eff = max(0, x1 - tile_size)
                patch = image_tensor[:, y0_eff:y1, x0_eff:x1].unsqueeze(0).to(device)
                out = model(patch).cpu().squeeze(0)
                if logits_sum is None:
                    logits_sum = torch.zeros((out.shape[0], h, w), dtype=torch.float32)
                logits_sum[:, y0_eff:y1, x0_eff:x1] += out
                count[:, y0_eff:y1, x0_eff:x1] += 1

    logits_sum = logits_sum / count.clamp_min(1.0)
    return torch.argmax(logits_sum, dim=0).numpy().astype("uint8")

