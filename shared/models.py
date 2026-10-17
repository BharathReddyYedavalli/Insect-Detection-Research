"""Torchvision model factory, with a 102-way classifier head."""
from __future__ import annotations
import torch.nn as nn
from torchvision import models

SUPPORTED = ("resnet18", "resnet50", "vgg16", "vgg19")


def build_model(name: str, num_classes: int = 102, pretrained: bool = True):
    name = name.lower()
    if name not in SUPPORTED:
        raise ValueError(f"Unsupported model {name!r}; available: {SUPPORTED}")
    # Torchvision caches official pretrained weights in the user's torch cache.
    constructor = getattr(models, name)
    model = constructor(weights="DEFAULT" if pretrained else None)
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
