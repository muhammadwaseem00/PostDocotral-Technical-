"""
Multiple model architectures for pneumonia classification.
All models are adapted for: 1-channel (grayscale) input, 28x28 size, binary output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_resnet18():
    """ResNet18 - lightweight, fast training."""
    try:
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except (AttributeError, ImportError):
        model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def get_resnet34():
    """ResNet34 - deeper than ResNet18, more capacity."""
    try:
        from torchvision.models import ResNet34_Weights
        model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    except (AttributeError, ImportError):
        model = models.resnet34(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def get_resnet50():
    """ResNet50 - larger model, potentially better accuracy, slower."""
    try:
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except (AttributeError, ImportError):
        model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def get_efficientnet_b0():
    """EfficientNet-B0 - efficient architecture, good accuracy/speed tradeoff."""
    try:
        from torchvision.models import EfficientNet_B0_Weights
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    except (AttributeError, ImportError):
        model = models.efficientnet_b0(pretrained=True)
    # Modify first conv: EfficientNet uses Conv2d(3, 32, ...) in features[0]
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                      stride=old_conv.stride, padding=old_conv.padding, bias=False)
    # Modify classifier: (classifier): Sequential( (1): Linear(1280, 1000) )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def get_vgg16():
    """VGG16 - classic deep CNN."""
    try:
        from torchvision.models import VGG16_Weights
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    except (AttributeError, ImportError):
        model = models.vgg16(pretrained=True)
    # Modify first conv in features
    old_conv = model.features[0]
    model.features[0] = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                   stride=old_conv.stride, padding=old_conv.padding)
    # Modify classifier: last Linear(4096, 1000) -> Linear(4096, 1)
    model.classifier[6] = nn.Linear(4096, 1)
    return model


def get_custom_cnn():
    """Custom lightweight CNN - no pretrained weights, trains from scratch."""
    return CustomCNN()


def get_pneumonia_net():
    """PneumoniaNet: custom residual architecture for 28x28 chest X-ray (no pretrained)."""
    return PneumoniaNet()


class ResidualBlock(nn.Module):
    """Residual block: two 3x3 convs with optional channel projection."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.downsample(x)
        if shortcut.shape[2:] != out.shape[2:]:
            shortcut = F.adaptive_avg_pool2d(shortcut, out.shape[2:])
        return torch.relu(out + shortcut)


class PneumoniaNet(nn.Module):
    """
    Custom residual CNN for PneumoniaMNIST (1x28x28, binary).
    Designed for small data: residual blocks + dropout, no pretrained weights.
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32, dropout: float = 0.35):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
        )
        self.layer1 = self._make_layer(base_ch, base_ch, 2, stride=1)   # 14x14
        self.layer2 = self._make_layer(base_ch, base_ch * 2, 2, stride=2)  # 7x7
        self.layer3 = self._make_layer(base_ch * 2, base_ch * 4, 2, stride=2)  # 3x3 or 4x4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base_ch * 4, 1)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


class CustomCNN(nn.Module):
    """Simple 5-layer CNN for 28x28 grayscale images. No pretrained weights."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Registry: map name -> getter function
MODEL_REGISTRY = {
    "resnet18": get_resnet18,
    "resnet34": get_resnet34,
    "resnet50": get_resnet50,
    "efficientnet_b0": get_efficientnet_b0,
    "vgg16": get_vgg16,
    "custom_cnn": get_custom_cnn,
    "pneumonia_net": get_pneumonia_net,
}


def get_model(name="resnet18"):
    """
    Get model by name.

    Available models: resnet18, resnet34, resnet50, efficientnet_b0, vgg16, custom_cnn, pneumonia_net
    """
    name = name.lower().strip()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]()
