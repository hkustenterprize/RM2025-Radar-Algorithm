import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet18Classifier, self).__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(EfficientNetClassifier, self).__init__()

        self.backbone = models.efficientnet_b0(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, num_classes
        )

    def forward(self, x):
        return self.backbone(x)


class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(MobileNetClassifier, self).__init__()

        self.backbone = models.mobilenet_v2(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, num_classes
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(model_type, num_classes, weights_path):
    """
    Load model weights from a specified path.
    """
    assert model_type in [
        "resnet18",
        "efficientnet",
        "mobilenet",
    ], f"Unsupported model type: {model_type}"
    if model_type == "resnet18":
        model = ResNet18Classifier(num_classes=num_classes)
    elif model_type == "efficientnet":
        model = EfficientNetClassifier(num_classes=num_classes)
    elif model_type == "mobilenet":
        model = MobileNetClassifier(num_classes=num_classes)
    try:
        ckpt = torch.load(weights_path, map_location=torch.device("cpu"))
        model.load_state_dict(ckpt["model_state_dict"])

        print(f"Weights loaded successfully from {weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise e
    return model
