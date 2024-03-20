import torchvision.models as models
import torch.nn as nn

class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        mobilenetv2 = models.mobilenet_v2(weights='DEFAULT')
        # num_features = mobilenetv2.classifier.in_features  # 1280
        mobilenetv2.classifier = nn.Sequential(
            # nn.BatchNorm1d(num_features),
            # nn.Linear(num_features, 416),
            # nn.BatchNorm1d(416),
            nn.Dropout(0.2),
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Linear(640, num_classes),
        )
        self.model = mobilenetv2

    def forward(self, x):
        return self.model(x)
