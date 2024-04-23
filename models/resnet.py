import torchvision.models as models
import torch.nn as nn
from models.utils import weights_init

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(weights='DEFAULT')
        num_features = resnet50.fc.in_features # 2048
        resnet50.fc = nn.Sequential(
            # nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.model = resnet50

    def forward(self, x):
        return self.model(x)

class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()
        resnet152 = models.resnet152(weights='DEFAULT')
        num_features = resnet152.fc.in_features # 2048
        resnet152.fc = nn.Linear(num_features, num_classes)
        resnet152.fc.apply(weights_init)
        self.model = resnet152

    def forward(self, x):
        return self.model(x)