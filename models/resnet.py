import torchvision.models as models
import torch.nn as nn

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