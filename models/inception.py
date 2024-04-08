import torchvision.models as models
import torch.nn as nn

class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3, self).__init__()
        inceptionv3 = models.inception_v3(weights='DEFAULT')
        num_features = inceptionv3.fc.in_features # 2048
        inceptionv3.fc = nn.Sequential(
            # nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.model = inceptionv3

    def forward(self, x):
        if self.training:
            return self.model(x)[0]
        else:
            return self.model(x)