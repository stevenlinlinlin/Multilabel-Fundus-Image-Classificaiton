import torchvision.models as models
import torch.nn as nn
from models.utils import weights_init


class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3, self).__init__()
        inceptionv3 = models.inception_v3(weights='DEFAULT')
        num_features = inceptionv3.fc.in_features # 2048
        inceptionv3.fc = nn.Linear(num_features, num_classes)
        inceptionv3.fc.apply(weights_init)
        self.model = inceptionv3

    def forward(self, x):
        if self.training:
            return self.model(x)[0]
        else:
            return self.model(x)