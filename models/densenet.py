import torchvision.models as models
import torch.nn as nn
from models.utils import weights_init


class DenseNet169(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet169, self).__init__()
        densenet169 = models.densenet169(weights='DEFAULT')
        num_features = densenet169.classifier.in_features  # 1664
        densenet169.classifier = nn.Linear(num_features, num_classes)
        densenet169.classifier.apply(weights_init)
        self.model = densenet169

    def forward(self, x):
        return self.model(x)
    
class DenseNet161(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet161, self).__init__()
        densenet161 = models.densenet161(weights='DEFAULT')
        # densenet161 = models.densenet161()
        num_features = densenet161.classifier.in_features  # 2208
        densenet161.classifier = nn.Linear(num_features, num_classes)
        densenet161.classifier.apply(weights_init)
        self.model = densenet161

    def forward(self, x):
        return self.model(x)