import torchvision.models as models
import torch.nn as nn

class DenseNet169(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet169, self).__init__()
        densenet169 = models.densenet169(weights='DEFAULT')
        num_features = densenet169.classifier.in_features  # 1664
        densenet169.classifier = nn.Sequential(
            # nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 416),
            nn.BatchNorm1d(416),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(416, num_classes),
        )
        self.model = densenet169

    def forward(self, x):
        return self.model(x)