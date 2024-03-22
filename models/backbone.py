from torch import nn
from torchvision import models

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        embedding_dim = 2048
        self.freeze_base = False
        self.freeze_base4 = False

        self.base_network = models.resnet101(weights='DEFAULT')
        
        self.base_network.avgpool = nn.AvgPool2d(kernel_size=7,stride=1,padding=0) # replace avg pool
        # self.base_network.avgpool = nn.AvgPool2d(2,stride=2) # replace avg pool
        # print(self.base_network)
        if self.freeze_base:
            for param in self.base_network.parameters():
                param.requires_grad = False
        elif self.freeze_base4:
            for p in self.base_network.layer4.parameters(): 
                p.requires_grad=True

    def forward(self,images):
        x = self.base_network.conv1(images)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)
        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)
        x = self.base_network.layer3(x)
        x = self.base_network.layer4(x)
        # x = self.base_network.avgpool(x)
        return x
    
class DenseNetBackbone(nn.Module):
    def __init__(self):
        super(DenseNetBackbone, self).__init__()
        embedding_dim = 2208
        # embedding_dim = 1664
        self.base_network = models.densenet161(weights='DEFAULT').features

    def forward(self,images):
        x = self.base_network(images)
        # print(x.shape)
        return x