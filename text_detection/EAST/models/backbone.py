from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, VGG16_BN_Weights

class VGG16BN(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        """
        VGG16-BN backbone for feature extraction
        """
        super(VGG16BN, self).__init__()
        
        if pretrained:
            weights = VGG16_BN_Weights.DEFAULT
        else:
            weights = None
            
        vgg = models.vgg16_bn(weights=weights)
        features = vgg.features
        
        # VGG16-BN pooling indices: 6, 13, 23, 33, 43
        self.layer1 = nn.Sequential(*features[:13])  # -> 1/4 (f1)
        self.layer2 = nn.Sequential(*features[13:23]) # -> 1/8 (f2)
        self.layer3 = nn.Sequential(*features[23:33]) # -> 1/16 (f3)
        self.layer4 = nn.Sequential(*features[33:43]) # -> 1/32 (f4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4

class ResNet50(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        """
        ResNet50 backbone for feature extraction
        """
        super(ResNet50, self).__init__()
        
        if pretrained:
            weights = ResNet50_Weights.DEFAULT 
        else:
            weights = None
            
        resnet = models.resnet50(weights=weights)
        
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4