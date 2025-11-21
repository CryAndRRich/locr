import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbone import ResNet50, VGG16BN

class EAST(nn.Module):
    def __init__(self, 
                 backbone: str = "resnet50",
                 pretrained: bool = True) -> None:
        super(EAST, self).__init__()

        if backbone == "resnet50":
            self.backbone = ResNet50(pretrained=pretrained)
            # Resnet channels
            f_channels = [256, 512, 1024, 2048]
        elif backbone == "vgg16":
            self.backbone = VGG16BN(pretrained=pretrained)
            # VGG16 channels
            f_channels = [128, 256, 512, 512]
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")
        
        # Merge stage 1: from f4 -> merge with f3
        self.conv_h1_1 = nn.Conv2d(f_channels[3] + f_channels[2], 128, 1)
        self.conv_h1_2 = nn.Conv2d(128, 128, 3, padding=1)

        # Merge stage 2: from h1 -> merge with f2
        self.conv_h2_1 = nn.Conv2d(128 + f_channels[1], 64, 1)
        self.conv_h2_2 = nn.Conv2d(64, 64, 3, padding=1)

        # Merge stage 3: from h2 -> merge with f1
        self.conv_h3_1 = nn.Conv2d(64 + f_channels[0], 32, 1)
        self.conv_h3_2 = nn.Conv2d(32, 32, 3, padding=1)

        # Output Layers
        self.conv_score = nn.Conv2d(32, 1, 1)
        self.conv_geo = nn.Conv2d(32, 5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EAST model
        
        Parameters:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            score: Text score map of shape (B, 1, H/4, W/4)
            geo: Geometry map of shape (B, 5, H/4, W/4)
        """
        # Backbone forward
        f1, f2, f3, f4 = self.backbone(x)

        # Upsample f4 and concatenate with f3
        g1 = F.interpolate(f4, scale_factor=2, mode="bilinear", align_corners=True)
        h1 = torch.cat((g1, f3), 1)
        h1 = F.relu(self.conv_h1_1(h1))
        h1 = F.relu(self.conv_h1_2(h1))

        # Upsample h1 and concatenate with f2
        g2 = F.interpolate(h1, scale_factor=2, mode="bilinear", align_corners=True)
        h2 = torch.cat((g2, f2), 1)
        h2 = F.relu(self.conv_h2_1(h2))
        h2 = F.relu(self.conv_h2_2(h2))

        # Upsample h2 and concatenate with f1
        g3 = F.interpolate(h2, scale_factor=2, mode="bilinear", align_corners=True)
        h3 = torch.cat((g3, f1), 1)
        h3 = F.relu(self.conv_h3_1(h3))
        h3 = F.relu(self.conv_h3_2(h3))

        # Prediction Heads
        score = self.sigmoid(self.conv_score(h3))
        
        # Geometry map: 
        # 4 channels for distances to box sides
        # 1 channel for angle
        geo = self.sigmoid(self.conv_geo(h3)) 
        
        # Scale output:
        # Distance * 512 (Input size)
        dists = geo[:, :4, :, :] * 512 
        # Angle: (val - 0.5) * pi -> Range (-pi/2, pi/2)
        angle = (geo[:, 4, :, :] - 0.5) * math.pi
        angle = angle.unsqueeze(1)
        
        final_geo = torch.cat((dists, angle), 1)

        return score, final_geo