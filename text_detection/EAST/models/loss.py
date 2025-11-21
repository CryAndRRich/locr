from typing import Tuple
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()

    def forward(self, 
                input: torch.Tensor, 
                target: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Dice Loss computation

        Parameters:
            input: Predicted score map of shape (N, H, W)
            target: Ground truth score map of shape (N, H, W)
            mask: Training mask of shape (N, H, W)
        """
        input = input * mask
        target = target * mask 
        
        intersection = (input * target).sum()
        union = input.sum() + target.sum() + 1e-5
        return 1. - (2. * intersection / union)

class EASTLoss(nn.Module):
    def __init__(self, lambda_geo: float = 1.0) -> None:
        super(EASTLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.lambda_geo = lambda_geo

    def forward(self, 
                prediction: Tuple[torch.Tensor, torch.Tensor],
                target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for EAST Loss computation
        
        Parameters:
            prediction: Tuple containing
                - pred_score: Predicted score map of shape (N, 1, H, W)
                - pred_geo: Predicted geometry map of shape (N, 5, H, W)
            target: Tuple containing
                - gt_score: Ground truth score map of shape (N, 1, H, W)
                - gt_geo: Ground truth geometry map of shape (N, 5, H, W)
                - gt_mask: Training mask of shape (N, 1, H, W)
        """
        pred_score, pred_geo = prediction
        gt_score, gt_geo, gt_mask = target

        # Score Loss (Dice Loss)
        loss_score = self.dice_loss(pred_score.squeeze(1), gt_score, gt_mask) # (N, H, W)

        # Geometry Loss (IoU Loss + Angle Loss)
        pixel_mask = (gt_score * gt_mask).squeeze(1).bool() # (N, H, W)
        
        if pixel_mask.sum() > 0:
            # Fetch positive pixels
            # Permute to (N, H, W, C) and then mask
            pred_geo = pred_geo.permute(0, 2, 3, 1)[pixel_mask] # (Num_Pos_Pixels, 5)
            gt_geo = gt_geo.permute(0, 2, 3, 1)[pixel_mask] # (Num_Pos_Pixels, 5)

            # IoU Loss for AABB
            # d1, d2, d3, d4 represent distances from pixel to top, right, bottom, left sides of the box
            d1_gt, d2_gt, d3_gt, d4_gt = torch.split(gt_geo[:, :4], 1, dim=1) 
            d1_pred, d2_pred, d3_pred, d4_pred = torch.split(pred_geo[:, :4], 1, dim=1) 

            area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt) 
            area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred) 
            
            w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred) 
            h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
            area_intersect = w_union * h_union
            
            iou = (area_intersect + 1e-5) / (area_gt + area_pred - area_intersect + 1e-5)
            loss_aabb = -torch.log(iou).mean()

            # Angle Loss
            angle_gt = gt_geo[:, 4]
            angle_pred = pred_geo[:, 4]
            loss_angle = (1 - torch.cos(angle_pred - angle_gt)).mean()

            loss_geo = loss_aabb + 20 * loss_angle
        else:
            loss_geo = 0.0

        return loss_score + self.lambda_geo * loss_geo