from typing import Dict, Tuple
import torch
import numpy as np
from models.pylanms import locality_aware_nms
from models.east import EAST
from shapely.geometry import Polygon
import cv2

def decode_predictions(score_map: np.ndarray, 
                       geo_map: np.ndarray, 
                       score_thresh: float = 0.8) -> np.ndarray:
    """
    Decode predictions from score and geometry maps to bounding boxes

    Parameters:
        score_map: Map of text scores, shape (1, H, W) or (H, W)
        geo_map: Map of geometry data, shape (5, H, W)
        score_thresh: Threshold to filter text pixels
    
    Returns:
        boxes: Array of shape (N, 9) where each row is 
               [x0, y0, x1, y1, x2, y2, x3, y3, score]
               representing the 4 corners of the box and its score
    """
    # Delete channel dimension of score map
    if len(score_map.shape) == 3:
        score_map = score_map[0]
    
    # Find coordinates of pixels with score > threshold
    xy_text = np.argwhere(score_map > score_thresh)
    if len(xy_text) == 0:
        return []
    
    # Sort by y, then by x
    y_list, x_list = xy_text[:, 0], xy_text[:, 1]
    scores = score_map[y_list, x_list]
    
    # Get d1, d2, d3, d4, angle
    geo_data = geo_map[:, y_list, x_list]
    d1, d2, d3, d4, angle = geo_data[0], geo_data[1], geo_data[2], geo_data[3], geo_data[4]
    
    scale_ratio = 4.0
    
    # Coordinate of box center in original image coordinate system
    cx = x_list * scale_ratio 
    cy = y_list * scale_ratio 
    
    # Calculate 4 corners based on d1, d2, d3, d4 and angle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rects = []
    for i in range(len(scores)):
        _cx, _cy = cx[i], cy[i]
        _d1, _d2, _d3, _d4 = d1[i], d2[i], d3[i], d4[i]
        _, _score = angle[i], scores[i]
        _cos, _sin = cos_a[i], sin_a[i]
        
        # p0 (top-left): (-d4, -d1), p1 (top-right): (d2, -d1)
        # p2 (bot-right): (d2, d3), p3 (bot-left): (-d4, d3)
        
        # Rotation matrix: [[cos, -sin], [sin, cos]] * [[x], [y]] + [[cx], [cy]]
        p0_x = -_d4 * _cos - (-_d1) * _sin + _cx
        p0_y = -_d4 * _sin + (-_d1) * _cos + _cy
        
        p1_x = _d2 * _cos - (-_d1) * _sin + _cx
        p1_y = _d2 * _sin + (-_d1) * _cos + _cy
        
        p2_x = _d2 * _cos - _d3 * _sin + _cx
        p2_y = _d2 * _sin + _d3 * _cos + _cy
        
        p3_x = -_d4 * _cos - _d3 * _sin + _cx
        p3_y = -_d4 * _sin + _d3 * _cos + _cy
        
        rects.append([p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, _score])

    return np.array(rects)

def detect(model: EAST, 
           image: np.ndarray, 
           input_size: int = 512, 
           device: str = "cuda", 
           conf_thresh: float = 0.8, 
           nms_thresh: float = 0.2) -> np.ndarray:
    """
    Perform text detection on an input image using the EAST model

    Parameters:
        model: Pre-trained EAST model
        image: Input image as a numpy array (H, W, C)
        input_size: Size to which the image is resized for the model
        device: Device to run the model on ("cuda" or "cpu")
        conf_thresh: Confidence threshold for filtering boxes
        nms_thresh: IoU threshold for Non-Maximum Suppression
    
    Returns:
        boxes: Array of detected boxes of shape (N, 8) where each row is 
               [x0, y0, x1, y1, x2, y2, x3, y3] representing the 4 corners of the box
    """
    model.eval()
    h, w = image.shape[:2]
    
    # Resize image
    img_resized = cv2.resize(image, (input_size, input_size))
    
    # Normalize & ToTensor
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_normalized = (img_resized / 255.0 - mean) / std
    img_normalized = img_normalized.transpose(2, 0, 1).astype(np.float32)
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        score, geo = model(img_tensor)
    
    score_map = score.cpu().numpy()[0] # (1, H/4, W/4)
    geo_map = geo.cpu().numpy()[0] # (5, H/4, W/4)
    
    # Decode boxes
    boxes = decode_predictions(score_map, geo_map, score_thresh=conf_thresh)
    
    if len(boxes) == 0:
        return []
    
    # Locality-Aware NMS
    boxes = locality_aware_nms(boxes, iou_threshold=nms_thresh)
    
    # Scale boxes back to original image size
    ratio_h = h / input_size
    ratio_w = w / input_size
    
    if len(boxes) > 0:
        boxes[:, [0, 2, 4, 6]] *= ratio_w
        boxes[:, [1, 3, 5, 7]] *= ratio_h
        
    return boxes

class Evaluator():
    def __init__(self, iou_thresh: float = 0.5) -> None:
        self.iou_thresh = iou_thresh
        self.reset()

    def reset(self) -> None:
        self.total_tp = 0
        self.total_fp = 0
        self.total_n_pos = 0

    def compute_iou(self, 
                    poly1: Polygon, 
                    poly2: Polygon) -> float:
        """
        Calculate IoU between two polygons
        """
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        if union == 0: 
            return 0.0
        return inter / union

    def evaluate_image(self, 
                       gt_polys: np.ndarray, 
                       gt_tags: np.ndarray, 
                       pred_boxes: np.ndarray) -> Tuple[int, int, int]:
        """
        Evaluate a single image's predictions against ground truth

        Parameters:
            gt_polys: Ground truth polygons, shape (N, 8)
            gt_tags: Boolean array indicating "Don't Care" regions, shape (N,)
            pred_boxes: Predicted bounding boxes, shape (M, 9)

        Returns:
            total_tp: Total True Positives
            total_fp: Total False Positives
            total_n_pos: Total number of positive samples in GT
        """
        gts = [Polygon(p.reshape(4, 2)) for p in gt_polys]
        preds = [Polygon(p[:8].reshape(4, 2)) for p in pred_boxes]
        
        # Flags to indicate which GTs and Preds have been matched
        gt_matched = [False] * len(gts)
        pred_matched = [False] * len(preds)
        
        # Calculate number of positive GTs
        n_pos = np.sum(~gt_tags)
        
        # Greedy Matching
        for i, pred in enumerate(preds):
            best_iou = -1
            best_gt_idx = -1
            
            for j, gt in enumerate(gts):
                if not gt_matched[j] and not gt_tags[j]:
                    iou = self.compute_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            if best_iou > self.iou_thresh:
                self.total_tp += 1
                gt_matched[best_gt_idx] = True
                pred_matched[i] = True

        # Process unmatched predictions
        for i, pred in enumerate(preds):
            if pred_matched[i]:
                continue
            
            is_ignored = False
            for j, gt in enumerate(gts):
                if gt_tags[j]: 
                    iou = self.compute_iou(pred, gt)
                    if iou > 0.5:
                        is_ignored = True
                        break
            
            if not is_ignored:
                self.total_fp += 1
        
        self.total_n_pos += n_pos
        
        return self.total_tp, self.total_fp, self.total_n_pos

    def get_metrics(self) -> Dict[str, float]:
        precision = self.total_tp / (self.total_tp + self.total_fp + 1e-6)
        recall = self.total_tp / (self.total_n_pos + 1e-6)
        hmean = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            "precision": precision,
            "recall": recall,
            "hmean": hmean,
            "tp": self.total_tp,
            "fp": self.total_fp,
            "n_pos": self.total_n_pos
        }