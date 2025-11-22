from typing import List
import numpy as np
from shapely.geometry import Polygon

def intersection(g: Polygon, p: Polygon) -> float:
    """
    Calculate the Intersection over Union (IoU) of two polygons
    """
    try:
        if not g.is_valid: 
            g = g.buffer(0)
        if not p.is_valid: 
            p = p.buffer(0)
        
        inter = g.intersection(p).area
        union = g.area + p.area - inter
        if union == 0:
            return 0
        else:
            return inter / union
    except Exception:
        return 0

def weighted_merge(g: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Merge 2 boxes g and p based on their scores
    """
    g_score = g[8]
    p_score = p[8]
    
    if g_score + p_score == 0:
        return g
        
    # New_coords = (Coords_1 * Score_1 + Coords_2 * Score_2) / (Score_1 + Score_2)
    new_coords = (g[:8] * g_score + p[:8] * p_score) / (g_score + p_score)
    new_score = max(g_score, p_score)
    
    return np.concatenate((new_coords, [new_score]))

def standard_nms(boxes: List[np.ndarray], 
                 iou_threshold: float = 0.2) -> List[int]:
    """
    Standard Non-Maximum Suppression
    """
    if len(boxes) == 0:
        return []
        
    boxes = np.array(boxes)
    # Sort boxes by score in descending order
    indices = np.argsort(boxes[:, 8])[::-1]
    keep_indices = []
    
    polys = []
    for box in boxes:
        polys.append(Polygon(box[:8].reshape(4, 2)))

    while len(indices) > 0:
        current_idx = indices[0]
        keep_indices.append(current_idx)
        
        if len(indices) == 1: break
            
        current_poly = polys[current_idx]
        rest_indices = indices[1:]
        remove_mask = []
        
        for i, idx in enumerate(rest_indices):
            iou = intersection(current_poly, polys[idx])
            if iou > iou_threshold:
                remove_mask.append(i)
                
        indices = np.delete(indices, [0] + [x + 1 for x in remove_mask])
        
    return np.array(keep_indices)

def locality_aware_nms(boxes: List[np.ndarray],
                       iou_threshold: float = 0.2) -> List[np.ndarray]:
    """
    Locality-Aware NMS
    """
    if boxes is None or len(boxes) == 0:
        return []
    boxes = np.array(boxes).astype(np.float32)
    
    # Use C++ lanms if available for better performance
    try:
        import lanms
        setup = True
    except ImportError:
        setup = False

    if setup:
        boxes = lanms.merge_quadrangle_n9(boxes, iou_threshold)
        return boxes
    
    # Use Python implementation of Locality-Aware NMS
    indices = np.argsort(boxes[:, 8])[::-1]
    polys = []
    for box in boxes:
        polys.append(Polygon(box[:8].reshape(4, 2)))
        
    merged_boxes = []
    
    is_merged = np.zeros(len(boxes), dtype=bool)
    
    for i, idx in enumerate(indices):
        if is_merged[i]:
            continue
            
        current_poly = polys[idx]
        current_box = boxes[idx]
        
        # Find overlapping boxes to merge into the current box
        merged_box = current_box.copy()
        
        for j in range(i + 1, len(indices)):
            neighbor_idx = indices[j]
            if is_merged[j]:
                continue
                
            neighbor_poly = polys[neighbor_idx]
            neighbor_box = boxes[neighbor_idx]
            
            iou = intersection(current_poly, neighbor_poly)
            
            if iou > iou_threshold:
                # Merge boxes
                merged_box = weighted_merge(merged_box, neighbor_box)
                is_merged[j] = True # Mark neighbor as merged
        
        merged_boxes.append(merged_box)
    
    merged_boxes = np.array(merged_boxes)
    
    # Run standard NMS on merged boxes to remove any remaining overlaps
    final_indices = standard_nms(merged_boxes, iou_threshold=iou_threshold)
    return merged_boxes[final_indices]