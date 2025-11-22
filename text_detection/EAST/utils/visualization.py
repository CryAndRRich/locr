from typing import List, Union

import cv2

import numpy as np
import torch

def draw_boxes(image: Union[np.ndarray, torch.Tensor], 
               boxes: List[List[float]], 
               output_path: str = None) -> np.ndarray:
    """
    Draw bounding boxes on the image
    
    Parameters:
        image: Input image as a numpy array or torch tensor
        boxes: List of bounding boxes, each box is a list of 8 coordinates + optional score
        output_path: If provided, save the resulting image to this path
    """
    img_draw = image.copy()
    if isinstance(img_draw, torch.Tensor):
        img_draw = img_draw.permute(1, 2, 0).cpu().numpy()
        img_draw = (img_draw * 255).astype(np.uint8)
    
    img_draw = np.ascontiguousarray(img_draw)
    
    for box in boxes:
        if len(box) >= 8:
            pts = np.array(box[:8], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_draw, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            if len(box) > 8:
                score = box[-1] 
                text_x = int(pts[0][0][0])
                text_y = int(pts[0][0][1] - 5) 
                
                cv2.putText(img_draw, f"{score:.2f}", (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if output_path:
        cv2.imwrite(output_path, img_draw)
    
    return img_draw