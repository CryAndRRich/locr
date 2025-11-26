from typing import List, Tuple
import random

import cv2

import numpy as np

class RandomRotate():
    """
    Rotate image and polygons randomly within a specified angle range
    """
    def __init__(self, max_angle: float = 10.0) -> None:    
        self.max_angle = max_angle

    def __call__(self, 
                 image: np.ndarray, 
                 polys: np.ndarray, 
                 tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            return image, polys, tags
            
        angle = random.uniform(-self.max_angle, self.max_angle)
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Rotation Matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_img = cv2.warpAffine(image, M, (w, h), borderValue=(0,0,0))
        
        rotated_polys = []
        for poly in polys:
            # Add one column of ones to the polygon points: (x, y, 1)
            pts = np.concatenate([poly, np.ones((4, 1))], axis=1)
            rotated_poly = np.dot(pts, M.T)
            rotated_polys.append(rotated_poly)
            
        return rotated_img, np.array(rotated_polys), tags

class RandomCrop():
    """
    Crop the image randomly while ensuring that text polygons are not cut through
    """
    def __init__(self, min_crop_ratio: float = 0.1) -> None:
        self.min_crop_ratio = min_crop_ratio

    def __call__(self, 
                 image: np.ndarray, 
                 polys: np.ndarray, 
                 tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        
        # Try cropping up to 10 times, if not successful return original image
        for _ in range(10):
            crop_h = random.randint(int(h * self.min_crop_ratio), h)
            crop_w = random.randint(int(w * self.min_crop_ratio), w)
            
            x_start = random.randint(0, w - crop_w)
            y_start = random.randint(0, h - crop_h)
            
            current_polys = []
            current_tags = []
            
            if len(polys) == 0:
                return image[y_start:y_start+crop_h, x_start:x_start+crop_w], polys, tags
            
            for poly_idx, poly in enumerate(polys):
                poly_copy = poly.copy()
                
                # Move coordinates to crop coordinate system
                poly_copy[:, 0] -= x_start
                poly_copy[:, 1] -= y_start
                
                # Check if the polygon is completely outside the crop area
                cx = np.mean(poly_copy[:, 0])
                cy = np.mean(poly_copy[:, 1])
                
                if 0 <= cx < crop_w and 0 <= cy < crop_h:
                    poly_copy[:, 0] = np.clip(poly_copy[:, 0], 0, crop_w)
                    poly_copy[:, 1] = np.clip(poly_copy[:, 1], 0, crop_h)
                    
                    if cv2.contourArea(poly_copy.astype(np.int32)) > 10:
                        current_polys.append(poly_copy)
                        current_tags.append(tags[poly_idx])
                
            if len(current_polys) > 0:
                return image[y_start:y_start+crop_h, x_start:x_start+crop_w], np.array(current_polys), np.array(current_tags)

        return image, polys, tags

class Compose():
    def __init__(self, transforms: List[object]) -> None:
        self.transforms = transforms

    def __call__(self, image, polys, tags):
        for t in self.transforms:
            image, polys, tags = t(image, polys, tags)
        return image, polys, tags