from typing import List, Tuple
import numpy as np
import cv2
from shapely.geometry import Polygon

def shrink_poly(poly: np.ndarray, 
                r: float = 0.4) -> np.ndarray:
    """
    Shrink polygon using Vatti clipping algorithm
    """
    try:
        # Use shapely to handle polygon operations
        polygon = Polygon(poly)
        area = polygon.area
        perimeter = polygon.length
        
        if perimeter == 0: return poly
        
        # Vatti's formula to calculate offset distance
        # D = (Area * (1 - r^2)) / Perimeter
        offset = (area * (1 - r * r)) / (perimeter + 1e-6)
        
        # Use negative buffer to shrink
        shrunk_polygon = polygon.buffer(-offset, join_style=2)
        
        # Convert back to numpy array
        if shrunk_polygon.is_empty:
            return None
            
        # If result is MultiPolygon, take the largest one
        if shrunk_polygon.geom_type == "MultiPolygon":
            shrunk_polygon = max(shrunk_polygon, key=lambda a: a.area)
            
        shrunk_coords = np.array(shrunk_polygon.exterior.coords)[:-1]
        return shrunk_coords.astype(np.int32)
        
    except Exception as e:
        # Fallback in case of error
        return None

def get_score_geo(img_size: Tuple[int, int], 
                  polys: np.ndarray, 
                  tags: List[bool], 
                  scale_ratio: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Score Map, Geometry Map, and Training Mask for EAST model

    Parameters:
        img_size: Tuple of (height, width) of the original image
        polys: Numpy array of shape (N, 4, 2) containing N polygons
        tags: List of booleans indicating "Don't Care" regions
        scale_ratio: Scaling factor from original image to feature map size
    """
    h, w = img_size
    # Output map size
    h_out, w_out = int(h * scale_ratio), int(w * scale_ratio)
    
    score_map = np.zeros((h_out, w_out), dtype=np.float32)
    geo_map = np.zeros((h_out, w_out, 5), dtype=np.float32)
    training_mask = np.ones((h_out, w_out), dtype=np.float32)

    for poly_idx, poly in enumerate(polys):
        tag = tags[poly_idx]
        poly_small = poly * scale_ratio
        
        # Process "Don't Care" regions
        if tag:
            cv2.fillPoly(training_mask, [poly_small.astype(np.int32)], 0)
            continue

        # Find minimum area rectangle (RBOX)
        rect = cv2.minAreaRect(poly_small.astype(np.float32))
        center, size, angle_deg = rect
        
        # Normalize angle and size
        if size[0] < size[1]:
            angle_deg -= 90
            size = (size[1], size[0])
            
        angle_rad = np.deg2rad(angle_deg)

        # Shrink polygon
        shrunk_poly = shrink_poly(poly_small, r=0.4)
        if shrunk_poly is None:
            continue

        # Draw Score Map
        cv2.fillPoly(score_map, [shrunk_poly.astype(np.int32)], 1)

        # Calculate Geometry Map
        poly_mask = np.zeros((h_out, w_out), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [shrunk_poly.astype(np.int32)], 1)
        ys, xs = np.where(poly_mask > 0) # Pixels inside the shrunk polygon

        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)

        # Vectorized computation
        dx = xs - center[0]
        dy = ys - center[1]
        
        # Rotate points to align with rectangle axes
        d1_x = dx * cos_a - dy * sin_a
        d1_y = dx * sin_a + dy * cos_a
        
        # Box half sizes
        w_half = size[0] / 2
        h_half = size[1] / 2
        
        d1 = h_half - d1_y # Dist to Top
        d2 = w_half - d1_x # Dist to Right
        d3 = h_half + d1_y # Dist to Bottom
        d4 = w_half + d1_x # Dist to Left
        
        valid = (d1 > 0) & (d2 > 0) & (d3 > 0) & (d4 > 0)
        
        # Only update valid pixels
        idx_y = ys[valid]
        idx_x = xs[valid]
        
        geo_map[idx_y, idx_x, 0] = d1[valid]
        geo_map[idx_y, idx_x, 1] = d2[valid]
        geo_map[idx_y, idx_x, 2] = d3[valid]
        geo_map[idx_y, idx_x, 3] = d4[valid]
        geo_map[idx_y, idx_x, 4] = angle_rad
    
    geo_map[:,:,:4] /= img_size[0]

    return score_map, geo_map, training_mask