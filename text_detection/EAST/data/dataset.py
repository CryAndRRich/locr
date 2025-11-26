from typing import Tuple
import os
import glob

import cv2

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .geometry_utils import get_score_geo

class ICDAR2015Dataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 input_size: int = 512, 
                 is_train: bool = True) -> None:
        """
        Dataset class for ICDAR 2015 text detection dataset

        Parameters:
            data_dir: Path to the folder containing both images and .txt ground truths
            input_size: The input size for the model (default 512)
            is_train: Whether this is for training or inference
        """
        self.input_size = input_size
        self.is_train = is_train
        self.data_dir = data_dir
        
        # Supported image extensions
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self.img_files = []
        
        # Recursively find all images in the directory matching extensions
        for ext in extensions:
            self.img_files.extend(glob.glob(os.path.join(self.data_dir, ext)))
            
        # Sort to ensure deterministic ordering
        self.img_files = sorted(self.img_files)
        
        # Color normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if len(self.img_files) == 0:
            print(f"Warning: No images found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.img_files)

    def load_gt(self, img_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Ground Truth text file for a specific image
        
        Parameters:
            img_path: Path to the image file

        Returns:
            polys: Array of shape (N, 4, 2) containing N polygons with 4 points each
            tags: Boolean array of shape (N,) indicating "don't care" regions
        """
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Try finding file with "gt_" prefix (Standard ICDAR format)
        gt_path = os.path.join(self.data_dir, f"gt_{name_no_ext}.txt")
        
        # If not found, try finding file with same name as image
        if not os.path.exists(gt_path):
            gt_path = os.path.join(self.data_dir, f"{name_no_ext}.txt")
            
        polys = []
        tags = []
        
        # Return empty arrays if no GT file exists
        if not os.path.exists(gt_path):
            return np.array(polys), np.array(tags)

        # Read file with "utf-8-sig" to handle potential BOM characters
        with open(gt_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                # Clean whitespace
                line = line.strip()
                if not line: 
                    continue
                
                parts = line.split(",")
                
                # ICDAR 2015 format: x1, y1, x2, y2, x3, y3, x4, y4, transcript
                # Filter invalid lines
                if len(parts) < 9: 
                    continue
                
                try:
                    # Extract coordinates
                    coords = list(map(float, parts[:8]))
                    # Reconstruct transcript (in case it contains commas)
                    transcript = ",".join(parts[8:])
                    
                    poly = np.array(coords).reshape(4, 2)
                    polys.append(poly)
                    
                    # Mark as "True" (don't care) if transcript is "###"
                    tags.append(transcript == "###")
                except ValueError:
                    continue # Skip malformed lines
                
        return np.array(polys), np.array(tags)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item for DataLoader
        
        Parameters:
            index: Index of the item
            
        Returns:
            img_tensor: Normalized image tensor of shape (3, H, W)
        """
        img_path = self.img_files[index]
        
        # Read Image and Ground Truth
        image = cv2.imread(img_path)
        
        # Robustness check: Ensure image is read correctly
        if image is None:
            print(f"Error reading image: {img_path}")
            # Return a dummy tensor or handle error appropriately
            raise ValueError(f"Could not read image {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        polys, tags = self.load_gt(img_path)

        # Resize Image & Polygons
        new_h, new_w = self.input_size, self.input_size
        image_resized = cv2.resize(image, (new_w, new_h))
        
        # Calculate scaling factors
        scale_x = new_w / w
        scale_y = new_h / h
        
        # Scale polygons
        if len(polys) > 0:
            polys[:, :, 0] *= scale_x
            polys[:, :, 1] *= scale_y
        
        # Generate EAST Maps (Score Map & Geometry Map)
        # The output maps are 1/4 size of the input image
        score_map, geo_map, training_mask = get_score_geo(
            (new_h, new_w), polys, tags, scale_ratio=0.25
        )

        # Normalize and convert to PyTorch Tensors
        # Image: (H, W, C) -> (C, H, W) normalized
        img_tensor = self.normalize(image_resized)
        
        # Score Map: (H/4, W/4) -> (1, H/4, W/4)
        score_tensor = torch.from_numpy(score_map).unsqueeze(0)
        
        # Geometry Map: (H/4, W/4, 5) -> (5, H/4, W/4)
        # Permute is needed if get_score_geo returns (H, W, 5)
        geo_tensor = torch.from_numpy(geo_map).permute(2, 0, 1)
        
        # Training Mask: (H/4, W/4) -> (1, H/4, W/4)
        mask_tensor = torch.from_numpy(training_mask).unsqueeze(0)

        return img_tensor, score_tensor, geo_tensor, mask_tensor