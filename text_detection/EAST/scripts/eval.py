import os
import sys
import yaml
import cv2
import torch
import glob
from tqdm import tqdm

sys.path.append(os.getcwd())

from models.east import EAST
from utils.evaluation import detect
from utils.visualization import draw_boxes
from utils.checkpoint import load_checkpoint

def evaluate():
    # Load Config
    with open("configs/east_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = EAST(backbone_name=config["model"].get("backbone", "resnet50"), pretrained=False)
    model = model.to(device)
    
    # Load Weights
    checkpoint_dir = "checkpoints/" + config["experiment_name"]
    ckpt_path = os.path.join(checkpoint_dir, "latest.pth") 
    
    load_checkpoint(ckpt_path, model, device=device)
    
    # Setup Input/Output
    test_dir = os.path.join(config["data"]["root_dir"], config["data"]["test_dir"])
    output_dir = f"outputs/{config["experiment_name"]}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img_paths = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                glob.glob(os.path.join(test_dir, "*.png"))
                
    print(f"Found {len(img_paths)} images for evaluation.")
    
    # Inference loop
    for img_path in tqdm(img_paths):
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None: continue
        
        # Detect
        boxes = detect(model, image, input_size=config["data"]["input_size"], 
                       device=device, conf_thresh=0.8, nms_thresh=0.2)
        
        # Draw & Save
        # Boxes output format: [x1, y1, x2, y2, x3, y3, x4, y4, score]
        res_path = os.path.join(output_dir, "res_" + filename)
        draw_boxes(image, boxes, output_path=res_path)
        
        # Save result text file for submitting to ICDAR challenges (if needed)
        txt_path = os.path.join(output_dir, "res_" + os.path.splitext(filename)[0] + ".txt")
        with open(txt_path, "w") as f:
            for box in boxes:
                # Format: x1, y1, x2, y2, x3, y3, x4, y4
                coords = [str(int(x)) for x in box[:8]]
                f.write(",".join(coords) + "\n")

    print(f"Evaluation finished. Results saved in {output_dir}")

if __name__ == "__main__":
    evaluate()