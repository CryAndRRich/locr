import os
import sys
sys.path.append(os.getcwd())

import yaml
import cv2
import glob
from tqdm import tqdm
from tabulate import tabulate

import torch

from models.east import EAST
from utils.evaluation import detect, Evaluator
from utils.visualization import draw_boxes
from utils.checkpoint import load_checkpoint
from data.dataset import ICDAR2015Dataset

def evaluate():
    # Load Config
    with open("configs/east_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # Load Model
    model = EAST(
        backbone=config["model"].get("backbone", "resnet50"),
        pretrained=False
    ).to(device)
    
    # Load Weights
    checkpoint_dir = "checkpoints/" + config["experiment_name"]
    ckpt_path = os.path.join(checkpoint_dir, f"{config['experiment_name']}_latest.pth")

    load_checkpoint(ckpt_path, model, device=device)
    
    # Prepare Dataset
    test_dir = os.path.join(config["data"]["root_dir"], config["data"]["test_dir"])
    test_dataset = ICDAR2015Dataset(
        data_dir=test_dir,
        input_size=config["data"]["input_size"],
        is_train=False
    )

    # Setup Input/Output
    output_dir = f"outputs/{config['experiment_name']}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img_paths = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                glob.glob(os.path.join(test_dir, "*.png"))
                
    print(f"Found {len(img_paths)} images for evaluation")
    
    # Initialize Evaluator
    eval_config = config.get("eval", {})
    iou_thresh = eval_config.get("iou_thresh", 0.5)
    conf_thresh = eval_config.get("conf_thresh", 0.8)
    nms_thresh = eval_config.get("nms_thresh", 0.2)

    evaluator = Evaluator(iou_thresh=iou_thresh)

    # Inference loop
    for i in tqdm(range(len(test_dataset))):
        img_path = test_dataset.img_files[i]
        filename = os.path.basename(img_path)
        
        # Load Image
        image = cv2.imread(img_path)
        if image is None: 
            continue
        
        # Detect
        boxes = detect(
            model, 
            image, 
            input_size=config["data"]["input_size"], 
            device=device, 
            conf_thresh=conf_thresh, 
            nms_thresh=nms_thresh
        )

        # Load Ground Truth
        gt_polys, gt_tags = test_dataset.load_gt(img_path)
        evaluator.evaluate_image(gt_polys, gt_tags, boxes)
        
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

    metrics = evaluator.get_metrics()
    
    results_table = [
        ["Metric", "Value"],
        ["Precision", f"{metrics['precision']:.4f}"],
        ["Recall", f"{metrics['recall']:.4f}"],
        ["H-mean (F1)", f"{metrics['hmean']:.4f}"],
        ["TP (Care)", metrics["tp"]], # True Positives
        ["FP", metrics["fp"]], # False Positives
        ["Num GT (Care)", metrics["n_pos"]] # Number of ground truth positives
    ]
    
    print("\n" + "=" * 40)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print("=" * 40)
    print(tabulate(results_table, headers="firstrow", tablefmt="fancy_grid"))
    print(f"Visualization saved to: {output_dir}")
    print("=" * 40)

    print(f"Evaluation finished. Results saved in {output_dir}")

if __name__ == "__main__":
    evaluate()