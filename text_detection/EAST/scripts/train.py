import os
import sys
sys.path.append(os.getcwd())

import time
import yaml
import torch

from torch.utils.data import DataLoader
from torch import optim

from data.dataset import ICDAR2015Dataset
from models.east import EAST
from models.loss import EASTLoss
from utils.checkpoint import save_checkpoint, load_checkpoint

def train():
    # Load Configuration
    with open("configs/east_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Prepare Dataset & DataLoader
    train_dataset = ICDAR2015Dataset(
        data_dir=os.path.join(config["data"]["root_dir"], config["data"]["train_dir"]),
        input_size=config["data"]["input_size"],
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=True, 
        num_workers=config["data"]["num_workers"],
        drop_last=True,
        pin_memory=True
    )

    # Prepare Model
    model = EAST(
        backbone=config["model"].get("backbone", "resnet50"),
        pretrained=config["model"]["pretrained"]
    ).to(device)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 500], gamma=0.1)
    
    criterion = EASTLoss(lambda_geo=config["train"]["lambda_geometry"])

    # Resume training if needed
    checkpoint_dir = "checkpoints/" + config["experiment_name"]
    last_ckpt = os.path.join(checkpoint_dir, f"{config['experiment_name']}_latest.pth")
    start_epoch = 0
    
    if os.path.exists(last_ckpt):
        start_epoch = load_checkpoint(last_ckpt, model, optimizer, scheduler, device)

    # Training Loop
    model.train()
    print("Start Training...")
    
    for epoch in range(start_epoch, config["train"]["max_epochs"]):
        epoch_loss = 0.0
        start_time = time.time()
        
        for i, (img, gt_score, gt_geo, gt_mask) in enumerate(train_loader):
            img = img.to(device)
            gt_score = gt_score.to(device)
            gt_geo = gt_geo.to(device)
            gt_mask = gt_mask.to(device)

            # Forward
            pred_score, pred_geo = model(img)
            
            # Compute Loss
            loss = criterion((pred_score, pred_geo), (gt_score, gt_geo, gt_mask))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if (i + 1) % config["train"]["log_interval"] == 0:
                print(f"Epoch [{epoch + 1}/{config['train']['max_epochs']}] | "
                      f"Step [{i + 1}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        duration = time.time() - start_time
        print(f"End Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Time: {duration:.1f}s")

        # Save Checkpoint
        if (epoch + 1) % config["train"]["save_interval"] == 0:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss
            }
            save_checkpoint(state, checkpoint_dir, f"{config['experiment_name']}_epoch{epoch + 1}.pth")
            save_checkpoint(state, checkpoint_dir, f"{config['experiment_name']}_latest.pth")

if __name__ == "__main__":
    train()