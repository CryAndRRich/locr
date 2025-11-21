import torch
import os

def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path, model, optimizer=None, device="cpu"):
    if not os.path.exists(path):
        print(f"Checkpoint not found at {path}")
        return 0
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    start_epoch = checkpoint.get("epoch", 0)
    print(f"Loaded checkpoint from {path}, resuming from epoch {start_epoch}")
    return start_epoch