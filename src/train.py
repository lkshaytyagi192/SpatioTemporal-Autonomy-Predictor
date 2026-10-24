"""
Training Pipeline for Seq2Seq Attention Trajectory Forecasting.
Includes CUDA device acceleration, adaptive learning rate decay, ADE/FDE validation tracking,
checkpointing, and training curve visualization.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import get_dataloader
from src.seq2seq_attention import Seq2SeqAttentionTrajectory
from src.metrics import compute_ade, compute_fde


def train_model(
    train_csv: str = "data/train_trajectories.csv",
    test_csv: str = "data/test_trajectories.csv",
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.003,
    save_dir: str = "checkpoints",
    results_dir: str = "results"
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Computation Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_loader = get_dataloader(train_csv, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(test_csv, batch_size=batch_size, shuffle=False)

    model = Seq2SeqAttentionTrajectory().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()  # Huber Loss (resilient to sudden trajectory jerks)

    best_val_ade = float("inf")
    history = {"train_loss": [], "val_ade": [], "val_fde": []}

    print(f"\n[+] Commencing Model Training for {epochs} Epochs...\n" + "=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        teacher_forcing = max(0.0, 0.5 - (epoch / epochs) * 0.5)  # Scheduled decay

        for obs, pred_gt, _ in train_loader:
            obs = obs.to(device)
            pred_gt = pred_gt.to(device)

            optimizer.zero_grad()
            preds, _ = model(obs, target_traj=pred_gt, teacher_forcing_ratio=teacher_forcing)
            loss = criterion(preds, pred_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_train_loss += loss.item() * obs.size(0)

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            for obs, pred_gt, _ in test_loader:
                obs = obs.to(device)
                pred_gt = pred_gt.to(device)
                preds, _ = model(obs, target_traj=None, teacher_forcing_ratio=0.0)
                val_preds.append(preds.cpu())
                val_gts.append(pred_gt.cpu())

        val_preds_cat = torch.cat(val_preds, dim=0)
        val_gts_cat = torch.cat(val_gts, dim=0)

        epoch_ade = compute_ade(val_preds_cat, val_gts_cat)
        epoch_fde = compute_fde(val_preds_cat, val_gts_cat)

        history["train_loss"].append(avg_train_loss)
        history["val_ade"].append(epoch_ade)
        history["val_fde"].append(epoch_fde)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.5f} | Val ADE: {epoch_ade:.4f}m | Val FDE: {epoch_fde:.4f}m")

        # Save best model checkpoint
        if epoch_ade < best_val_ade:
            best_val_ade = epoch_ade
            best_checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_ade": best_val_ade,
                "fde": epoch_fde
            }, best_checkpoint_path)

    print("=" * 60)
    print(f"[OK] Training Complete! Best Validation ADE: {best_val_ade:.4f}m")
    print(f"[OK] Checkpoint successfully saved to: {os.path.join(save_dir, 'best_model.pt')}")

    # Generate and save publication-grade training curve plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    epochs_range = range(1, epochs + 1)
    line1 = ax1.plot(epochs_range, history["train_loss"], "b-", linewidth=2, label="SmoothL1 Loss")
    line2 = ax2.plot(epochs_range, history["val_ade"], "r--", linewidth=2, label="Validation ADE (m)")
    line3 = ax2.plot(epochs_range, history["val_fde"], "g-.", linewidth=2, label="Validation FDE (m)")

    ax1.set_xlabel("Training Epochs", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss Magnitude", color="b", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Error Metric (Meters)", color="darkred", fontsize=11, fontweight="bold")
    plt.title("Convergence & Generalization: Seq2Seq Attention Trajectory Predictor", fontsize=12, fontweight="bold")
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    plot_path = os.path.join(results_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Convergence plot saved to: {plot_path}")


if __name__ == "__main__":
    train_model()
