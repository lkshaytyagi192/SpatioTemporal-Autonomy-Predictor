"""
Comprehensive Benchmark & Evaluation Pipeline.
Compares the Kinematic Constant Velocity Baseline against the Seq2Seq Attention Deep Learning Model.
Evaluates ADE, FDE, and inference latency (FPS/ms) on the test dataset.
Generates results/benchmark_summary.json.
"""

import os
import time
import json
import torch
import numpy as np
import pandas as pd

from src.dataset import TrajectoryDataset, get_dataloader
from src.baseline_model import ConstantVelocityBaseline
from src.seq2seq_attention import Seq2SeqAttentionTrajectory
from src.metrics import compute_ade, compute_fde, compute_rmse
from src.collision_risk import CollisionRiskEngine


def run_benchmark(
    test_csv: str = "data/test_trajectories.csv",
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_json: str = "results/benchmark_summary.json"
):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Benchmarking on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    test_loader = get_dataloader(test_csv, batch_size=32, shuffle=False)
    num_samples = len(test_loader.dataset)

    # ---------------- 1. Kinematic Baseline Evaluation ---------------- #
    baseline = ConstantVelocityBaseline(pred_len=12)
    baseline_preds, true_targets = [], []
    
    t0 = time.perf_counter()
    for obs, pred_gt, _ in test_loader:
        preds = baseline.predict(obs)
        baseline_preds.append(preds)
        true_targets.append(pred_gt)
    t_baseline = (time.perf_counter() - t0) * 1000.0 / num_samples  # ms per sample

    b_preds_cat = torch.cat(baseline_preds, dim=0)
    gts_cat = torch.cat(true_targets, dim=0)

    base_ade = compute_ade(b_preds_cat, gts_cat)
    base_fde = compute_fde(b_preds_cat, gts_cat)
    base_rmse = compute_rmse(b_preds_cat, gts_cat)

    # ---------------- 2. Seq2Seq Attention Model Evaluation ---------------- #
    model = Seq2SeqAttentionTrajectory().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dl_preds = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for obs, _, _ in test_loader:
            obs = obs.to(device)
            preds, _ = model(obs, target_traj=None, teacher_forcing_ratio=0.0)
            dl_preds.append(preds.cpu())
    t_dl = (time.perf_counter() - t0) * 1000.0 / num_samples  # ms per sample

    dl_preds_cat = torch.cat(dl_preds, dim=0)
    dl_ade = compute_ade(dl_preds_cat, gts_cat)
    dl_fde = compute_fde(dl_preds_cat, gts_cat)
    dl_rmse = compute_rmse(dl_preds_cat, gts_cat)

    # Compute improvements
    ade_improvement = ((base_ade - dl_ade) / base_ade) * 100.0
    fde_improvement = ((base_fde - dl_fde) / base_fde) * 100.0

    # ---------------- 3. Collision Risk Audit ---------------- #
    risk_engine = CollisionRiskEngine(safety_radius=2.0, critical_radius=1.0)
    # Evaluate risk on first 20 agents
    dataset = TrajectoryDataset(test_csv, normalize_relative=True)
    sample_indices = list(range(min(20, len(dataset))))
    obs_sample = dataset.obs[sample_indices].to(device)
    last_obs_sample = dataset.last_obs[sample_indices]
    
    with torch.no_grad():
        preds_sample, _ = model(obs_sample)
    pred_abs = (preds_sample.cpu() + last_obs_sample.unsqueeze(1)).numpy()
    risk_events = risk_engine.assess_risk(pred_abs, sample_indices)

    # ---------------- Results Display & Export ---------------- #
    summary = {
        "test_dataset_samples": num_samples,
        "observation_horizon_steps": 8,
        "prediction_horizon_steps": 12,
        "dt_seconds": 0.4,
        "models": {
            "constant_velocity_baseline": {
                "ade_meters": round(base_ade, 4),
                "fde_meters": round(base_fde, 4),
                "rmse_meters": round(base_rmse, 4),
                "latency_ms_per_agent": round(t_baseline, 4)
            },
            "seq2seq_attention_lstm": {
                "ade_meters": round(dl_ade, 4),
                "fde_meters": round(dl_fde, 4),
                "rmse_meters": round(dl_rmse, 4),
                "latency_ms_per_agent": round(t_dl, 4),
                "ade_reduction_percent": round(ade_improvement, 2),
                "fde_reduction_percent": round(fde_improvement, 2)
            }
        },
        "collision_risk_audit": {
            "evaluated_agents": len(sample_indices),
            "critical_risk_events_detected": len([r for r in risk_events if r["risk_level"] == "CRITICAL_COLLISION"]),
            "caution_proximity_events": len([r for r in risk_events if r["risk_level"] == "CAUTION_PROXIMITY"])
        }
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=4)

    # Print clean benchmark table
    print("\n" + "=" * 75)
    print("      SPATIO-TEMPORAL TRAJECTORY PREDICTION BENCHMARK RESULTS")
    print("=" * 75)
    print(f"{'Model Architecture':<30} | {'ADE (m)':<10} | {'FDE (m)':<10} | {'Latency (ms)':<12}")
    print("-" * 75)
    print(f"{'Constant Velocity (Baseline)':<30} | {base_ade:<10.4f} | {base_fde:<10.4f} | {t_baseline:<12.4f}")
    print(f"{'Seq2Seq + Temporal Attention':<30} | {dl_ade:<10.4f} | {dl_fde:<10.4f} | {t_dl:<12.4f}")
    print("=" * 75)
    print(f"[>] Relative Performance Gain : ADE reduced by {ade_improvement:.2f}% | FDE reduced by {fde_improvement:.2f}%")
    print(f"[>] Multi-Agent Risk Audit    : {summary['collision_risk_audit']['critical_risk_events_detected']} Critical & {summary['collision_risk_audit']['caution_proximity_events']} Caution Proximity Events Detected.")
    print(f"[OK] Full benchmark report saved to: {output_json}\n")


if __name__ == "__main__":
    run_benchmark()
