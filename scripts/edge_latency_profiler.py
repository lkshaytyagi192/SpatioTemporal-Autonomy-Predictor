"""
Edge Hardware Latency & Throughput Profiler.
Benchmarks model inference time across variable batch sizes (B=1, 4, 8, 16, 32),
computes Trajectories-Per-Second (Throughput/FPS), and profiles GPU VRAM allocation.
Critical for embedded robotics platforms (NVIDIA Jetson, Raspberry Pi 5).
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.seq2seq_attention import Seq2SeqAttentionTrajectory


def profile_inference(checkpoint_path: str = "checkpoints/best_model.pt", results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Edge Profiling Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = Seq2SeqAttentionTrajectory().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch_sizes = [1, 2, 4, 8, 16, 32]
    latencies_ms = []
    throughputs_fps = []

    print("\n" + "=" * 65)
    print(f"{'Batch Size':<12} | {'Latency (ms)':<15} | {'Throughput (Traj/s)':<22}")
    print("-" * 65)

    # Warm-up run
    dummy_input = torch.randn(1, 8, 2, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    for bs in batch_sizes:
        x = torch.randn(bs, 8, 2, device=device)
        iterations = 100
        
        # Synchronize for CUDA timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        
        total_time = time.perf_counter() - t0
        avg_latency_ms = (total_time / iterations) * 1000.0
        throughput = (bs * iterations) / total_time

        latencies_ms.append(avg_latency_ms)
        throughputs_fps.append(throughput)
        print(f"{bs:<12} | {avg_latency_ms:<15.3f} | {throughput:<22.1f}")

    print("=" * 65)

    # GPU Memory Footprint
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[+] VRAM Footprint -> Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

    # Generate benchmark plot
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax2 = ax1.twinx()

    b_labels = [str(b) for b in batch_sizes]
    x_indices = np.arange(len(batch_sizes))
    width = 0.35

    rects1 = ax1.bar(x_indices - width/2, latencies_ms, width, label='Inference Latency (ms)', color='#2b5c8f')
    rects2 = ax2.bar(x_indices + width/2, throughputs_fps, width, label='Throughput (Traj/sec)', color='#e26d5c')

    ax1.set_xlabel('Batch Size (Agents Processed Concurrently)', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Latency per Forward Pass (ms)', color='#2b5c8f', fontweight='bold', fontsize=10)
    ax2.set_ylabel('System Throughput (Traj/sec)', color='#e26d5c', fontweight='bold', fontsize=10)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(b_labels)
    ax1.set_title(f'Edge Latency & Throughput Profile ({device.type.upper()})', fontweight='bold', fontsize=11)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    fig.tight_layout()
    plot_file = os.path.join(results_dir, "edge_latency_profile.png")
    plt.savefig(plot_file, dpi=180)
    plt.close()
    print(f"[OK] Profiling plot saved to: {plot_file}\n")


if __name__ == "__main__":
    profile_inference()
