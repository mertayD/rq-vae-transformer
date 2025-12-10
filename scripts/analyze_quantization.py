#!/usr/bin/env python3
# =============================================================================
# Analyze Quantization Error at Each Depth for RQ-VAE
# =============================================================================
# Analyzes:
#   1. Residual norm ||r_d|| at each depth d=1,2,3,4
#   2. Codebook utilization (which codes are used at each depth)
#
# Generates:
#   - Bar chart: avg residual norm per depth for each model
#   - Histogram: codebook utilization per depth
#   - CSV files with detailed statistics
#
# Usage:
#   python scripts/analyze_quantization.py --checkpoint-dir ./checkpoints
#
# =============================================================================

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rqvae.img_datasets import create_dataset
from rqvae.models import create_model
from rqvae.utils.config import load_config, augment_arch_defaults


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENTS = {
    "baseline": {
        "name": "Baseline",
        "weights": "[1.0, 1.0, 1.0, 1.0]",
        "color": "#1f77b4"  # blue
    },
    "coarse": {
        "name": "Coarse-First",
        "weights": "[4.0, 2.0, 1.0, 0.5]",
        "color": "#ff7f0e"  # orange
    },
    "fine": {
        "name": "Fine-First",
        "weights": "[0.5, 1.0, 2.0, 4.0]",
        "color": "#2ca02c"  # green
    },
    "mid": {
        "name": "Mid-Focus",
        "weights": "[0.5, 1.0, 1.0, 0.5]",
        "color": "#d62728"  # red
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"quantization_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the best/latest checkpoint in the directory."""
    checkpoints = list(checkpoint_dir.glob("**/epoch*_model.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    checkpoints.sort(key=lambda x: int(x.stem.split("epoch")[1].split("_")[0]))
    return checkpoints[-1]


def load_model(checkpoint_path: Path, use_ema: bool = True, device: torch.device = None):
    """Load RQ-VAE model from checkpoint."""
    config_path = checkpoint_path.parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    config = load_config(str(config_path))
    config.arch = augment_arch_defaults(config.arch)

    model, _ = create_model(config.arch, ema=False)

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if use_ema and 'state_dict_ema' in ckpt:
        model.load_state_dict(ckpt['state_dict_ema'])
    else:
        model.load_state_dict(ckpt['state_dict'])

    if device:
        model = model.to(device)

    return model, config


@torch.no_grad()
def analyze_quantization_depth(
    model: nn.Module,
    dataset,
    batch_size: int = 32,
    device: torch.device = None,
    max_samples: int = None
) -> dict:
    """
    Analyze quantization at each depth:
    - Residual norms at each depth
    - Code usage at each depth
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Get the quantizer
    if hasattr(model, 'module'):
        quantizer = model.module.quantizer
        encoder = model.module.encoder
        quant_conv = model.module.quant_conv
    else:
        quantizer = model.quantizer
        encoder = model.encoder
        quant_conv = model.quant_conv

    n_codebooks = quantizer.code_shape[-1]  # depth (e.g., 4)
    n_embed = quantizer.n_embed[0] if hasattr(quantizer.n_embed, '__iter__') else quantizer.n_embed

    # Storage for metrics
    residual_norms = [[] for _ in range(n_codebooks)]
    code_counts = [defaultdict(int) for _ in range(n_codebooks)]
    total_samples = 0

    for batch_idx, (xs, _) in enumerate(tqdm(loader, desc="Analyzing")):
        if max_samples and total_samples >= max_samples:
            break

        xs = xs.to(device, non_blocking=True)
        batch_size_actual = xs.shape[0]
        total_samples += batch_size_actual

        # Encode
        z_e = encoder(xs)
        z_e = quant_conv(z_e).permute(0, 2, 3, 1).contiguous()

        # Reshape to code shape
        x_reshaped = quantizer.to_code_shape(z_e)

        # Manual residual quantization to get intermediate residuals
        B, h, w, embed_dim = x_reshaped.shape
        residual_feature = x_reshaped.clone()

        for d in range(n_codebooks):
            # Quantize current residual
            quant, code = quantizer.codebooks[d](residual_feature)

            # Compute residual norm BEFORE subtracting
            # ||r_d|| = ||x - sum_{i<d} q_i|| for input to depth d
            residual_norm = residual_feature.pow(2).sum(dim=-1).sqrt()  # [B, h, w]
            residual_norms[d].extend(residual_norm.view(-1).cpu().numpy().tolist())

            # Update residual
            residual_feature = residual_feature - quant

            # Count code usage
            code_flat = code.view(-1).cpu().numpy()
            for c in code_flat:
                code_counts[d][int(c)] += 1

    # Compute statistics
    results = {
        "n_codebooks": n_codebooks,
        "n_embed": n_embed,
        "total_samples": total_samples,
        "depth_stats": []
    }

    for d in range(n_codebooks):
        norms = np.array(residual_norms[d])
        codes = code_counts[d]

        # Codebook utilization
        unique_codes = len(codes)
        utilization = unique_codes / n_embed * 100

        # Code frequency distribution
        total_code_usage = sum(codes.values())
        code_probs = np.array([codes.get(i, 0) / total_code_usage for i in range(n_embed)])

        # Entropy of code distribution
        code_probs_nonzero = code_probs[code_probs > 0]
        entropy = -np.sum(code_probs_nonzero * np.log2(code_probs_nonzero))
        max_entropy = np.log2(n_embed)
        normalized_entropy = entropy / max_entropy

        depth_stat = {
            "depth": d + 1,
            "residual_norm_mean": float(np.mean(norms)),
            "residual_norm_std": float(np.std(norms)),
            "residual_norm_median": float(np.median(norms)),
            "residual_norm_min": float(np.min(norms)),
            "residual_norm_max": float(np.max(norms)),
            "unique_codes": unique_codes,
            "utilization_pct": utilization,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "code_counts": dict(codes)
        }
        results["depth_stats"].append(depth_stat)

        logging.info(
            f"  Depth {d+1}: residual_norm={np.mean(norms):.4f} +/- {np.std(norms):.4f}, "
            f"utilization={utilization:.1f}%, entropy={normalized_entropy:.3f}"
        )

    return results


def plot_residual_norms(all_results: dict, output_dir: Path):
    """Plot bar chart of average residual norm per depth for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    experiments = list(all_results.keys())
    n_depths = all_results[experiments[0]]["n_codebooks"]
    x = np.arange(n_depths)
    width = 0.2

    for i, exp_name in enumerate(experiments):
        results = all_results[exp_name]
        means = [s["residual_norm_mean"] for s in results["depth_stats"]]
        stds = [s["residual_norm_std"] for s in results["depth_stats"]]

        offset = (i - len(experiments) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=f"{EXPERIMENTS[exp_name]['name']}",
            color=EXPERIMENTS[exp_name]['color'],
            capsize=3
        )

    ax.set_xlabel('Depth', fontsize=12)
    ax.set_ylabel('Residual Norm (mean ± std)', fontsize=12)
    ax.set_title('Residual Norm at Each Quantization Depth', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d+1}" for d in range(n_depths)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "residual_norms_by_depth.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {save_path}")


def plot_codebook_utilization(all_results: dict, output_dir: Path):
    """Plot codebook utilization per depth for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    experiments = list(all_results.keys())
    n_depths = all_results[experiments[0]]["n_codebooks"]

    for i, exp_name in enumerate(experiments):
        ax = axes[i]
        results = all_results[exp_name]

        depths = [s["depth"] for s in results["depth_stats"]]
        utilizations = [s["utilization_pct"] for s in results["depth_stats"]]
        entropies = [s["normalized_entropy"] * 100 for s in results["depth_stats"]]

        x = np.arange(len(depths))
        width = 0.35

        bars1 = ax.bar(x - width/2, utilizations, width, label='Utilization %',
                       color=EXPERIMENTS[exp_name]['color'], alpha=0.8)
        bars2 = ax.bar(x + width/2, entropies, width, label='Norm. Entropy %',
                       color=EXPERIMENTS[exp_name]['color'], alpha=0.4)

        ax.set_xlabel('Depth')
        ax.set_ylabel('Percentage')
        ax.set_title(f"{EXPERIMENTS[exp_name]['name']}\n{EXPERIMENTS[exp_name]['weights']}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"d={d}" for d in depths])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = output_dir / "codebook_utilization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {save_path}")


def plot_code_frequency_histogram(all_results: dict, output_dir: Path, depth: int = 1):
    """Plot code frequency histogram for a specific depth across all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    experiments = list(all_results.keys())

    for i, exp_name in enumerate(experiments):
        ax = axes[i]
        results = all_results[exp_name]
        n_embed = results["n_embed"]

        depth_stat = results["depth_stats"][depth - 1]
        code_counts = depth_stat["code_counts"]

        # Create histogram data
        counts = [code_counts.get(str(j), code_counts.get(j, 0)) for j in range(n_embed)]

        # Bin the histogram for visualization
        n_bins = 50
        bin_size = n_embed // n_bins
        binned_counts = []
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size
            binned_counts.append(sum(counts[start:end]))

        ax.bar(range(n_bins), binned_counts, color=EXPERIMENTS[exp_name]['color'], alpha=0.7)
        ax.set_xlabel(f'Code Index (binned, {bin_size} codes/bin)')
        ax.set_ylabel('Frequency')
        ax.set_title(f"{EXPERIMENTS[exp_name]['name']} - Depth {depth}\n"
                    f"Utilization: {depth_stat['utilization_pct']:.1f}%")
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Code Usage Distribution at Depth {depth}', fontsize=14)
    plt.tight_layout()
    save_path = output_dir / f"code_frequency_depth{depth}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {save_path}")


def save_results_csv(all_results: dict, output_dir: Path):
    """Save detailed results to CSV."""
    # Summary CSV
    summary_path = output_dir / "quantization_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'experiment', 'depth', 'residual_norm_mean', 'residual_norm_std',
            'unique_codes', 'utilization_pct', 'normalized_entropy'
        ])

        for exp_name, results in all_results.items():
            for stat in results["depth_stats"]:
                writer.writerow([
                    exp_name,
                    stat["depth"],
                    f"{stat['residual_norm_mean']:.6f}",
                    f"{stat['residual_norm_std']:.6f}",
                    stat["unique_codes"],
                    f"{stat['utilization_pct']:.2f}",
                    f"{stat['normalized_entropy']:.4f}"
                ])

    logging.info(f"Saved: {summary_path}")


def print_summary_table(all_results: dict):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("Quantization Analysis Summary")
    print("=" * 100)

    experiments = list(all_results.keys())
    n_depths = all_results[experiments[0]]["n_codebooks"]

    # Header
    header = f"{'Experiment':<12} | "
    for d in range(1, n_depths + 1):
        header += f"{'Depth ' + str(d):^20} | "
    print(header)
    print("-" * 100)

    # Residual norms
    print("Residual Norm (mean):")
    for exp_name in experiments:
        row = f"{exp_name:<12} | "
        for stat in all_results[exp_name]["depth_stats"]:
            row += f"{stat['residual_norm_mean']:^20.4f} | "
        print(row)

    print("-" * 100)

    # Utilization
    print("Codebook Utilization (%):")
    for exp_name in experiments:
        row = f"{exp_name:<12} | "
        for stat in all_results[exp_name]["depth_stats"]:
            row += f"{stat['utilization_pct']:^20.1f} | "
        print(row)

    print("=" * 100 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze quantization error at each depth for RQ-VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint-dir", "-c",
        type=str,
        default="./checkpoints",
        help="Base directory containing experiment checkpoints"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./results/quantization_analysis",
        help="Directory to save results and plots"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to analyze (None for all)"
    )
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help="Experiments to analyze"
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Use EMA weights if available"
    )

    args = parser.parse_args()

    # Setup
    checkpoint_base = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Results storage
    all_results = {}

    # Process each experiment
    logger.info(f"Analyzing {len(args.experiments)} experiments: {args.experiments}")

    for exp_name in args.experiments:
        exp_dir = checkpoint_base / exp_name

        logger.info("=" * 60)
        logger.info(f"Experiment: {exp_name} ({EXPERIMENTS[exp_name]['name']})")
        logger.info("=" * 60)

        if not exp_dir.exists():
            logger.warning(f"Checkpoint directory not found: {exp_dir}")
            continue

        try:
            # Find and load checkpoint
            checkpoint_path = find_best_checkpoint(exp_dir)
            logger.info(f"Using checkpoint: {checkpoint_path.name}")

            model, config = load_model(checkpoint_path, use_ema=args.use_ema, device=device)
            model = model.to(device).eval()

            # Create dataset
            _, dataset_val = create_dataset(config, is_eval=True)
            logger.info(f"Dataset size: {len(dataset_val)}")

            # Clear GPU memory
            torch.cuda.empty_cache()

            # Analyze
            results = analyze_quantization_depth(
                model,
                dataset_val,
                batch_size=args.batch_size,
                device=device,
                max_samples=args.max_samples
            )

            all_results[exp_name] = results

            # Free memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate plots and save results
    if all_results:
        logger.info("Generating plots...")

        # Bar chart: residual norms
        plot_residual_norms(all_results, output_dir)

        # Codebook utilization
        plot_codebook_utilization(all_results, output_dir)

        # Code frequency histograms for each depth
        n_depths = list(all_results.values())[0]["n_codebooks"]
        for d in range(1, n_depths + 1):
            plot_code_frequency_histogram(all_results, output_dir, depth=d)

        # Save CSV
        save_results_csv(all_results, output_dir)

        # Print summary
        print_summary_table(all_results)

    logger.info("Done!")


if __name__ == "__main__":
    main()
