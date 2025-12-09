#!/usr/bin/env python3
# =============================================================================
# Compute rFID (Reconstruction FID) for All RQ-VAE Experiments
# =============================================================================
# Computes FID between original validation images and their reconstructions
# for all trained RQ-VAE checkpoints (baseline, coarse, fine, mid).
#
# Usage:
#   python scripts/compute_all_rfid.py --checkpoint-dir ./checkpoints
#   python scripts/compute_all_rfid.py --checkpoint-dir ./checkpoints --batch-size 50
#
# =============================================================================

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rqvae.img_datasets import create_dataset
from rqvae.models import create_model
from rqvae.metrics.fid import compute_rfid, get_inception_model
from rqvae.utils.config import load_config, augment_arch_defaults


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENTS = {
    "baseline": {
        "name": "Baseline",
        "weights": "[1.0, 1.0, 1.0, 1.0]",
        "description": "Uniform weights"
    },
    "coarse": {
        "name": "Coarse-First",
        "weights": "[4.0, 2.0, 1.0, 0.5]",
        "description": "Emphasize early codebooks"
    },
    "fine": {
        "name": "Fine-First",
        "weights": "[0.5, 1.0, 2.0, 4.0]",
        "description": "Emphasize later codebooks"
    },
    "mid": {
        "name": "Mid-Focus",
        "weights": "[0.5, 1.0, 1.0, 0.5]",
        "description": "Emphasize middle codebooks"
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"rfid_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
    checkpoints = list(checkpoint_dir.glob("epoch*_model.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Sort by epoch number and get the latest
    checkpoints.sort(key=lambda x: int(x.stem.split("epoch")[1].split("_")[0]))
    return checkpoints[-1]


def load_model(checkpoint_path: Path, use_ema: bool = True, device: torch.device = None):
    """Load RQ-VAE model from checkpoint."""
    # Load config from the same directory
    config_path = checkpoint_path.parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    config = load_config(str(config_path))
    config.arch = augment_arch_defaults(config.arch)

    # Create model
    model, _ = create_model(config.arch, ema=False)

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if use_ema and 'state_dict_ema' in ckpt:
        model.load_state_dict(ckpt['state_dict_ema'])
        logging.info(f"Loaded EMA weights from {checkpoint_path.name}")
    else:
        model.load_state_dict(ckpt['state_dict'])
        logging.info(f"Loaded weights from {checkpoint_path.name}")

    if device:
        model = model.to(device)

    return model, config


def compute_rfid_for_checkpoint(
    checkpoint_path: Path,
    batch_size: int = 100,
    device: torch.device = None,
    use_ema: bool = True,
    inception_model: nn.Module = None
) -> dict:
    """Compute rFID for a single checkpoint."""

    # Load model
    model, config = load_model(checkpoint_path, use_ema=use_ema, device=device)
    model = nn.DataParallel(model).eval()

    # Create dataset
    _, dataset_val = create_dataset(config, is_eval=True)
    logging.info(f"Validation dataset size: {len(dataset_val)}")

    # Compute rFID
    start_time = datetime.now()
    rfid = compute_rfid(
        dataset_val,
        model,
        batch_size=batch_size,
        device=device
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    # Get epoch from checkpoint name
    epoch = int(checkpoint_path.stem.split("epoch")[1].split("_")[0])

    return {
        "rfid": rfid,
        "epoch": epoch,
        "elapsed_seconds": elapsed,
        "num_samples": len(dataset_val)
    }


def save_results_csv(results: list, output_path: Path):
    """Save results to CSV file."""
    fieldnames = [
        "experiment",
        "name",
        "commitment_weights",
        "description",
        "rfid",
        "epoch",
        "checkpoint_path",
        "num_samples",
        "elapsed_seconds",
        "timestamp"
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logging.info(f"Results saved to: {output_path}")


def print_results_table(results: list):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("rFID Results Summary")
    print("=" * 80)
    print(f"{'Experiment':<12} {'rFID':>10} {'Weights':<25} {'Epoch':>6}")
    print("-" * 80)

    # Sort by rFID (lower is better)
    sorted_results = sorted(results, key=lambda x: x['rfid'])

    for r in sorted_results:
        print(f"{r['experiment']:<12} {r['rfid']:>10.4f} {r['commitment_weights']:<25} {r['epoch']:>6}")

    print("-" * 80)

    # Highlight best result
    best = sorted_results[0]
    print(f"\nBest: {best['experiment']} with rFID = {best['rfid']:.4f}")
    print("=" * 80 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute rFID for all RQ-VAE experiments",
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
        default="./results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="Batch size for FID computation"
    )
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help="Experiments to evaluate"
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Use EMA weights if available"
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Do not use EMA weights"
    )

    args = parser.parse_args()

    # Setup
    checkpoint_base = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_ema = args.use_ema and not args.no_ema

    # Setup logging
    logger = setup_logging(output_dir)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Pre-load inception model (shared across experiments)
    logger.info("Loading Inception model for FID computation...")
    inception_model = get_inception_model().to(device).eval()

    # Results storage
    results = []

    # Process each experiment
    logger.info(f"Evaluating {len(args.experiments)} experiments: {args.experiments}")

    for exp_name in args.experiments:
        exp_info = EXPERIMENTS[exp_name]
        exp_dir = checkpoint_base / exp_name

        logger.info("=" * 60)
        logger.info(f"Experiment: {exp_name} ({exp_info['name']})")
        logger.info(f"Commitment weights: {exp_info['weights']}")
        logger.info("=" * 60)

        if not exp_dir.exists():
            logger.warning(f"Checkpoint directory not found: {exp_dir}")
            logger.warning("Skipping this experiment")
            continue

        try:
            # Find checkpoint
            checkpoint_path = find_best_checkpoint(exp_dir)
            logger.info(f"Using checkpoint: {checkpoint_path.name}")

            # Clear GPU memory
            torch.cuda.empty_cache()

            # Compute rFID
            metrics = compute_rfid_for_checkpoint(
                checkpoint_path,
                batch_size=args.batch_size,
                device=device,
                use_ema=use_ema
            )

            logger.info(f"rFID: {metrics['rfid']:.4f}")
            logger.info(f"Computation time: {metrics['elapsed_seconds']:.1f}s")

            # Store results
            results.append({
                "experiment": exp_name,
                "name": exp_info['name'],
                "commitment_weights": exp_info['weights'],
                "description": exp_info['description'],
                "rfid": metrics['rfid'],
                "epoch": metrics['epoch'],
                "checkpoint_path": str(checkpoint_path),
                "num_samples": metrics['num_samples'],
                "elapsed_seconds": metrics['elapsed_seconds'],
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Error processing {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save and display results
    if results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = output_dir / f"rfid_results_{timestamp}.csv"
        save_results_csv(results, csv_path)
        print_results_table(results)

        # Also save a "latest" version for easy access
        latest_path = output_dir / "rfid_results_latest.csv"
        save_results_csv(results, latest_path)
    else:
        logger.warning("No results to save!")

    logger.info("Done!")


if __name__ == "__main__":
    main()
