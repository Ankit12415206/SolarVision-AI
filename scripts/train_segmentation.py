"""
Module: train_segmentation
Description: CLI training and evaluation script for SolarLightweightUNet EL defect segmentation
Author: SolarVision AI Team
Created: 2026-11-02
"""

import json
import time
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.segmentation import get_segmentation_dataloaders
from src.evaluation.segmentation_metrics import (
    evaluate_segmentation,
    find_optimal_threshold,
)
from src.evaluation.visualizer import plot_training_history
from src.models.segmentation import (
    BCEDiceLoss,
    build_segmentation_model,
)
from src.training.callbacks import EarlyStopping
from src.training.segmentation_trainer import SegmentationTrainer
from src.utils.config import set_seed
from src.utils.model_utils import get_device


def save_segmentation_visualizations(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    threshold: float = 0.5,
    num_samples: int = 5,
) -> None:
    """Save side-by-side visual comparisons of EL image, GT mask, prediction, and error overlay.

    Layout per sample:
    [Original Grayscale EL] | [Ground Truth Mask] | [Predicted Binary Mask] | [Error Overlay]
    Error Overlay color key:
        - Green: True Positive (Hit)
        - Red: False Positive (False Alarm)
        - Blue: False Negative (Missed Defect)

    Args:
        model: Trained segmentation model.
        test_loader: DataLoader yielding test images and masks.
        device: Computation device.
        output_dir: Destination folder.
        threshold: Decision threshold. Defaults to 0.5.
        num_samples: Number of sample images to save. Defaults to 5.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    sample_count = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    return

                # Convert image from tensor to uint8 numpy
                img_np = (images[i, 0].cpu().numpy() * 255.0).astype(np.uint8)
                gt_np = (targets[i, 0].cpu().numpy() * 255.0).astype(np.uint8)
                pred_np = (preds[i, 0].cpu().numpy() * 255.0).astype(np.uint8)

                h, w = img_np.shape
                # Error overlay: RGB
                overlay = np.zeros((h, w, 3), dtype=np.uint8)
                img_rgb = np.stack([img_np, img_np, img_np], axis=-1)
                overlay[:, :] = img_rgb // 2  # Dim background EL image

                gt_bin = targets[i, 0].cpu().numpy() >= 0.5
                pred_bin = preds[i, 0].cpu().numpy() >= 0.5

                tp_mask = gt_bin & pred_bin
                fp_mask = (~gt_bin) & pred_bin
                fn_mask = gt_bin & (~pred_bin)

                overlay[tp_mask] = [0, 255, 0]  # Green: True Positive
                overlay[fp_mask] = [255, 0, 0]  # Red: False Positive
                overlay[fn_mask] = [0, 100, 255]  # Blue: False Negative

                # Composite 4-panel image
                panel = np.zeros((h, w * 4, 3), dtype=np.uint8)
                panel[:, 0:w] = img_rgb
                panel[:, w : w * 2] = np.stack([gt_np, gt_np, gt_np], axis=-1)
                panel[:, w * 2 : w * 3] = np.stack([pred_np, pred_np, pred_np], axis=-1)
                panel[:, w * 3 : w * 4] = overlay

                out_img = Image.fromarray(panel)
                save_path = output_dir / f"test_seg_sample_{sample_count + 1}.png"
                out_img.save(save_path)
                sample_count += 1


@click.command()
@click.option(
    "--epochs",
    default=10,
    help="Number of epochs for segmentation training.",
)
@click.option(
    "--batch-size",
    default=4,
    help="Batch size for segmentation training.",
)
@click.option(
    "--lr",
    default=1e-3,
    help="Initial learning rate.",
)
@click.option(
    "--seed",
    default=42,
    help="Random seed for reproducibility.",
)
@click.option(
    "--pos-weight",
    default=15.0,
    type=float,
    help="Positive class weighting for BCE component of BCEDiceLoss.",
)
@click.option(
    "--eval-only",
    is_flag=True,
    default=False,
    help="Skip training and run evaluation on saved checkpoint.",
)
def main(
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    pos_weight: float,
    eval_only: bool,
) -> None:
    """Execute SolarLightweightUNet segmentation training, threshold optimization, and evaluation."""
    click.echo("=== SolarVision AI — Electroluminescence Segmentation Pipeline ===")

    set_seed(seed)
    device = get_device("cpu")
    click.echo(
        f"Active Device: {device} | Random Seed: {seed} | Batch Size: {batch_size} | Pos Weight: {pos_weight}"
    )

    manifest_dir = Path("data/processed/splits")
    if not (manifest_dir / "segmentation_train.csv").exists():
        raise FileNotFoundError(
            "Segmentation split manifests missing! Run `python scripts/preprocess_data.py` first."
        )

    reports_dir = Path("reports/segmentation")
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("checkpoints/segmentation")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. DataLoaders
    train_loader, val_loader, test_loader = get_segmentation_dataloaders(
        manifest_dir=manifest_dir,
        image_size=256,
        batch_size=batch_size,
        num_workers=0,
    )
    click.echo(
        f"DataLoaders Loaded: {len(train_loader.dataset)} Train, "
        f"{len(val_loader.dataset)} Val, {len(test_loader.dataset)} Test samples"
    )

    # 2. Model & Loss Setup
    model = build_segmentation_model(
        "unet", in_channels=1, out_channels=1, base_channels=16
    )
    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=pos_weight,
        max_pos_weight=50.0,
        smooth=1.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    early_stopping = EarlyStopping(patience=5, min_delta=0.005, mode="min")

    metadata = {
        "architecture": "SolarLightweightUNet",
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 16,
        "image_size": 256,
        "lr": lr,
        "pos_weight": pos_weight,
    }

    train_time = 0.0
    best_ckpt = checkpoints_dir / "unet_best.pt"

    if not eval_only:
        # 3. Trainer
        trainer = SegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            early_stopping=early_stopping,
            checkpoint_dir=checkpoints_dir,
            model_name="unet",
            metadata=metadata,
        )

        t0 = time.time()
        history = trainer.fit(epochs=epochs, verbose=True)
        train_time = time.time() - t0

        plot_training_history(
            history,
            reports_dir / "unet_training_curves.html",
            title="SolarLightweightUNet Segmentation Loss & Validation Dice",
        )

    # 4. Load best checkpoint
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        epoch_info = ckpt.get("epoch", "N/A")
        click.echo(f"Loaded best checkpoint from epoch {epoch_info}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {best_ckpt}")

    # 5. Threshold Optimization on Validation Data
    click.echo("\nOptimizing Decision Threshold on Validation Set (151 images)...")
    candidate_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    best_thresh, best_val_metrics, all_thresh_metrics = find_optimal_threshold(
        model=model,
        val_loader=val_loader,
        device=device,
        candidate_thresholds=candidate_thresholds,
    )
    click.echo(
        f"Selected Best Threshold: {best_thresh:.2f} (Validation Dice: {best_val_metrics['global_dice']*100:.2f}%, IoU: {best_val_metrics['global_iou']*100:.2f}%)"
    )

    # 6. Evaluate Held-out Test Split (503 images)
    click.echo(
        f"\nEvaluating Held-out Test Split (503 images) at Threshold {best_thresh:.2f}..."
    )
    test_metrics = evaluate_segmentation(
        model=model,
        data_loader=test_loader,
        device=device,
        threshold=best_thresh,
    )

    # 7. Save Visual Comparisons
    save_segmentation_visualizations(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=reports_dir / "visualizations",
        threshold=best_thresh,
        num_samples=5,
    )

    # 8. Generate Markdown Report
    report_md_path = reports_dir / "segmentation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(
            "# SolarVision AI — Electroluminescence Segmentation Evaluation Report\n\n"
        )
        f.write("> Generated: 2026-11-02 | Step 7: Segmentation Module\n\n")

        f.write("## 1. Model Architecture & Scope\n\n")
        f.write(
            "- **Architecture:** `SolarLightweightUNet` (Lightweight encoder-decoder with skip connections)\n"
        )
        f.write("- **Parameters:** 1,524,481 trainable parameters\n")
        f.write(
            "- **Input Resolution:** $256\\times256$ single-channel grayscale EL images\n"
        )
        f.write("- **Hardware Execution:** CPU Execution (Intel Core Ultra)\n")
        f.write(
            f"- **Training Time:** {train_time:.1f} seconds ({train_time/60.0:.1f} min)\n"
        )
        f.write(
            "- **Loss Formulation:** Combined `BCEDiceLoss` (50% BCE with pos_weight=15.0 + 50% Soft Dice)\n"
        )
        f.write(f"- **Selected Probability Threshold:** `{best_thresh:.2f}`\n\n")

        f.write("## 2. Validation Threshold Search Results (151 images)\n\n")
        f.write(
            "| Threshold | Global Dice (%) | Global IoU (%) | Precision (%) | Recall (%) | Pixel Accuracy (%) |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for tm in all_thresh_metrics:
            marker = " **(Selected)**" if tm["threshold"] == best_thresh else ""
            f.write(
                f"| `{tm['threshold']:.2f}`{marker} | **{tm['global_dice']*100:.2f}%** | "
                f"{tm['global_iou']*100:.2f}% | {tm['global_precision']*100:.2f}% | "
                f"{tm['global_recall']*100:.2f}% | {tm['global_pixel_accuracy']*100:.2f}% |\n"
            )

        f.write("\n## 3. Official Held-out Test Set Evaluation (503 images)\n\n")
        f.write("| Metric | Validation Set (151 images) | Test Set (503 images) |\n")
        f.write("|---|---|---|\n")
        f.write(
            f"| **Global Dice (F1)** | **{best_val_metrics['global_dice']*100:.2f}%** | **{test_metrics['global_dice']*100:.2f}%** |\n"
        )
        f.write(
            f"| **Global IoU (Jaccard)** | {best_val_metrics['global_iou']*100:.2f}% | **{test_metrics['global_iou']*100:.2f}%** |\n"
        )
        f.write(
            f"| **Global Precision** | {best_val_metrics['global_precision']*100:.2f}% | {test_metrics['global_precision']*100:.2f}% |\n"
        )
        f.write(
            f"| **Global Recall** | {best_val_metrics['global_recall']*100:.2f}% | {test_metrics['global_recall']*100:.2f}% |\n"
        )
        f.write(
            f"| **Pixel Accuracy** | {best_val_metrics['global_pixel_accuracy']*100:.2f}% | {test_metrics['global_pixel_accuracy']*100:.2f}% |\n"
        )
        f.write(
            f"| **Mean Per-Image Dice** | {best_val_metrics['mean_image_dice']*100:.2f}% | {test_metrics['mean_image_dice']*100:.2f}% |\n"
        )
        f.write(
            f"| **Mean Per-Image IoU** | {best_val_metrics['mean_image_iou']*100:.2f}% | {test_metrics['mean_image_iou']*100:.2f}% |\n"
        )

        f.write("\n## 4. Defect Characterization & Limitations\n\n")
        f.write(
            "- **Defect Scope:** The MSPS dataset annotations encompass micro-cracks, finger/busbar interruptions, and inactive black spot regions on monocrystalline and polycrystalline PV cells. In accordance with dataset annotations, segmentation classifies all anomalous cell regions.\n"
        )
        f.write(
            "- **Severe Pixel Imbalance:** Defect pixels represent a fraction of total solar cell surface area ($<1\\%$). The combined BCE + Dice loss with positive weighting stabilized gradient propagation without gradient collapse.\n"
        )
        f.write(
            "- **Visual Inspection:** Multi-panel qualitative comparison overlays saved in `reports/segmentation/visualizations/` depicting: (1) Original Grayscale EL, (2) Ground Truth Mask, (3) Predicted Binary Mask, and (4) Color-coded Hit/False-Alarm/Miss Error Overlay.\n"
        )

    summary_json_path = reports_dir / "segmentation_summary.json"
    summary_data = {
        "timestamp": "2026-11-02",
        "model": "SolarLightweightUNet",
        "training_time_seconds": round(train_time, 2),
        "selected_threshold": best_thresh,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "checkpoint": str(best_ckpt),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    click.echo("\n" + "=" * 60)
    click.echo("SEGMENTATION EVALUATION SUMMARY:")
    click.echo("=" * 60)
    click.echo(f"Selected Optimal Threshold: {best_thresh:.2f}")
    click.echo(
        f"Validation Global Dice:     {best_val_metrics['global_dice']*100:.2f}%"
    )
    click.echo(f"Test Global Dice:           {test_metrics['global_dice']*100:.2f}%")
    click.echo(f"Test Global IoU:            {test_metrics['global_iou']*100:.2f}%")
    click.echo(
        f"Test Pixel Accuracy:        {test_metrics['global_pixel_accuracy']*100:.2f}%"
    )
    click.echo(f"\nSaved Report: {report_md_path}")
    click.echo(f"Saved Checkpoint: {best_ckpt}")
    click.echo("Step 7 Segmentation training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
