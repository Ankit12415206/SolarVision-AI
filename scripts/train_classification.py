"""
Module: train_classification
Description: CLI training and evaluation script for SolarVision AI classification models
Author: SolarVision AI Team
Created: 2026-11-02
"""

import json
import time
from pathlib import Path
from typing import Any, Dict

import click
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.classification import (
    CLASSIFICATION_CLASSES,
    get_classification_dataloaders,
)
from src.evaluation.metrics import evaluate_classification
from src.evaluation.visualizer import (
    generate_classification_report_markdown,
    plot_confusion_matrix,
    plot_training_history,
)
from src.models.classification import build_classification_model
from src.training.callbacks import EarlyStopping
from src.training.trainer import ClassificationTrainer
from src.utils.config import load_yaml_config, set_seed
from src.utils.model_utils import get_device


@click.command()
@click.option(
    "--config",
    default="configs/default.yaml",
    help="Path to YAML configuration file.",
)
@click.option(
    "--epochs-cnn",
    default=12,
    help="Number of epochs for Custom CNN training.",
)
@click.option(
    "--epochs-mobilenet",
    default=12,
    help="Number of epochs for MobileNetV2 fine-tuning.",
)
@click.option(
    "--batch-size",
    default=8,
    help="Batch size for training and evaluation.",
)
@click.option(
    "--seed",
    default=42,
    help="Random seed for reproducibility.",
)
def main(
    config: str,
    epochs_cnn: int,
    epochs_mobilenet: int,
    batch_size: int,
    seed: int,
) -> None:
    """Execute sequential training and evaluation for Custom CNN and MobileNetV2."""
    click.echo("=== SolarVision AI — Classification Training Pipeline ===")

    set_seed(seed)
    device = get_device("cpu")
    click.echo(
        f"Active Device: {device} | Random Seed: {seed} | Batch Size: {batch_size}"
    )

    manifest_dir = Path("data/processed/splits")
    if not (manifest_dir / "classification_train.csv").exists():
        raise FileNotFoundError(
            "Split manifests missing! Run `python scripts/preprocess_data.py` first."
        )

    reports_dir = Path("reports/classification")
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("checkpoints/classification")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_classification_dataloaders(
        manifest_dir=manifest_dir,
        image_size=224,
        batch_size=batch_size,
        num_workers=0,
    )
    click.echo(
        f"DataLoaders Loaded: {len(train_loader.dataset)} Train, "
        f"{len(val_loader.dataset)} Val, {len(test_loader.dataset)} Test samples"
    )

    metadata_base = {
        "classes": CLASSIFICATION_CLASSES,
        "num_classes": 6,
        "image_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }

    # =========================================================================
    # 1. Train Custom CNN Baseline
    # =========================================================================
    click.echo("\n" + "=" * 60)
    click.echo("PHASE 1: Training Custom CNN Baseline...")
    click.echo("=" * 60)

    cnn_model = build_classification_model("custom_cnn", num_classes=6, dropout=0.3)
    cnn_opt = torch.optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    cnn_sched = CosineAnnealingLR(cnn_opt, T_max=epochs_cnn, eta_min=1e-5)
    cnn_early = EarlyStopping(patience=5, min_delta=0.005, mode="min")

    cnn_metadata = {**metadata_base, "architecture": "SolarCustomCNN", "lr": 1e-3}
    cnn_trainer = ClassificationTrainer(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=cnn_opt,
        scheduler=cnn_sched,
        device=device,
        early_stopping=cnn_early,
        checkpoint_dir=checkpoints_dir,
        model_name="custom_cnn",
        metadata=cnn_metadata,
    )

    t0_cnn = time.time()
    cnn_history = cnn_trainer.fit(epochs=epochs_cnn, verbose=True)
    cnn_train_time = time.time() - t0_cnn

    # Load best CNN weights for evaluation
    cnn_best_path = checkpoints_dir / "custom_cnn_best.pt"
    if cnn_best_path.exists():
        ckpt = torch.load(cnn_best_path, map_location=device)
        cnn_model.load_state_dict(ckpt["model_state_dict"])

    click.echo("\nEvaluating Custom CNN...")
    cnn_val_metrics = evaluate_classification(
        cnn_model, val_loader, device, CLASSIFICATION_CLASSES
    )
    cnn_test_metrics = evaluate_classification(
        cnn_model, test_loader, device, CLASSIFICATION_CLASSES
    )

    plot_training_history(
        cnn_history,
        reports_dir / "custom_cnn_history.html",
        title="Custom CNN Training Curves",
    )
    plot_confusion_matrix(
        cnn_test_metrics["confusion_matrix"],
        CLASSIFICATION_CLASSES,
        reports_dir / "custom_cnn_confusion_matrix.html",
        title="Custom CNN Test Confusion Matrix",
    )

    click.echo(
        f"Custom CNN — Val Acc: {cnn_val_metrics['accuracy']*100:.2f}% | "
        f"Test Acc: {cnn_test_metrics['accuracy']*100:.2f}% | Macro F1: {cnn_test_metrics['f1_macro']:.4f}"
    )

    # Clean up GPU/RAM before model 2
    del cnn_model, cnn_opt, cnn_sched, cnn_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # 2. Train MobileNetV2 (Transfer Learning)
    # =========================================================================
    click.echo("\n" + "=" * 60)
    click.echo("PHASE 2: Training MobileNetV2 (Pretrained Transfer Learning)...")
    click.echo("=" * 60)

    # Stage 1: Linear probe on frozen backbone
    click.echo("Stage 2.1: Frozen backbone linear probe (warm-up classifier head)...")
    mob_model = build_classification_model(
        "mobilenet_v2",
        num_classes=6,
        pretrained=True,
        freeze_backbone=True,
        dropout=0.3,
    )
    mob_opt1 = torch.optim.AdamW(
        [p for p in mob_model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )
    mob_trainer_stage1 = ClassificationTrainer(
        model=mob_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=mob_opt1,
        device=device,
        early_stopping=None,
        checkpoint_dir=checkpoints_dir,
        model_name="mobilenet_v2_probe",
        metadata={**metadata_base, "architecture": "SolarMobileNetV2_probe"},
    )
    mob_trainer_stage1.fit(epochs=4, verbose=True)

    # Stage 2: Fine-tuning top inverted residual blocks
    click.echo("\nStage 2.2: Fine-tuning top inverted residual blocks...")
    # Unfreeze last 4 blocks
    for i in range(max(0, len(mob_model.features) - 4), len(mob_model.features)):
        for param in mob_model.features[i].parameters():
            param.requires_grad = True

    mob_opt2 = torch.optim.AdamW(
        [p for p in mob_model.parameters() if p.requires_grad],
        lr=2e-4,
        weight_decay=1e-4,
    )
    mob_sched2 = CosineAnnealingLR(mob_opt2, T_max=epochs_mobilenet, eta_min=1e-6)
    mob_early2 = EarlyStopping(patience=5, min_delta=0.005, mode="min")

    mob_metadata = {
        **metadata_base,
        "architecture": "SolarMobileNetV2_finetuned",
        "lr": 2e-4,
    }
    mob_trainer = ClassificationTrainer(
        model=mob_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=mob_opt2,
        scheduler=mob_sched2,
        device=device,
        early_stopping=mob_early2,
        checkpoint_dir=checkpoints_dir,
        model_name="mobilenet_v2",
        metadata=mob_metadata,
    )

    t0_mob = time.time()
    mob_history = mob_trainer.fit(epochs=epochs_mobilenet, verbose=True)
    mob_train_time = time.time() - t0_mob

    # Load best MobileNetV2 weights for evaluation
    mob_best_path = checkpoints_dir / "mobilenet_v2_best.pt"
    if mob_best_path.exists():
        ckpt = torch.load(mob_best_path, map_location=device)
        mob_model.load_state_dict(ckpt["model_state_dict"])

    click.echo("\nEvaluating MobileNetV2...")
    mob_val_metrics = evaluate_classification(
        mob_model, val_loader, device, CLASSIFICATION_CLASSES
    )
    mob_test_metrics = evaluate_classification(
        mob_model, test_loader, device, CLASSIFICATION_CLASSES
    )

    plot_training_history(
        mob_history,
        reports_dir / "mobilenet_v2_history.html",
        title="MobileNetV2 Fine-Tuning Curves",
    )
    plot_confusion_matrix(
        mob_test_metrics["confusion_matrix"],
        CLASSIFICATION_CLASSES,
        reports_dir / "mobilenet_v2_confusion_matrix.html",
        title="MobileNetV2 Test Confusion Matrix",
    )

    click.echo(
        f"MobileNetV2 — Val Acc: {mob_val_metrics['accuracy']*100:.2f}% | "
        f"Test Acc: {mob_test_metrics['accuracy']*100:.2f}% | Macro F1: {mob_test_metrics['f1_macro']:.4f}"
    )

    # =========================================================================
    # 3. Save Comparative Report & Summary
    # =========================================================================
    report_md_path = reports_dir / "classification_report.md"
    generate_classification_report_markdown(
        custom_cnn_val=cnn_val_metrics,
        custom_cnn_test=cnn_test_metrics,
        mobilenet_val=mob_val_metrics,
        mobilenet_test=mob_test_metrics,
        class_names=CLASSIFICATION_CLASSES,
        output_path=report_md_path,
    )

    summary_json_path = reports_dir / "classification_summary.json"
    summary_data = {
        "timestamp": "2026-11-02",
        "seed": seed,
        "batch_size": batch_size,
        "device": str(device),
        "custom_cnn": {
            "training_time_seconds": round(cnn_train_time, 2),
            "checkpoint": str(cnn_best_path),
            "val_metrics": cnn_val_metrics,
            "test_metrics": cnn_test_metrics,
        },
        "mobilenet_v2": {
            "training_time_seconds": round(mob_train_time, 2),
            "checkpoint": str(mob_best_path),
            "val_metrics": mob_val_metrics,
            "test_metrics": mob_test_metrics,
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    click.echo("\n" + "=" * 60)
    click.echo("FINAL COMPARISON ON TEST SET:")
    click.echo("=" * 60)
    click.echo(
        f"1. Custom CNN:  Acc={cnn_test_metrics['accuracy']*100:.2f}% | "
        f"Macro F1={cnn_test_metrics['f1_macro']:.4f} | Time={cnn_train_time:.1f}s"
    )
    click.echo(
        f"2. MobileNetV2: Acc={mob_test_metrics['accuracy']*100:.2f}% | "
        f"Macro F1={mob_test_metrics['f1_macro']:.4f} | Time={mob_train_time:.1f}s"
    )
    diff = (mob_test_metrics["accuracy"] - cnn_test_metrics["accuracy"]) * 100
    click.echo(f"Delta: MobileNetV2 is {diff:+.2f}% over baseline")
    click.echo(f"\nSaved Report: {report_md_path}")
    click.echo(f"Saved Summary JSON: {summary_json_path}")
    click.echo("Step 5 Classification Training and Evaluation completed successfully!")


if __name__ == "__main__":
    main()
