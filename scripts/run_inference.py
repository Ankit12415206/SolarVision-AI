"""
Module: run_inference
Description: CLI runner for unified SolarVision AI inference pipeline across all modalities
Author: SolarVision AI Team
Created: 2026-11-02
"""

import json
from pathlib import Path

import click

from src.inference import InspectionType, SolarVisionInferencePipeline


@click.command()
@click.option(
    "--image",
    "-i",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to input image for inspection.",
)
@click.option(
    "--type",
    "-t",
    "inspection_type",
    type=click.Choice(
        ["classification", "detection", "segmentation"],
        case_sensitive=False,
    ),
    required=True,
    help="Inspection modality to execute.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("reports/inference"),
    show_default=True,
    help="Directory to save annotated visualization and JSON results.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.70,
    show_default=True,
    help="Binary segmentation probability threshold.",
)
@click.option(
    "--conf-threshold",
    type=float,
    default=0.10,
    show_default=True,
    help="Object detection confidence threshold.",
)
@click.option(
    "--iou-threshold",
    type=float,
    default=0.45,
    show_default=True,
    help="Object detection NMS IoU threshold.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Device for model execution ('cpu', 'cuda').",
)
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
    help="Save annotated visualization image and JSON result to output directory.",
)
def main(
    image: Path,
    inspection_type: str,
    output_dir: Path,
    threshold: float,
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    save: bool,
) -> None:
    """Run SolarVision AI unified inference pipeline on a solar panel inspection image."""
    click.echo("=" * 70)
    click.echo("  SolarVision AI -- Unified Inference Pipeline")
    click.echo("=" * 70)
    click.echo(f"Input image:      {image}")
    click.echo(f"Inspection type:  {inspection_type}")
    click.echo(f"Target device:    {device}")

    itype = InspectionType.from_str(inspection_type)
    pipeline = SolarVisionInferencePipeline(device=device)

    # Route inference
    kwargs = {}
    if itype == InspectionType.SEGMENTATION:
        kwargs["threshold"] = threshold
    elif itype == InspectionType.DETECTION:
        kwargs["conf_threshold"] = conf_threshold
        kwargs["iou_threshold"] = iou_threshold

    result = pipeline.predict(image, inspection_type=itype, **kwargs)

    click.echo("-" * 70)
    click.echo(f"Model:            {result.model_name} ({result.model_version})")
    click.echo(f"Processing time:  {result.processing_time_ms:.1f} ms")
    click.echo(
        f"Image dimensions: {result.image_metadata['width']}x{result.image_metadata['height']}"
    )

    if itype == InspectionType.CLASSIFICATION:
        click.echo(
            f"Predicted class:  {result.predicted_class} (confidence: {result.confidence * 100:.1f}%)"
        )
        click.echo("Class probabilities:")
        if result.class_probabilities:
            for cname, p in sorted(
                result.class_probabilities.items(), key=lambda x: x[1], reverse=True
            ):
                bar = "#" * int(p * 30)
                click.echo(f"  - {cname:20s}: {p * 100:5.1f}% [{bar:<30s}]")

    elif itype == InspectionType.DETECTION:
        num_dets = len(result.boxes) if result.boxes else 0
        click.echo(f"Detections found: {num_dets}")
        if result.boxes:
            for idx, box in enumerate(result.boxes, 1):
                click.echo(
                    f"  [{idx}] {box.class_name} (conf: {box.score:.2f}) at "
                    f"[{box.x1:.3f}, {box.y1:.3f}, {box.x2:.3f}, {box.y2:.3f}]"
                )

    elif itype == InspectionType.SEGMENTATION:
        has_defect = (result.defect_ratio or 0.0) > 0.0
        defect_pct = (result.defect_ratio or 0.0) * 100.0
        click.echo(f"Defect detected:  {'YES' if has_defect else 'NO'}")
        click.echo(f"Defect area:      {defect_pct:.4f}% of cell area")
        click.echo(f"Threshold used:   {result.threshold}")

    # Save outputs
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        img_out_path = output_dir / f"{image.stem}_{itype.value}_annotated.png"
        json_out_path = output_dir / f"{image.stem}_{itype.value}_result.json"

        if result.annotated_image:
            result.annotated_image.save(img_out_path)
            click.echo(f"Saved visualization: {img_out_path}")

        res_dict = result.to_dict(include_mask_array=False)
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=2)
        click.echo(f"Saved result JSON:   {json_out_path}")

    click.echo("=" * 70)


if __name__ == "__main__":
    main()
