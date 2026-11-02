"""
Module: demo_samples
Description: Curated demo samples and metadata for evaluator-friendly demonstration
Author: SolarVision AI Team
Created: 2026-11-03
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.inference.types import InspectionType


@dataclass(frozen=True)
class DemoSample:
    """Metadata container for a curated evaluation demo sample."""

    key: str
    title: str
    modality: InspectionType
    category: str
    description: str
    expected_finding: str
    attribution: str
    primary_path: Path
    fallback_paths: List[Path]

    def resolve_path(self) -> Optional[Path]:
        """Resolve valid path on disk, trying primary then fallbacks.

        Returns:
            Optional[Path]: Existing Path object, or None if file cannot be found.
        """
        if self.primary_path.exists():
            return self.primary_path
        for fb in self.fallback_paths:
            if fb.exists():
                return fb
        return None


DEMO_SAMPLES: Dict[str, DemoSample] = {
    "cc_by_pv_array": DemoSample(
        key="cc_by_pv_array",
        title="[Classification] Aerial Solar Panel Array (CC BY 4.0 Verified)",
        modality=InspectionType.CLASSIFICATION,
        category="Classification",
        description="Verified CC BY 4.0 aerial RGB drone survey image of crystalline silicon PV panels.",
        expected_finding="Clean / Surface Anomaly",
        attribution="Dataset B (Roboflow / PMC11821171) — CC BY 4.0",
        primary_path=Path("data/demo/classification/aerial_pv_panel_cc_by.jpg"),
        fallback_paths=[Path("data/demo/detection/aerial_soiling_sample1.jpg")],
    ),
    "clean_panel": DemoSample(
        key="clean_panel",
        title="[Classification] Clean PV Module Surface (Local Dataset A)",
        modality=InspectionType.CLASSIFICATION,
        category="Classification",
        description="Nominal unblemished solar panel surface (requires local Dataset A download).",
        expected_finding="Clean",
        attribution="Dataset A (Kaggle: pythonafroz) — Unspecified / Web (Academic Fair Use)",
        primary_path=Path("data/demo/classification/clean_panel.jpg"),
        fallback_paths=[Path("data/raw/classification/Clean/Clean (10).JPG")],
    ),
    "dusty_panel": DemoSample(
        key="dusty_panel",
        title="[Classification] Dusty / Soiled Panel (Local Dataset A)",
        modality=InspectionType.CLASSIFICATION,
        category="Classification",
        description="Particulate dust accumulation reducing irradiance (requires local Dataset A).",
        expected_finding="Dusty",
        attribution="Dataset A (Kaggle: pythonafroz) — Unspecified / Web (Academic Fair Use)",
        primary_path=Path("data/demo/classification/dusty_panel.jpg"),
        fallback_paths=[Path("data/raw/classification/Dusty/Dust (10).jpg")],
    ),
    "bird_drop_panel": DemoSample(
        key="bird_drop_panel",
        title="[Classification] Bird-Drop Contamination (Local Dataset A)",
        modality=InspectionType.CLASSIFICATION,
        category="Classification",
        description="Localized opaque avian fouling creating hot-spot risks (requires local Dataset A).",
        expected_finding="Bird-drop",
        attribution="Dataset A (Kaggle: pythonafroz) — Unspecified / Web (Academic Fair Use)",
        primary_path=Path("data/demo/classification/bird_drop_panel.jpg"),
        fallback_paths=[Path("data/raw/classification/Bird-drop/10.JPG")],
    ),
    "damage_panel": DemoSample(
        key="damage_panel",
        title="[Classification] Physical Damage Glass (Local Dataset A)",
        modality=InspectionType.CLASSIFICATION,
        category="Classification",
        description="Mechanical impact fracture causing shattered glass (requires local Dataset A).",
        expected_finding="Physical-Damage",
        attribution="Dataset A (Kaggle: pythonafroz) — Unspecified / Web (Academic Fair Use)",
        primary_path=Path("data/demo/classification/physical_damage_panel.jpg"),
        fallback_paths=[
            Path("data/raw/classification/Physical-Damage/Physical (10).jpg")
        ],
    ),
    "aerial_detection_1": DemoSample(
        key="aerial_detection_1",
        title="[Detection] Aerial Drone Soiling Crop #1 (CC BY 4.0 Verified)",
        modality=InspectionType.DETECTION,
        category="Detection",
        description="Aerial drone RGB survey crop capturing discrete dust clusters on PV arrays.",
        expected_finding="Dust Bounding Boxes",
        attribution="Dataset B (Roboflow Aerial Soiling) — CC BY 4.0",
        primary_path=Path("data/demo/detection/aerial_soiling_sample1.jpg"),
        fallback_paths=[
            Path(
                "data/raw/dust/aerial_soiling/test/images/ROI_102_jpg.rf.fa9f2d998fd0b9874bc2a3a58394e73e.jpg"
            )
        ],
    ),
    "aerial_detection_2": DemoSample(
        key="aerial_detection_2",
        title="[Detection] Aerial Drone Soiling Crop #2 (CC BY 4.0 Verified)",
        modality=InspectionType.DETECTION,
        category="Detection",
        description="Aerial drone survey frame showing localized surface soiling patches.",
        expected_finding="Dust Bounding Boxes",
        attribution="Dataset B (Roboflow Aerial Soiling) — CC BY 4.0",
        primary_path=Path("data/demo/detection/aerial_soiling_sample2.jpg"),
        fallback_paths=[
            Path(
                "data/raw/dust/aerial_soiling/test/images/ROI_116_jpg.rf.ced8d5064d82fd9a7d327899502fb0e2.jpg"
            )
        ],
    ),
    "el_segmentation_1": DemoSample(
        key="el_segmentation_1",
        title="[Segmentation] EL Silicon Cell Microcrack #1 (CC BY 4.0 Verified)",
        modality=InspectionType.SEGMENTATION,
        category="Segmentation",
        description="Electroluminescence (EL) image of silicon cell showing internal wafer microcrack.",
        expected_finding="Microcrack Binary Mask",
        attribution="Dataset C (MSPS Benchmark, Zenodo) — CC BY 4.0",
        primary_path=Path("data/demo/segmentation/el_crack_sample1.png"),
        fallback_paths=[
            Path(
                "data/raw/microcracks/msps/test/images/First_101_20210705143317_5.png"
            )
        ],
    ),
    "el_segmentation_2": DemoSample(
        key="el_segmentation_2",
        title="[Segmentation] EL Silicon Cell Microcrack #2 (CC BY 4.0 Verified)",
        modality=InspectionType.SEGMENTATION,
        category="Segmentation",
        description="EL scan displaying dendritic microcrack branching across silicon busbars.",
        expected_finding="Microcrack Binary Mask",
        attribution="Dataset C (MSPS Benchmark, Zenodo) — CC BY 4.0",
        primary_path=Path("data/demo/segmentation/el_crack_sample2.png"),
        fallback_paths=[
            Path(
                "data/raw/microcracks/msps/test/images/First_101_20210706215956_5.png"
            )
        ],
    ),
}


def get_demo_samples_for_modality(modality: InspectionType) -> List[DemoSample]:
    """Retrieve all demo samples registered for a specific inspection modality.

    Args:
        modality: Target InspectionType.

    Returns:
        List of matching DemoSample objects.
    """
    return [s for s in DEMO_SAMPLES.values() if s.modality == modality]
