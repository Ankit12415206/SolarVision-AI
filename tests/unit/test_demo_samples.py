"""
Module: test_demo_samples
Description: Unit tests for evaluator demo samples registry and path resolution
Author: SolarVision AI Team
Created: 2026-11-03
"""

from src.inference.types import InspectionType
from src.utils.demo_samples import (
    DEMO_SAMPLES,
    DemoSample,
    get_demo_samples_for_modality,
)


class TestDemoSamples:
    """Test suite for DemoSample registry and resolution."""

    def test_samples_registry_completeness(self) -> None:
        """Verify all demo samples contain valid non-empty fields and attributions."""
        assert len(DEMO_SAMPLES) >= 6

        modalities_found = set()
        for key, sample in DEMO_SAMPLES.items():
            assert isinstance(sample, DemoSample)
            assert sample.key == key
            assert len(sample.title) > 0
            assert len(sample.description) > 0
            assert len(sample.expected_finding) > 0
            assert len(sample.attribution) > 0
            modalities_found.add(sample.modality)

        assert InspectionType.CLASSIFICATION in modalities_found
        assert InspectionType.DETECTION in modalities_found
        assert InspectionType.SEGMENTATION in modalities_found

    def test_get_demo_samples_for_modality(self) -> None:
        """Verify filtering demo samples by inspection modality."""
        clf_samples = get_demo_samples_for_modality(InspectionType.CLASSIFICATION)
        det_samples = get_demo_samples_for_modality(InspectionType.DETECTION)
        seg_samples = get_demo_samples_for_modality(InspectionType.SEGMENTATION)

        assert len(clf_samples) >= 3
        assert len(det_samples) >= 2
        assert len(seg_samples) >= 2

        for s in clf_samples:
            assert s.modality == InspectionType.CLASSIFICATION
        for s in det_samples:
            assert s.modality == InspectionType.DETECTION
        for s in seg_samples:
            assert s.modality == InspectionType.SEGMENTATION

    def test_bundled_cc_by_samples_resolve(self) -> None:
        """Verify that all publicly bundled CC BY 4.0 demo samples resolve to valid files."""
        cc_by_samples = [
            s for s in DEMO_SAMPLES.values() if "CC BY 4.0" in s.attribution
        ]
        assert len(cc_by_samples) >= 4
        for sample in cc_by_samples:
            resolved = sample.resolve_path()
            assert resolved is not None, f"Sample '{sample.key}' could not be resolved."
            assert resolved.exists(), f"Path '{resolved}' does not exist on disk."
            assert resolved.stat().st_size > 1000

    def test_dataset_a_samples_excluded_from_public_bundle(self) -> None:
        """Verify that Dataset A samples are excluded from public demo bundle."""
        ds_a_samples = [
            s for s in DEMO_SAMPLES.values() if "Dataset A" in s.attribution
        ]
        assert len(ds_a_samples) == 4
        for s in ds_a_samples:
            assert not s.primary_path.exists(), (
                f"Unverified Dataset A sample '{s.primary_path}' must not exist in public bundle."
            )

    def test_missing_sample_returns_none_without_substitution(self) -> None:
        """Verify that non-existent sample paths return None rather than substituting."""
        from pathlib import Path

        dummy_sample = DemoSample(
            key="non_existent",
            title="Non-existent",
            modality=InspectionType.CLASSIFICATION,
            category="Classification",
            description="Testing missing file handling",
            expected_finding="None",
            attribution="None",
            primary_path=Path("data/demo/missing_file_xyz.jpg"),
            fallback_paths=[Path("data/raw/missing_fallback_xyz.jpg")],
        )
        assert dummy_sample.resolve_path() is None
