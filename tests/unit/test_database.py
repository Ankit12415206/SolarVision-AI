"""
Module: test_database
Description: Unit tests for SQLite inspection database manager and analytics
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import pytest

from src.database.manager import InspectionDatabaseManager
from src.inference.types import BoundingBox, InferenceResult, InspectionType


@pytest.fixture
def db_manager(tmp_path: Path) -> InspectionDatabaseManager:
    """Provide isolated SQLite database manager instance in temporary folder."""
    db_file = tmp_path / "test_inspections.db"
    return InspectionDatabaseManager(db_path=db_file)


class TestInspectionDatabaseManager:
    """Test suite for InspectionDatabaseManager operations."""

    def test_save_and_retrieve_classification(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify saving and querying a classification inspection."""
        res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=85.2,
            image_metadata={"width": 640, "height": 480, "source": "test_panel.jpg"},
            predicted_class="Dusty",
            confidence=0.92,
            class_probabilities={"Dusty": 0.92, "Clean": 0.08},
        )

        rec_id = db_manager.save_inspection(
            result=res,
            image_name="test_panel.jpg",
            notes="Zone A solar farm",
        )
        assert rec_id > 0

        record = db_manager.get_inspection_by_id(rec_id)
        assert record is not None
        assert record["image_name"] == "test_panel.jpg"
        assert record["inspection_type"] == "classification"
        assert record["predicted_class"] == "Dusty"
        assert record["confidence"] == 0.92
        assert record["has_defect"] == 1
        assert record["notes"] == "Zone A solar farm"

    def test_save_and_retrieve_detection(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify saving and querying a detection inspection."""
        boxes = [
            BoundingBox(0.1, 0.2, 0.3, 0.4, 0.88, 0, "Dust"),
        ]
        res = InferenceResult(
            inspection_type=InspectionType.DETECTION,
            model_name="SolarYOLOv8n",
            model_version="v1.1",
            processing_time_ms=120.0,
            image_metadata={"width": 640, "height": 640, "source": "drone_aerial.jpg"},
            boxes=boxes,
        )

        rec_id = db_manager.save_inspection(res, image_name="drone_aerial.jpg")
        assert rec_id > 0

        record = db_manager.get_inspection_by_id(rec_id)
        assert record is not None
        assert record["inspection_type"] == "detection"
        assert record["num_detections"] == 1
        assert record["has_defect"] == 1

    def test_save_and_retrieve_segmentation(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify saving and querying a segmentation inspection."""
        res = InferenceResult(
            inspection_type=InspectionType.SEGMENTATION,
            model_name="SolarLightweightUNet",
            model_version="v1.0",
            processing_time_ms=55.4,
            image_metadata={"width": 256, "height": 256, "source": "el_sample.png"},
            defect_ratio=0.0035,
            threshold=0.70,
        )

        rec_id = db_manager.save_inspection(res, image_name="el_sample.png")
        assert rec_id > 0

        record = db_manager.get_inspection_by_id(rec_id)
        assert record is not None
        assert record["inspection_type"] == "segmentation"
        assert record["defect_ratio"] == 0.0035
        assert record["has_defect"] == 1

    def test_filtering_and_search(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify filtering by type, defect, and search keywords."""
        # 1. Clean panel
        clean_res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=50.0,
            image_metadata={"width": 100, "height": 100, "source": "clean_panel.jpg"},
            predicted_class="Clean",
            confidence=0.99,
        )
        db_manager.save_inspection(clean_res, image_name="clean_panel.jpg")

        # 2. Defect panel
        defect_res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=52.0,
            image_metadata={"width": 100, "height": 100, "source": "broken_panel.jpg"},
            predicted_class="Physical-Damage",
            confidence=0.89,
        )
        db_manager.save_inspection(
            defect_res, image_name="broken_panel.jpg", notes="Severe hail damage"
        )

        # Query all
        all_recs = db_manager.get_inspections()
        assert len(all_recs) == 2

        # Query only defects
        defects = db_manager.get_inspections(has_defect=True)
        assert len(defects) == 1
        assert defects[0]["image_name"] == "broken_panel.jpg"

        # Query only clean
        cleans = db_manager.get_inspections(has_defect=False)
        assert len(cleans) == 1
        assert cleans[0]["image_name"] == "clean_panel.jpg"

        # Search keyword
        search_res = db_manager.get_inspections(search_query="hail")
        assert len(search_res) == 1
        assert search_res[0]["image_name"] == "broken_panel.jpg"

    def test_delete_inspection(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify record deletion."""
        res = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=50.0,
            image_metadata={"width": 100, "height": 100, "source": "p.jpg"},
            predicted_class="Clean",
            confidence=0.99,
        )
        rec_id = db_manager.save_inspection(res, image_name="p.jpg")
        assert db_manager.delete_inspection(rec_id) is True
        assert db_manager.get_inspection_by_id(rec_id) is None
        assert db_manager.delete_inspection(99999) is False

    def test_analytics_summary(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify aggregated analytics calculation."""
        # Empty summary
        empty_summary = db_manager.get_analytics_summary()
        assert empty_summary["total_inspections"] == 0
        assert empty_summary["defect_rate_pct"] == 0.0

        # Insert 1 clean and 1 defect
        r1 = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=100.0,
            image_metadata={"width": 100, "height": 100, "source": "p1.jpg"},
            predicted_class="Clean",
            confidence=0.95,
        )
        r2 = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=200.0,
            image_metadata={"width": 100, "height": 100, "source": "p2.jpg"},
            predicted_class="Dusty",
            confidence=0.91,
        )
        db_manager.save_inspection(r1, image_name="p1.jpg")
        db_manager.save_inspection(r2, image_name="p2.jpg")

        summary = db_manager.get_analytics_summary()
        assert summary["total_inspections"] == 2
        assert summary["defect_count"] == 1
        assert summary["clean_count"] == 1
        assert summary["defect_rate_pct"] == 50.0
        assert summary["avg_latency_ms"] == 150.0
        assert summary["classification_distribution"]["Dusty"] == 1

    def test_export_to_dataframe(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify exporting history to pandas DataFrame."""
        df_empty = db_manager.export_to_dataframe()
        assert len(df_empty) == 0

        r = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=60.0,
            image_metadata={"width": 100, "height": 100, "source": "p.jpg"},
            predicted_class="Clean",
            confidence=0.95,
        )
        db_manager.save_inspection(r, image_name="p.jpg")
        df = db_manager.export_to_dataframe()
        assert len(df) == 1
        assert "timestamp" in df.columns
        assert "predicted_class" in df.columns
        assert "inspection_id" in df.columns

    def test_inspection_id_generation_and_uniqueness(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify unique readable Inspection ID generation and lookup."""
        r1 = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=50.0,
            image_metadata={"width": 100, "height": 100, "source": "panel1.jpg"},
            predicted_class="Clean",
            confidence=0.95,
        )
        r2 = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=55.0,
            image_metadata={"width": 100, "height": 100, "source": "panel2.jpg"},
            predicted_class="Dusty",
            confidence=0.88,
        )

        id1 = db_manager.save_inspection(r1, image_name="panel1.jpg")
        id2 = db_manager.save_inspection(r2, image_name="panel2.jpg")

        rec1 = db_manager.get_inspection_by_id(id1)
        rec2 = db_manager.get_inspection_by_id(id2)

        assert rec1 is not None and rec2 is not None
        uid1 = rec1["inspection_id"]
        uid2 = rec2["inspection_id"]

        assert uid1.startswith("SV-")
        assert uid2.startswith("SV-")
        assert uid1 != uid2

        # Retrieve by string UID
        fetched_by_uid1 = db_manager.get_inspection_by_id(uid1)
        assert fetched_by_uid1 is not None
        assert fetched_by_uid1["id"] == id1

        # Search by UID substring
        search_results = db_manager.get_inspections(search_query=uid1)
        assert len(search_results) == 1
        assert search_results[0]["inspection_id"] == uid1

    def test_demo_flag_persistence_and_filtering(
        self,
        db_manager: InspectionDatabaseManager,
    ) -> None:
        """Verify is_demo flag persistence and querying."""
        r_real = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=60.0,
            image_metadata={"width": 100, "height": 100, "source": "user_upload.jpg"},
            predicted_class="Clean",
            confidence=0.99,
        )
        r_demo = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=62.0,
            image_metadata={"width": 100, "height": 100, "source": "demo_sample.jpg"},
            predicted_class="Dusty",
            confidence=0.85,
        )

        id_real = db_manager.save_inspection(
            r_real, image_name="user_upload.jpg", is_demo=False
        )
        id_demo = db_manager.save_inspection(
            r_demo, image_name="demo_sample.jpg", is_demo=True
        )

        assert db_manager.get_inspection_by_id(id_real)["is_demo"] == 0
        assert db_manager.get_inspection_by_id(id_demo)["is_demo"] == 1

        demos = db_manager.get_inspections(is_demo=True)
        assert len(demos) == 1
        assert demos[0]["image_name"] == "demo_sample.jpg"

        reals = db_manager.get_inspections(is_demo=False)
        assert len(reals) == 1
        assert reals[0]["image_name"] == "user_upload.jpg"

        analytics = db_manager.get_analytics_summary()
        assert analytics["demo_count"] == 1
        assert analytics["real_count"] == 1

    def test_legacy_database_migration_preserves_records(
        self, tmp_path: Path
    ) -> None:
        """Verify non-destructive migration of legacy database without inspection_id."""
        import sqlite3

        legacy_db = tmp_path / "legacy_v1.db"
        # 1. Build legacy schema
        with sqlite3.connect(str(legacy_db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    image_path TEXT,
                    inspection_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    processing_time_ms REAL NOT NULL,
                    predicted_class TEXT,
                    confidence REAL,
                    num_detections INTEGER DEFAULT 0,
                    defect_ratio REAL,
                    has_defect INTEGER NOT NULL DEFAULT 0,
                    details_json TEXT,
                    notes TEXT
                );
                """
            )
            # Insert 3 legacy records
            cur.execute(
                """
                INSERT INTO inspections (
                    timestamp, image_name, inspection_type, model_name,
                    model_version, processing_time_ms, predicted_class,
                    confidence, has_defect, notes
                ) VALUES
                ('2025-10-12T08:00:00Z', 'legacy1.jpg', 'classification', 'ModelA', '1.0', 70.0, 'Clean', 0.95, 0, 'Legacy 2025'),
                ('2026-01-05T09:00:00Z', 'legacy2.jpg', 'classification', 'ModelA', '1.0', 75.0, 'Dusty', 0.88, 1, 'Legacy 2026-1'),
                ('2026-02-14T10:00:00Z', 'legacy3.jpg', 'classification', 'ModelA', '1.0', 80.0, 'Clean', 0.92, 0, 'Legacy 2026-2')
                """
            )
            conn.commit()

        # 2. Instantiate manager on legacy DB
        mgr = InspectionDatabaseManager(db_path=legacy_db)

        # 3. Confirm all 3 records preserved (no loss or duplication)
        summary = mgr.get_analytics_summary()
        assert summary["total_inspections"] == 3

        recs = mgr.get_inspections(limit=10)
        assert len(recs) == 3

        # 4. Confirm backfilled IDs
        r1 = mgr.get_inspection_by_id(1)
        r2 = mgr.get_inspection_by_id(2)
        r3 = mgr.get_inspection_by_id(3)
        assert r1 is not None and r2 is not None and r3 is not None
        assert r1["inspection_id"] == "SV-2025-000001"
        assert r2["inspection_id"] == "SV-2026-000002"
        assert r3["inspection_id"] == "SV-2026-000003"
        assert r1["is_demo"] == 0
        assert r2["is_demo"] == 0

        # 5. Lookup by newly backfilled string UID
        fetched = mgr.get_inspection_by_id("SV-2025-000001")
        assert fetched is not None
        assert fetched["image_name"] == "legacy1.jpg"

    def test_id_collision_prevention_on_duplicate_hint(
        self, db_manager: InspectionDatabaseManager
    ) -> None:
        """Verify that passing an existing inspection_id triggers collision resolution."""
        r = InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name="SolarMobileNetV2",
            model_version="v1.0",
            processing_time_ms=60.0,
            image_metadata={"width": 100, "height": 100, "source": "test.jpg"},
            predicted_class="Clean",
            confidence=0.99,
        )

        id1 = db_manager.save_inspection(r, inspection_id="SV-2026-000099")
        rec1 = db_manager.get_inspection_by_id(id1)
        assert rec1["inspection_id"] == "SV-2026-000099"

        # Attempt to insert with duplicate ID
        id2 = db_manager.save_inspection(r, inspection_id="SV-2026-000099")
        rec2 = db_manager.get_inspection_by_id(id2)
        assert rec2["inspection_id"] != "SV-2026-000099"
        assert rec2["inspection_id"].startswith("SV-2026-")
