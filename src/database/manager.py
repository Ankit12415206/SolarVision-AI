"""
Module: manager
Description: SQLite database manager for SolarVision AI inspection history and analytics
Author: SolarVision AI Team
Created: 2026-11-02
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.database.schema import CREATE_INDEXES, CREATE_INSPECTIONS_TABLE
from src.inference.types import InferenceResult, InspectionType

DEFAULT_DB_PATH = Path("data/inspections.db")


def format_inspection_id(year: int, sequence_num: int) -> str:
    """Format a standard, human-readable unique Inspection ID.

    Args:
        year: Calendar year (e.g., 2026).
        sequence_num: Sequential record integer ID.

    Returns:
        str: Unique identifier like 'SV-2026-000001'.
    """
    return f"SV-{year}-{sequence_num:06d}"


class InspectionDatabaseManager:
    """Manages SQLite storage, querying, and analytics for solar panel inspections.

    Thread-safe implementation suitable for multi-user Streamlit sessions.
    """

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file. Defaults to data/inspections.db.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a SQLite connection configured with dict rows.

        Returns:
            sqlite3.Connection: SQLite connection object.
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables, run backward-compatible schema migrations, and build indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(CREATE_INSPECTIONS_TABLE)

            # Inspect existing columns to apply non-destructive backward-compatible migrations
            cursor.execute("PRAGMA table_info(inspections);")
            columns = {row["name"] for row in cursor.fetchall()}

            if "inspection_id" not in columns:
                cursor.execute("ALTER TABLE inspections ADD COLUMN inspection_id TEXT;")

            if "is_demo" not in columns:
                cursor.execute(
                    "ALTER TABLE inspections ADD COLUMN is_demo INTEGER DEFAULT 0;"
                )

            for index_sql in CREATE_INDEXES:
                try:
                    cursor.execute(index_sql)
                except sqlite3.OperationalError:
                    pass

            # Backfill any legacy records lacking an inspection_id
            cursor.execute(
                """
                UPDATE inspections
                SET inspection_id = 'SV-' || substr(timestamp, 1, 4) || '-' || printf('%06d', id)
                WHERE inspection_id IS NULL OR inspection_id = '';
                """
            )
            conn.commit()

    def save_inspection(
        self,
        result: InferenceResult,
        image_name: Optional[str] = None,
        image_path: Optional[str] = None,
        notes: str = "",
        is_demo: bool = False,
        inspection_id: Optional[str] = None,
    ) -> int:
        """Persist an InferenceResult to the SQLite database with unique traceability.

        Args:
            result: The InferenceResult to record.
            image_name: Display filename of the inspected image.
            image_path: Storage path or reference for the image.
            notes: Optional human inspection notes or tags.
            is_demo: True if run on a pre-bundled demo preset.
            inspection_id: Optional predefined unique Inspection ID.

        Returns:
            int: The inserted inspection record primary key ID.
        """
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        current_year = now_dt.year
        img_name = (
            image_name or Path(result.image_metadata.get("source", "unknown")).name
        )

        itype = (
            result.inspection_type.value
            if isinstance(result.inspection_type, InspectionType)
            else str(result.inspection_type)
        )

        has_defect = 0
        predicted_class = None
        confidence = None
        num_detections = 0
        defect_ratio = None

        if result.inspection_type == InspectionType.CLASSIFICATION:
            predicted_class = result.predicted_class
            confidence = result.confidence
            has_defect = (
                1 if predicted_class and predicted_class.lower() != "clean" else 0
            )

        elif result.inspection_type == InspectionType.DETECTION:
            num_detections = len(result.boxes) if result.boxes else 0
            has_defect = 1 if num_detections > 0 else 0

        elif result.inspection_type == InspectionType.SEGMENTATION:
            defect_ratio = result.defect_ratio
            has_defect = 1 if (defect_ratio or 0.0) > 0.0 else 0

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Determine next sequential identifier
            cursor.execute("SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM inspections")
            next_seq = cursor.fetchone()["next_id"]
            if inspection_id:
                cursor.execute(
                    "SELECT id FROM inspections WHERE inspection_id = ?",
                    (inspection_id,),
                )
                if cursor.fetchone() is not None:
                    uid = format_inspection_id(current_year, next_seq)
                else:
                    uid = inspection_id
            else:
                uid = format_inspection_id(current_year, next_seq)

            # Store generated inspection_id in metadata for downstream export
            res_dict = result.to_dict(include_mask_array=False)
            res_dict["inspection_id"] = uid
            res_dict["is_demo"] = is_demo
            details_json = json.dumps(res_dict)

            insert_sql = """
            INSERT INTO inspections (
                inspection_id, timestamp, image_name, image_path, inspection_type,
                model_name, model_version, processing_time_ms,
                predicted_class, confidence, num_detections,
                defect_ratio, has_defect, is_demo, details_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            cursor.execute(
                insert_sql,
                (
                    uid,
                    now_iso,
                    img_name,
                    image_path,
                    itype,
                    result.model_name,
                    result.model_version,
                    result.processing_time_ms,
                    predicted_class,
                    confidence,
                    num_detections,
                    defect_ratio,
                    has_defect,
                    1 if is_demo else 0,
                    details_json,
                    notes,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_next_inspection_id(self, year: Optional[int] = None) -> str:
        """Generate the next unique Inspection ID based on maximum record sequence.

        Args:
            year: Optional year. Defaults to current UTC year.

        Returns:
            str: Unique Inspection ID (e.g. 'SV-2026-000001').
        """
        yr = year or datetime.now(timezone.utc).year
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM inspections")
            next_seq = cursor.fetchone()["next_id"]
            return format_inspection_id(yr, next_seq)

    def get_inspections(
        self,
        limit: int = 100,
        offset: int = 0,
        inspection_type: Optional[str] = None,
        has_defect: Optional[bool] = None,
        search_query: Optional[str] = None,
        is_demo: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve filtered inspection history records.

        Args:
            limit: Maximum records to return. Defaults to 100.
            offset: Record offset for pagination. Defaults to 0.
            inspection_type: Filter by modality ('classification', 'detection', 'segmentation').
            has_defect: Filter by defect presence (True/False).
            search_query: Search substring matching image_name, notes, or inspection_id.
            is_demo: Optional filter by demo status (True: demo only, False: real uploads only).

        Returns:
            List[Dict[str, Any]]: List of matching records.
        """
        conditions: List[str] = []
        params: List[Any] = []

        if inspection_type:
            conditions.append("inspection_type = ?")
            params.append(inspection_type)

        if has_defect is not None:
            conditions.append("has_defect = ?")
            params.append(1 if has_defect else 0)

        if is_demo is not None:
            conditions.append("is_demo = ?")
            params.append(1 if is_demo else 0)

        if search_query:
            conditions.append(
                "(image_name LIKE ? OR notes LIKE ? OR inspection_id LIKE ?)"
            )
            q = f"%{search_query.strip()}%"
            params.extend([q, q, q])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query_sql = f"""
        SELECT * FROM inspections
        {where_clause}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query_sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_inspection_by_id(
        self, inspection_id: Union[int, str]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a single inspection record by primary key ID or unique Inspection ID.

        Args:
            inspection_id: The record primary key (int/str) or string UID (e.g. 'SV-2026-000001').

        Returns:
            Optional[Dict[str, Any]]: Inspection dictionary or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            id_str = str(inspection_id).strip()

            if id_str.startswith("SV-") or not id_str.isdigit():
                cursor.execute(
                    "SELECT * FROM inspections WHERE inspection_id = ?",
                    (id_str,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM inspections WHERE id = ? OR inspection_id = ?",
                    (int(id_str), id_str),
                )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_inspection(self, inspection_id: Union[int, str]) -> bool:
        """Delete an inspection record by primary key ID or unique Inspection ID.

        Args:
            inspection_id: Primary key or unique Inspection ID to delete.

        Returns:
            bool: True if a record was deleted, False otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            id_str = str(inspection_id).strip()
            if id_str.startswith("SV-") or not id_str.isdigit():
                cursor.execute(
                    "DELETE FROM inspections WHERE inspection_id = ?", (id_str,)
                )
            else:
                cursor.execute(
                    "DELETE FROM inspections WHERE id = ? OR inspection_id = ?",
                    (int(id_str), id_str),
                )
            conn.commit()
            return cursor.rowcount > 0

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Compute aggregated inspection statistics for the dashboard.

        Returns:
            Dict[str, Any]: Total counts, defect distribution, type breakdown, and trend data.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total, Defect counts, and Demo counts
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_count,
                    SUM(has_defect) AS defect_count,
                    SUM(is_demo) AS demo_count,
                    AVG(processing_time_ms) AS avg_latency
                FROM inspections
                """
            )
            row = cursor.fetchone()
            total = int(row["total_count"] or 0)
            defects = int(row["defect_count"] or 0)
            demos = int(row["demo_count"] or 0)
            avg_latency = float(row["avg_latency"] or 0.0)
            defect_rate = (defects / total * 100.0) if total > 0 else 0.0

            # Count by inspection type
            cursor.execute(
                """
                SELECT inspection_type, COUNT(*) AS count
                FROM inspections
                GROUP BY inspection_type
                """
            )
            by_type = {r["inspection_type"]: r["count"] for r in cursor.fetchall()}

            # Classification class distribution
            cursor.execute(
                """
                SELECT predicted_class, COUNT(*) AS count
                FROM inspections
                WHERE inspection_type = 'classification' AND predicted_class IS NOT NULL
                GROUP BY predicted_class
                ORDER BY count DESC
                """
            )
            by_class = {r["predicted_class"]: r["count"] for r in cursor.fetchall()}

            # Trend by date (YYYY-MM-DD)
            cursor.execute(
                """
                SELECT substr(timestamp, 1, 10) AS date_str, COUNT(*) AS count
                FROM inspections
                GROUP BY date_str
                ORDER BY date_str ASC
                LIMIT 30
                """
            )
            daily_trend = [
                {"date": r["date_str"], "count": r["count"]}
                for r in cursor.fetchall()
            ]

            return {
                "total_inspections": total,
                "defect_count": defects,
                "clean_count": total - defects,
                "demo_count": demos,
                "real_count": total - demos,
                "defect_rate_pct": round(defect_rate, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "inspections_by_type": by_type,
                "classification_distribution": by_class,
                "daily_trend": daily_trend,
            }

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export full inspection history table to a pandas DataFrame.

        Returns:
            pd.DataFrame: Complete inspection history dataframe.
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(
                """
                SELECT
                    id,
                    inspection_id,
                    timestamp,
                    image_name,
                    is_demo,
                    inspection_type,
                    model_name,
                    model_version,
                    processing_time_ms,
                    predicted_class,
                    confidence,
                    num_detections,
                    defect_ratio,
                    has_defect,
                    notes
                FROM inspections
                ORDER BY id DESC
                """,
                conn,
            )
