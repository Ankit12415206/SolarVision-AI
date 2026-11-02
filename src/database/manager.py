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
        """Create tables and performance indexes if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(CREATE_INSPECTIONS_TABLE)
            for index_sql in CREATE_INDEXES:
                cursor.execute(index_sql)
            conn.commit()

    def save_inspection(
        self,
        result: InferenceResult,
        image_name: Optional[str] = None,
        image_path: Optional[str] = None,
        notes: str = "",
    ) -> int:
        """Persist an InferenceResult to the SQLite database.

        Args:
            result: The InferenceResult to record.
            image_name: Display filename of the inspected image.
            image_path: Storage path or reference for the image.
            notes: Optional human inspection notes or tags.

        Returns:
            int: The inserted inspection record ID.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
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

        details_json = json.dumps(result.to_dict(include_mask_array=False))

        insert_sql = """
        INSERT INTO inspections (
            timestamp, image_name, image_path, inspection_type,
            model_name, model_version, processing_time_ms,
            predicted_class, confidence, num_detections,
            defect_ratio, has_defect, details_json, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                insert_sql,
                (
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
                    details_json,
                    notes,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_inspections(
        self,
        limit: int = 100,
        offset: int = 0,
        inspection_type: Optional[str] = None,
        has_defect: Optional[bool] = None,
        search_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve filtered inspection history records.

        Args:
            limit: Maximum records to return. Defaults to 100.
            offset: Record offset for pagination. Defaults to 0.
            inspection_type: Filter by modality ('classification', 'detection', 'segmentation').
            has_defect: Filter by defect presence (True/False).
            search_query: Search substring matching image_name or notes.

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

        if search_query:
            conditions.append("(image_name LIKE ? OR notes LIKE ?)")
            q = f"%{search_query.strip()}%"
            params.extend([q, q])

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

    def get_inspection_by_id(self, inspection_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single inspection record by ID.

        Args:
            inspection_id: The record primary key.

        Returns:
            Optional[Dict[str, Any]]: Inspection dictionary or None if not found.
        """
        query_sql = "SELECT * FROM inspections WHERE id = ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query_sql, (inspection_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_inspection(self, inspection_id: int) -> bool:
        """Delete an inspection record by ID.

        Args:
            inspection_id: Primary key of record to delete.

        Returns:
            bool: True if a record was deleted, False otherwise.
        """
        delete_sql = "DELETE FROM inspections WHERE id = ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(delete_sql, (inspection_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Compute aggregated inspection statistics for the dashboard.

        Returns:
            Dict[str, Any]: Total counts, defect distribution, type breakdown, and trend data.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total and Defect counts
            cursor.execute("""
                SELECT
                    COUNT(*) AS total_count,
                    SUM(has_defect) AS defect_count,
                    AVG(processing_time_ms) AS avg_latency
                FROM inspections
                """)
            row = cursor.fetchone()
            total = int(row["total_count"] or 0)
            defects = int(row["defect_count"] or 0)
            avg_latency = float(row["avg_latency"] or 0.0)
            defect_rate = (defects / total * 100.0) if total > 0 else 0.0

            # Count by inspection type
            cursor.execute("""
                SELECT inspection_type, COUNT(*) AS count
                FROM inspections
                GROUP BY inspection_type
                """)
            by_type = {r["inspection_type"]: r["count"] for r in cursor.fetchall()}

            # Classification class distribution
            cursor.execute("""
                SELECT predicted_class, COUNT(*) AS count
                FROM inspections
                WHERE inspection_type = 'classification' AND predicted_class IS NOT NULL
                GROUP BY predicted_class
                ORDER BY count DESC
                """)
            by_class = {r["predicted_class"]: r["count"] for r in cursor.fetchall()}

            # Trend by date (YYYY-MM-DD)
            cursor.execute("""
                SELECT substr(timestamp, 1, 10) AS date_str, COUNT(*) AS count
                FROM inspections
                GROUP BY date_str
                ORDER BY date_str ASC
                LIMIT 30
                """)
            daily_trend = [
                {"date": r["date_str"], "count": r["count"]} for r in cursor.fetchall()
            ]

            return {
                "total_inspections": total,
                "defect_count": defects,
                "clean_count": total - defects,
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
            return pd.read_sql_query("SELECT * FROM inspections ORDER BY id DESC", conn)
