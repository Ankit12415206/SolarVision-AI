"""
Module: schema
Description: SQLite database schema and index definitions for SolarVision AI inspection history
Author: SolarVision AI Team
Created: 2026-11-02
"""

CREATE_INSPECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id TEXT UNIQUE,
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
    is_demo INTEGER DEFAULT 0,
    details_json TEXT,
    notes TEXT
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_inspections_type ON inspections(inspection_type);",
    "CREATE INDEX IF NOT EXISTS idx_inspections_defect ON inspections(has_defect);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_inspections_uid ON inspections(inspection_id);",
    "CREATE INDEX IF NOT EXISTS idx_inspections_demo ON inspections(is_demo);",
]
