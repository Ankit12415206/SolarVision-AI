"""
Module: database
Description: SQLite database management and schemas for SolarVision AI
Author: SolarVision AI Team
Created: 2026-11-02
"""

from src.database.manager import DEFAULT_DB_PATH, InspectionDatabaseManager
from src.database.schema import CREATE_INDEXES, CREATE_INSPECTIONS_TABLE

__all__ = [
    "InspectionDatabaseManager",
    "DEFAULT_DB_PATH",
    "CREATE_INSPECTIONS_TABLE",
    "CREATE_INDEXES",
]
