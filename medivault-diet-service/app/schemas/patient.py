"""
Pydantic v2 models for patient context data fetched from MediVault backend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MedicationInfo(BaseModel):
    name: str
    generic_name: Optional[str] = None
    dose: str
    frequency: str
    route: str = "oral"
    instructions: Optional[str] = None
    food_interactions: Optional[List[str]] = []


class LabResultSummary(BaseModel):
    test_name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    abnormal_flag: Optional[str] = None  # NORMAL / HIGH / LOW / CRITICAL
    reported_at: Optional[datetime] = None


class PatientContext(BaseModel):
    # Demographics
    patient_id: str
    name: str
    age: int
    gender: str
    blood_group: Optional[str] = None

    # Medical
    medical_conditions: List[str] = []
    current_medications: List[MedicationInfo] = []
    allergies: List[str] = []

    # Recent lab results (last 30 days)
    recent_lab_results: List[LabResultSummary] = []

    # Lifestyle
    dietary_restrictions: List[str] = []

    # Vector context from Atlas Vector Search (clinical document chunks)
    vector_context_chunks: List[Dict[str, Any]] = []

    # Metadata
    fetched_at: datetime
