"""
Fetches all relevant patient medical data from the MediVault main backend API.
Called once at session creation and the result is stored in MongoDB.

Endpoints called on the MediVault backend (all require the patient's JWT):
  GET /api/v1/patients/me                           — demographics, allergies
  GET /api/v1/encounters/mine                       — encounter history
  GET /api/v1/prescriptions/patient/:id/active      — active medications
  GET /api/v1/diagnoses/encounter/:encounterId      — diagnoses (for latest encounter)
  GET /api/v1/lab/orders/patient/:id                — recent lab orders
  GET /api/v1/lab/results/:labOrderId               — lab results (for recent orders)

All calls use httpx.AsyncClient with the patient's JWT as Bearer token.
Set a 10-second timeout on all calls.
If a call fails (not the patient/me call), log the error and continue with
partial data — never crash session creation due to missing lab results.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx
import structlog

from app.config.database import get_db
from app.config.settings import get_settings
from app.schemas.patient import (
    LabResultSummary,
    MedicationInfo,
    PatientContext,
)
from app.services.vector_context import fetch_vector_context

logger = structlog.get_logger("patient_context")

# Known drug-food interactions database (simplified)
DRUG_FOOD_INTERACTIONS: Dict[str, List[str]] = {
    "warfarin": [
        "Avoid large amounts of vitamin K-rich foods (spinach, kale, broccoli)",
        "Avoid cranberry juice — increases bleeding risk",
        "Limit alcohol — increases bleeding risk",
    ],
    "metformin": [
        "Avoid excessive alcohol — risk of lactic acidosis",
        "Take with food to reduce GI side effects",
    ],
    "atorvastatin": [
        "Avoid grapefruit and grapefruit juice — increases drug levels",
    ],
    "rosuvastatin": [
        "Avoid grapefruit and grapefruit juice — increases drug levels",
    ],
    "simvastatin": [
        "Avoid grapefruit and grapefruit juice — increases drug levels significantly",
    ],
    "lisinopril": [
        "Avoid potassium-rich foods in excess (bananas, oranges) — risk of hyperkalemia",
        "Avoid salt substitutes containing potassium",
    ],
    "enalapril": [
        "Avoid potassium-rich foods in excess — risk of hyperkalemia",
    ],
    "amlodipine": [
        "Avoid grapefruit — may increase drug levels",
    ],
    "levothyroxine": [
        "Take on empty stomach, 30-60 min before food",
        "Avoid calcium and iron supplements within 4 hours",
        "Avoid soy products close to dosing time",
    ],
    "ciprofloxacin": [
        "Avoid dairy products and calcium-fortified foods within 2 hours of dose",
        "Avoid caffeine — drug increases caffeine levels",
    ],
    "methotrexate": [
        "Avoid alcohol completely — liver toxicity risk",
        "Ensure adequate folate intake",
    ],
    "prednisone": [
        "Take with food to reduce stomach irritation",
        "Monitor sodium and potassium intake",
        "Increase calcium and vitamin D intake",
    ],
    "aspirin": [
        "Take with food to reduce GI irritation",
        "Avoid alcohol — increases bleeding risk",
    ],
    "omeprazole": [
        "Take before meals",
        "May reduce absorption of calcium, magnesium, vitamin B12 long-term",
    ],
    "insulin": [
        "Coordinate meal timing with dosing schedule",
        "Consistent carbohydrate intake at meals is important",
        "Avoid skipping meals — risk of hypoglycemia",
    ],
    "clopidogrel": [
        "Avoid grapefruit — may reduce drug effectiveness",
    ],
}


def _enrich_medication_interactions(med: MedicationInfo) -> MedicationInfo:
    """Add known drug-food interactions to a medication entry."""
    name_lower = med.name.lower().strip()
    generic_lower = (med.generic_name or "").lower().strip()

    interactions: List[str] = []
    for drug, notes in DRUG_FOOD_INTERACTIONS.items():
        if drug in name_lower or drug in generic_lower:
            interactions.extend(notes)

    if interactions:
        med.food_interactions = interactions
    return med


def _calculate_age(dob: str | None) -> int:
    """Calculate age from date of birth string."""
    if not dob:
        return 0
    try:
        birth = datetime.fromisoformat(dob.replace("Z", "+00:00"))
        today = datetime.now(timezone.utc)
        age = today.year - birth.year
        if (today.month, today.day) < (birth.month, birth.day):
            age -= 1
        return age
    except (ValueError, TypeError):
        return 0


async def fetch_patient_context(patient_id: str, jwt_token: str) -> PatientContext:
    """
    Fetch the complete patient context from MediVault backend.
    Makes multiple API calls. The patient/me call is critical; others are non-critical.
    """
    settings = get_settings()
    base_url = settings.medivault_api_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {jwt_token}"}
    timeout = httpx.Timeout(10.0)

    patient_data: Dict[str, Any] = {}
    encounters: List[Dict[str, Any]] = []
    prescriptions: List[Dict[str, Any]] = []
    diagnoses: List[str] = []
    lab_results: List[LabResultSummary] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        # 1. Fetch patient demographics (critical — fail if this fails)
        try:
            resp = await client.get(f"{base_url}/patients/me", headers=headers)
            resp.raise_for_status()
            body = resp.json()
            patient_data = body.get("data", body.get("patient", body))
        except Exception as exc:
            logger.warning(
                "patient_api_fetch_failed_trying_db_fallback",
                patient_id=patient_id,
                error=str(exc),
            )
            # ── Direct MongoDB fallback ──────────────────────────────
            try:
                from bson import ObjectId

                db = get_db()
                if db is None:
                    raise ValueError("Database not initialized")
                main_db_name = settings.medivault_db_name if hasattr(settings, 'medivault_db_name') else "medivault"
                main_db = db.client[main_db_name]
                patient_doc = await main_db.patients.find_one(
                    {"_id": ObjectId(patient_id)}
                )
                if patient_doc is None:
                    raise ValueError(
                        f"Could not fetch patient data from MediVault backend: {exc}"
                    )
                patient_data = patient_doc
                logger.info(
                    "patient_fetched_from_db_fallback",
                    patient_id=patient_id,
                )
            except ValueError:
                raise
            except Exception as db_exc:
                logger.error(
                    "patient_db_fallback_also_failed",
                    patient_id=patient_id,
                    error=str(db_exc),
                )
                raise ValueError(
                    f"Could not fetch patient data from MediVault backend: {exc}"
                ) from exc

        # 2. Fetch encounters (non-critical)
        try:
            resp = await client.get(f"{base_url}/encounters/mine", headers=headers)
            resp.raise_for_status()
            body = resp.json()
            encounters = body.get("data", body.get("encounters", []))
            if isinstance(encounters, dict):
                encounters = encounters.get("encounters", [])
        except Exception as exc:
            logger.warning(
                "encounters_fetch_failed",
                patient_id=patient_id,
                error=str(exc),
            )

        # 3. Fetch active prescriptions (non-critical)
        try:
            resp = await client.get(
                f"{base_url}/prescriptions/patient/{patient_id}/active",
                headers=headers,
            )
            resp.raise_for_status()
            body = resp.json()
            prescriptions = body.get("data", body.get("prescriptions", []))
            if isinstance(prescriptions, dict):
                prescriptions = prescriptions.get("prescriptions", [])
        except Exception as exc:
            logger.warning(
                "prescriptions_fetch_failed",
                patient_id=patient_id,
                error=str(exc),
            )

        # 4. Fetch diagnoses from ALL encounters (non-critical)
        #    Aggregate unique diagnoses across all encounters to build a
        #    complete medical history for the LLM.
        seen_diagnoses: set = set()
        for enc in encounters:
            encounter_id = enc.get("_id", enc.get("id", ""))
            if not encounter_id:
                continue
            try:
                resp = await client.get(
                    f"{base_url}/diagnoses/encounter/{encounter_id}",
                    headers=headers,
                )
                resp.raise_for_status()
                body = resp.json()
                diag_list = body.get("data", body.get("diagnoses", []))
                if isinstance(diag_list, dict):
                    diag_list = diag_list.get("diagnoses", [])
                for d in diag_list:
                    desc = d.get(
                        "description", d.get("icd_description", "")
                    )
                    if desc and desc not in seen_diagnoses:
                        seen_diagnoses.add(desc)
                        diagnoses.append(desc)
            except Exception as exc:
                logger.warning(
                    "diagnoses_fetch_failed",
                    patient_id=patient_id,
                    encounter_id=encounter_id,
                    error=str(exc),
                )

        # 5. Fetch lab orders and results (non-critical)
        try:
            resp = await client.get(
                f"{base_url}/lab/orders/patient/{patient_id}",
                headers=headers,
            )
            resp.raise_for_status()
            body = resp.json()
            lab_orders = body.get("data", body.get("labOrders", []))
            if isinstance(lab_orders, dict):
                lab_orders = lab_orders.get("labOrders", [])

            # Filter to last 30 days
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)

            for order in lab_orders:
                order_date_str = order.get(
                    "createdAt", order.get("ordered_at", "")
                )
                try:
                    order_date = datetime.fromisoformat(
                        order_date_str.replace("Z", "+00:00")
                    )
                    if order_date < thirty_days_ago:
                        continue
                except (ValueError, TypeError):
                    continue

                lab_order_id = order.get("_id", order.get("id", ""))
                if not lab_order_id:
                    continue

                # Fetch results for this order
                try:
                    resp = await client.get(
                        f"{base_url}/lab/results/{lab_order_id}",
                        headers=headers,
                    )
                    resp.raise_for_status()
                    result_body = resp.json()
                    results = result_body.get(
                        "data", result_body.get("results", [])
                    )
                    if isinstance(results, dict):
                        results = [results]

                    for r in results:
                        lab_results.append(
                            LabResultSummary(
                                test_name=r.get(
                                    "test_name", r.get("testName", "Unknown")
                                ),
                                value=str(
                                    r.get("value", r.get("result", ""))
                                ),
                                unit=r.get("unit", r.get("units", None)),
                                reference_range=r.get(
                                    "reference_range",
                                    r.get("referenceRange", None),
                                ),
                                abnormal_flag=r.get(
                                    "abnormal_flag", r.get("flag", None)
                                ),
                                reported_at=order_date,
                            )
                        )
                except Exception as exc:
                    logger.warning(
                        "lab_results_fetch_failed",
                        lab_order_id=lab_order_id,
                        error=str(exc),
                    )
        except Exception as exc:
            logger.warning(
                "lab_orders_fetch_failed",
                patient_id=patient_id,
                error=str(exc),
            )

    # Build medications list
    medications: List[MedicationInfo] = []
    for rx in prescriptions:
        # Backend may embed items in 'items', 'medications', or 'drugs'
        meds = rx.get("items", rx.get("medications", rx.get("drugs", [])))
        if isinstance(meds, dict):
            meds = [meds]
        for med_data in meds:
            # drug_id can be an object (populated) or a string
            drug_ref = med_data.get("drug_id", {})
            if isinstance(drug_ref, dict):
                drug_name = drug_ref.get(
                    "name",
                    drug_ref.get("brand_name",
                                 med_data.get("name",
                                              med_data.get("drugName", "Unknown")))
                )
                generic = drug_ref.get(
                    "generic_name",
                    med_data.get("generic_name",
                                 med_data.get("genericName"))
                )
            else:
                drug_name = med_data.get("name", med_data.get("drugName", "Unknown"))
                generic = med_data.get("generic_name", med_data.get("genericName"))

            med = MedicationInfo(
                name=drug_name,
                generic_name=generic,
                dose=str(med_data.get("dose", med_data.get("dosage", ""))),
                frequency=med_data.get("frequency", ""),
                route=med_data.get("route", "oral"),
                instructions=med_data.get(
                    "instructions", med_data.get("notes")
                ),
            )
            med = _enrich_medication_interactions(med)
            medications.append(med)

    # Extract patient info
    name = patient_data.get("name", {})
    if isinstance(name, dict) and (name.get("firstName") or name.get("lastName")):
        full_name = (
            f"{name.get('firstName', '')} {name.get('lastName', '')}".strip()
        )
    elif isinstance(name, str) and name.strip():
        full_name = name
    else:
        # Fallback: check top-level first_name / last_name fields
        first = patient_data.get("first_name", patient_data.get("firstName", ""))
        last = patient_data.get("last_name", patient_data.get("lastName", ""))
        full_name = f"{first} {last}".strip() or "Patient"

    allergies: List[str] = patient_data.get(
        "allergens", patient_data.get("allergies", [])
    )
    if isinstance(allergies, str):
        allergies = [a.strip() for a in allergies.split(",") if a.strip()]

    context = PatientContext(
        patient_id=patient_id,
        name=full_name,
        age=_calculate_age(
            patient_data.get("dob", patient_data.get("dateOfBirth"))
        ),
        gender=patient_data.get("gender", "Unknown"),
        blood_group=patient_data.get(
            "bloodGroup", patient_data.get("blood_group")
        ),
        medical_conditions=diagnoses,
        current_medications=medications,
        allergies=allergies,
        recent_lab_results=lab_results,
        dietary_restrictions=[],
        fetched_at=datetime.now(timezone.utc),
    )

    logger.info(
        "patient_context_fetched",
        patient_id=patient_id,
        conditions_count=len(diagnoses),
        medications_count=len(medications),
        lab_results_count=len(lab_results),
    )

    # ── Vector context enrichment ─────────────────────────────────────
    # Query MongoDB Atlas Vector Search for narrative clinical documents
    # (discharge summaries, prescription details, doctor's notes, etc.)
    try:
        db = get_db()
        vector_chunks = await fetch_vector_context(
            patient_id=patient_id,
            query=(
                "medical conditions diet food restrictions allergies "
                "medications nutrition recommendations"
            ),
            db=db,
            settings=settings,
            top_k=settings.vector_search_num_results,
        )
        context.vector_context_chunks = vector_chunks
        logger.info(
            "vector_context_enrichment",
            patient_id=patient_id,
            chunks_retrieved=len(vector_chunks),
        )
    except Exception as exc:
        logger.warning(
            "vector_context_enrichment_failed",
            patient_id=patient_id,
            error=str(exc),
        )
        # Non-critical — continue without vector context

    return context
