"""
Response parser — extracts structured diet plan data from the LLM's
Markdown response.

Attempts to parse the 7-day diet plan tables into a structured DietPlan
object. If parsing fails, falls back to returning the raw text plan.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger("utils.response_parser")


def parse_diet_plan(raw_text: str) -> Dict[str, Any]:
    """
    Attempt to extract structured diet-plan data from the LLM's
    Markdown response.

    Returns a dict with:
        - raw_text: the original Markdown (always present)
        - days: list of per-day dicts if extraction succeeded
        - daily_summary: nutritional summary if found
        - parsed: bool indicating if structured data was extracted

    This is best-effort — the LLM output is non-deterministic, so we
    gracefully degrade to just the raw text.
    """
    result: Dict[str, Any] = {
        "raw_text": raw_text,
        "parsed": False,
        "days": [],
        "daily_summary": None,
        "clinical_notes": None,
    }

    try:
        days = _extract_days(raw_text)
        if days:
            result["days"] = days
            result["parsed"] = True

        summary = _extract_nutritional_summary(raw_text)
        if summary:
            result["daily_summary"] = summary

        notes = _extract_clinical_notes(raw_text)
        if notes:
            result["clinical_notes"] = notes

    except Exception as exc:
        logger.warning("diet_plan_parse_error", error=str(exc))

    return result


def _extract_days(text: str) -> List[Dict[str, Any]]:
    """
    Look for day headers (e.g. "## Day 1", "### Day 1 — Monday")
    and extract the table rows underneath each.
    """
    day_pattern = re.compile(
        r"#{2,3}\s*Day\s*(\d+)[^\n]*",
        re.IGNORECASE,
    )
    day_matches = list(day_pattern.finditer(text))
    if not day_matches:
        return []

    days: List[Dict[str, Any]] = []

    for i, match in enumerate(day_matches):
        day_num = int(match.group(1))
        start = match.end()
        end = day_matches[i + 1].start() if i + 1 < len(day_matches) else len(text)
        day_section = text[start:end]

        meals = _extract_meals_from_table(day_section)
        days.append({"day": day_num, "meals": meals})

    return days


def _extract_meals_from_table(section: str) -> List[Dict[str, str]]:
    """
    Parse Markdown table rows into meal dicts.
    Expected columns: Meal | Food Items | Portion | Calories (approx)
    """
    meals: List[Dict[str, str]] = []

    # Find table rows (lines starting with |)
    table_rows = re.findall(r"^\|(.+)\|$", section, re.MULTILINE)

    for row in table_rows:
        cells = [c.strip() for c in row.split("|")]
        cells = [c for c in cells if c]

        # Skip header rows and separator rows
        if not cells or all(c.startswith("-") or c.startswith(":") for c in cells):
            continue
        if any(
            keyword in cells[0].lower()
            for keyword in ("meal", "time", "slot", "---")
        ):
            continue

        meal: Dict[str, str] = {"meal_name": cells[0]}
        if len(cells) > 1:
            meal["food_items"] = cells[1]
        if len(cells) > 2:
            meal["portion"] = cells[2]
        if len(cells) > 3:
            meal["calories"] = cells[3]

        meals.append(meal)

    return meals


def _extract_nutritional_summary(text: str) -> Optional[Dict[str, str]]:
    """
    Try to find a "Daily Nutritional Summary" section and extract
    key-value pairs.
    """
    summary_pattern = re.compile(
        r"(?:daily\s+)?nutritional\s+summary",
        re.IGNORECASE,
    )
    match = summary_pattern.search(text)
    if not match:
        return None

    section = text[match.end(): match.end() + 1000]

    summary: Dict[str, str] = {}
    nutrient_pattern = re.compile(
        r"(calories|protein|carbs?|carbohydrates?|fat|fibre|fiber|sodium)"
        r"\s*[:\|]\s*([^\n\|]+)",
        re.IGNORECASE,
    )
    for m in nutrient_pattern.finditer(section):
        key = m.group(1).lower().strip()
        value = m.group(2).strip().rstrip("|")
        summary[key] = value

    return summary if summary else None


def _extract_clinical_notes(text: str) -> Optional[str]:
    """
    Extract the clinical notes section if present.
    """
    notes_pattern = re.compile(
        r"#{2,3}\s*(?:clinical\s+)?notes?\s*\n([\s\S]+?)(?=\n#{2,3}\s|\Z)",
        re.IGNORECASE,
    )
    match = notes_pattern.search(text)
    if match:
        return match.group(1).strip()
    return None
