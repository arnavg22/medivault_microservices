"""
Response parser -- extracts structured diet plan data from the LLM's
Markdown response into the DietPlan format the frontend expects.

The LLM produces a SINGLE-DAY plan with a 6-column Markdown table:
  | Time | Meal | Foods & Quantities | Approx. Calories | Medical Rationale | Warnings |

This module converts that into:
  {
    "raw_text": "...",
    "parsed": true,
    "patient_name": "...",
    "generated_for_conditions": [...],
    "meals": [
      {
        "time": "6:00 AM",
        "meal_name": "Early Morning",
        "items": [{"food": "...", "quantity": "...", "unit": "..."}],
        "total_calories_approx": 5,
        "medical_rationale": "...",
        "warnings": ["..."]
      }
    ],
    "general_guidelines": ["..."],
    "foods_to_avoid": ["..."],
    "hydration_notes": "...",
    "disclaimer": "..."
  }
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger("utils.response_parser")

_DEFAULT_DISCLAIMER = (
    "This diet plan is AI-generated using your verified hospital records. "
    "It is not a substitute for advice from a registered dietitian (RD) or "
    "your treating physician. Please consult your doctor before making "
    "significant dietary changes."
)


def parse_diet_plan(raw_text: str) -> Dict[str, Any]:
    """
    Parse the LLM's Markdown diet plan into a structured dict matching
    the frontend DietPlan interface.

    Always returns at least ``raw_text`` and ``parsed: False`` so the
    frontend can fall back to rendering raw markdown.
    """
    result: Dict[str, Any] = {
        "raw_text": raw_text,
        "parsed": False,
        "patient_name": None,
        "generated_for_conditions": [],
        "meals": [],
        "general_guidelines": [],
        "foods_to_avoid": [],
        "hydration_notes": "",
        "disclaimer": _DEFAULT_DISCLAIMER,
    }

    try:
        result["patient_name"] = _extract_patient_name(raw_text)
        result["generated_for_conditions"] = _extract_conditions(raw_text)

        meals = _extract_meals(raw_text)
        if meals:
            result["meals"] = meals
            result["parsed"] = True

        result["general_guidelines"] = _extract_general_guidelines(raw_text)
        result["foods_to_avoid"] = _extract_foods_to_avoid(raw_text)
        result["hydration_notes"] = _extract_hydration(raw_text)

        disclaimer = _extract_disclaimer(raw_text)
        if disclaimer:
            result["disclaimer"] = disclaimer

    except Exception as exc:
        logger.warning("diet_plan_parse_error", error=str(exc))

    return result


# ---------------------------------------------------------------------------
# Header extraction
# ---------------------------------------------------------------------------

def _extract_patient_name(text: str) -> Optional[str]:
    """Extract name from ``### Personalised Daily Diet Plan for {name}``."""
    m = re.search(
        r"#{1,4}\s*Personali[sz]ed\s+(?:Daily\s+)?Diet\s+Plan\s+for\s+(.+)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip("*#").strip()
    return None


def _extract_conditions(text: str) -> List[str]:
    """Extract from ``**Designed for:** [conditions]``."""
    m = re.search(
        r"\*\*Designed\s+for:\*\*\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE,
    )
    if not m:
        return []
    raw = m.group(1).strip()
    # Split on comma, semicolon, or " and "
    parts = re.split(r"[;,]|\band\b", raw)
    return [p.strip().strip("*").strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Meal table extraction  (6-column table)
# ---------------------------------------------------------------------------

def _extract_meals(text: str) -> List[Dict[str, Any]]:
    """
    Parse the 6-column Markdown meal table.

    Expected columns:
      Time | Meal | Foods & Quantities | Approx. Calories | Medical Rationale | Warnings
    """
    # Find all table rows (lines starting and ending with |)
    table_rows = re.findall(r"^\|(.+)\|$", text, re.MULTILINE)

    meals: List[Dict[str, Any]] = []

    for row in table_rows:
        cells = [c.strip() for c in row.split("|")]
        cells = [c for c in cells if c != ""]

        # Skip separator rows (---) and header rows
        if not cells:
            continue
        if all(re.match(r"^[-:]+$", c) for c in cells):
            continue
        # Skip header row by checking for known header keywords
        first_lower = cells[0].lower().strip()
        if first_lower in ("time", "meal", "food", "slot", "---"):
            continue

        # We need at least 3 columns to extract meaningful data
        if len(cells) < 3:
            continue

        # Determine layout based on column count
        if len(cells) >= 6:
            # Full 6-column: Time | Meal | Foods | Calories | Rationale | Warnings
            time_str = cells[0]
            meal_name = cells[1]
            foods_text = cells[2]
            calories_text = cells[3]
            rationale = cells[4]
            warnings_text = cells[5]
        elif len(cells) == 5:
            # 5-column: Time | Meal | Foods | Calories | Rationale
            time_str = cells[0]
            meal_name = cells[1]
            foods_text = cells[2]
            calories_text = cells[3]
            rationale = cells[4]
            warnings_text = ""
        elif len(cells) == 4:
            # 4-column: could be Time|Meal|Foods|Calories or Meal|Foods|Portion|Calories
            if re.match(r"\d{1,2}[:.]\d{2}", cells[0]):
                time_str = cells[0]
                meal_name = cells[1]
                foods_text = cells[2]
                calories_text = cells[3]
            else:
                time_str = ""
                meal_name = cells[0]
                foods_text = cells[1]
                calories_text = cells[3]
            rationale = ""
            warnings_text = ""
        elif len(cells) == 3:
            time_str = ""
            meal_name = cells[0]
            foods_text = cells[1]
            calories_text = cells[2]
            rationale = ""
            warnings_text = ""
        else:
            continue

        # Skip if meal_name looks like a header
        if meal_name.lower() in ("meal", "time", "food", "slot"):
            continue

        items = _parse_food_items(foods_text)
        cal = _parse_calories(calories_text)
        warnings = _parse_warnings(warnings_text)

        meals.append({
            "time": time_str.strip(),
            "meal_name": meal_name.strip(),
            "items": items,
            "total_calories_approx": cal,
            "medical_rationale": rationale.strip() if rationale else "",
            "warnings": warnings,
        })

    return meals


def _parse_food_items(text: str) -> List[Dict[str, Any]]:
    """
    Parse a foods-and-quantities cell into a list of
    ``{food, quantity, unit, notes}`` dicts.

    Handles formats like:
      - "Warm jeera water -- 1 glass (250ml)"
      - "Brown rice idli (2), sambar (1 cup)"
      - "Moong dal chilla + green chutney"
      - "Roti (2 medium) with dal (1 katori)"
    """
    if not text or not text.strip():
        return []

    # Split on comma, semicolon, newline, <br>, or "+"
    # but NOT commas inside parentheses
    fragments = re.split(r"(?:<br\s*/?>|\n|;\s*|\+\s*)", text)
    # Further split on commas that aren't inside parentheses
    expanded: List[str] = []
    for frag in fragments:
        parts = _split_outside_parens(frag, ",")
        expanded.extend(parts)

    items: List[Dict[str, Any]] = []
    for frag in expanded:
        frag = frag.strip().strip("•-*").strip()
        if not frag or len(frag) < 2:
            continue

        food, quantity, unit = _parse_single_food(frag)
        items.append({
            "food": food,
            "quantity": quantity,
            "unit": unit if unit else None,
            "notes": None,
        })

    return items if items else [{"food": text.strip(), "quantity": "1", "unit": "serving", "notes": None}]


def _split_outside_parens(text: str, sep: str) -> List[str]:
    """Split text on *sep* but not when inside parentheses."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _parse_single_food(text: str) -> tuple:
    """
    Extract (food_name, quantity, unit) from a single food fragment.

    Returns (food, quantity, unit).
    """
    # Pattern: "Food -- Quantity Unit (detail)"  e.g. "Jeera water -- 1 glass (250ml)"
    m = re.match(
        r"^(.+?)\s*(?:--|—|–)\s*(\d+[\d./]*)\s*(.+)$",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    # Pattern: "Food (quantity unit)" e.g. "Brown rice idli (2 pieces)"
    m = re.match(
        r"^(.+?)\s*\((\d+[\d./]*\s*[^)]*)\)(.*)$",
        text,
    )
    if m:
        food = m.group(1).strip()
        qty_part = m.group(2).strip()
        rest = m.group(3).strip()
        if rest:
            food = food + " " + rest.strip()
        qty_m = re.match(r"^(\d+[\d./]*)\s*(.*)$", qty_part)
        if qty_m:
            return food, qty_m.group(1), qty_m.group(2).strip() or None
        return food, qty_part, None

    # Pattern: "Quantity Unit Food" e.g. "2 medium roti"
    m = re.match(
        r"^(\d+[\d./]*)\s+(small|medium|large|cups?|glasses?|katori|bowls?|pieces?|slices?|tbsp|tsp|tablespoons?|teaspoons?|ml|g|gm|grams?)\s+(.+)$",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(3).strip(), m.group(1), m.group(2).strip()

    # Pattern: "Food Quantity Unit" e.g. "Roti 2 pieces"
    m = re.match(
        r"^(.+?)\s+(\d+[\d./]*)\s*(small|medium|large|cups?|glasses?|katori|bowls?|pieces?|slices?|tbsp|tsp|tablespoons?|teaspoons?|ml|g|gm|grams?|servings?|nos?\.?)$",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(), m.group(2), m.group(3).strip()

    # Fallback: just the food name with quantity "1 serving"
    return text.strip(), "1", "serving"


def _parse_calories(text: str) -> Optional[int]:
    """Extract numeric calorie value from text like '280 kcal', '~300', '5 kcal'."""
    if not text:
        return None
    m = re.search(r"~?\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _parse_warnings(text: str) -> List[str]:
    """Parse the warnings cell into a list of warning strings."""
    if not text or not text.strip():
        return []
    text = text.strip()
    if text.lower() in ("none", "nil", "n/a", "-", "--", "no warnings"):
        return []
    # Split on comma or semicolon
    parts = re.split(r"[;,]", text)
    warnings = []
    for p in parts:
        p = p.strip()
        if p and p.lower() not in ("none", "nil", "n/a"):
            warnings.append(p)
    return warnings


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def _extract_general_guidelines(text: str) -> List[str]:
    """Extract numbered/bulleted items from the General Guidelines section."""
    m = re.search(
        r"#{2,4}\s*General\s+Guidelines?\s*\n([\s\S]+?)(?=\n#{2,4}\s|\Z)",
        text, re.IGNORECASE,
    )
    if not m:
        return []
    section = m.group(1)
    # Match numbered items (1. ...) or bullet items (- ... / * ...)
    items = re.findall(
        r"(?:^\s*\d+[.)]\s*(.+)|^\s*[-*]\s+(.+))",
        section, re.MULTILINE,
    )
    return [(g1 or g2).strip() for g1, g2 in items if (g1 or g2).strip()]


def _extract_foods_to_avoid(text: str) -> List[str]:
    """Extract foods to avoid, from either a table or a bullet list."""
    m = re.search(
        r"#{2,4}\s*Foods?\s+to\s+Avoid\s*\n([\s\S]+?)(?=\n#{2,4}\s|\Z)",
        text, re.IGNORECASE,
    )
    if not m:
        return []
    section = m.group(1)
    results: List[str] = []

    # Try table rows first: | Food | Reason |
    table_rows = re.findall(r"^\|(.+)\|$", section, re.MULTILINE)
    for row in table_rows:
        cells = [c.strip() for c in row.split("|")]
        cells = [c for c in cells if c]
        if not cells or all(re.match(r"^[-:]+$", c) for c in cells):
            continue
        if cells[0].lower() in ("food", "item", "foods"):
            continue
        reason = cells[1].strip() if len(cells) > 1 else ""
        entry = cells[0].strip()
        if reason:
            entry += f" - {reason}"
        results.append(entry)

    if results:
        return results

    # Fallback: bullet/numbered list
    items = re.findall(
        r"(?:^\s*\d+[.)]\s*(.+)|^\s*[-*]\s+(.+))",
        section, re.MULTILINE,
    )
    return [(g1 or g2).strip() for g1, g2 in items if (g1 or g2).strip()]


def _extract_hydration(text: str) -> str:
    """Extract the Hydration Schedule section as a single string."""
    m = re.search(
        r"#{2,4}\s*Hydration\s+(?:Schedule|Notes?|Plan)\s*\n([\s\S]+?)(?=\n#{2,4}\s|\Z)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


def _extract_disclaimer(text: str) -> Optional[str]:
    """Extract the medical disclaimer if present."""
    m = re.search(
        r"\*Medical\s+Disclaimer:\s*(.+?)(?:\*|$)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return None
