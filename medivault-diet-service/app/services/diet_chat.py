"""
Core chat processing logic for both initial plan generation
and follow-up refinement messages.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import structlog

from app.config.settings import get_settings
from app.models.diet_message import DietMessage
from app.schemas.diet import (
    ChatResponse,
    CreateSessionRequest,
    DietPlan,
    SessionResponse,
)
from app.schemas.patient import PatientContext
from app.services import diet_session as session_service
from app.services.llm.base import AllProvidersExhaustedException
from app.services.llm.router import LLMRouter
from app.services.patient_context import fetch_patient_context
from app.utils.prompt_builder import build_system_prompt
from app.utils.response_parser import parse_diet_plan

logger = structlog.get_logger("diet_chat")

# ---------------------------------------------------------------------------
# Cuisine switch detection
# ---------------------------------------------------------------------------

_CUISINE_SWITCH_PATTERNS = [
    re.compile(
        r"switch\s+(?:to\s+)?(?:a\s+)?(north[\s_-]?indian|south[\s_-]?indian|"
        r"gujarati|maharashtrian|bengali|punjabi|kerala)\s*(?:cuisine|diet|food|style)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:change|update|make)\s+(?:my\s+)?(?:cuisine|diet|food)\s+(?:to\s+)?"
        r"(north[\s_-]?indian|south[\s_-]?indian|gujarati|maharashtrian|bengali|punjabi|kerala)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:i\s+want|give\s+me|i\s+prefer)\s+(?:a\s+)?"
        r"(north[\s_-]?indian|south[\s_-]?indian|gujarati|maharashtrian|bengali|punjabi|kerala)"
        r"\s*(?:cuisine|diet|food|style)?",
        re.IGNORECASE,
    ),
]

_REGIONAL_KEY_MAP = {
    "north indian": "north_indian",
    "northindian": "north_indian",
    "north-indian": "north_indian",
    "south indian": "south_indian",
    "southindian": "south_indian",
    "south-indian": "south_indian",
    "gujarati": "gujarati",
    "maharashtrian": "maharashtrian",
    "bengali": "bengali",
    "punjabi": "punjabi",
    "kerala": "kerala",
}


def _detect_cuisine_switch(text: str) -> Optional[str]:
    """Return a normalised regional key if the user is requesting a switch."""
    for pattern in _CUISINE_SWITCH_PATTERNS:
        m = pattern.search(text)
        if m:
            raw = m.group(1).lower().strip()
            return _REGIONAL_KEY_MAP.get(raw)
    return None


# Module-level LLM router instance, set at startup
_llm_router: Optional[LLMRouter] = None


def set_llm_router(router: LLMRouter) -> None:
    global _llm_router
    _llm_router = router


def get_llm_router() -> Optional[LLMRouter]:
    return _llm_router


async def create_session_and_generate_plan(
    patient_id: str,
    jwt_token: str,
    request: CreateSessionRequest,
) -> SessionResponse:
    """
    Create a new diet session, fetch patient data, generate initial diet plan.

    Steps:
    1. Call patient_context.fetch_patient_context() to get medical data
    2. Create a DietSession document in MongoDB
    3. Build the system prompt using prompt_builder with patient context
    4. Build the initial user message
    5. Call LLMRouter.chat() with messages=[system_prompt, initial_user_msg]
    6. Store the system message and initial user message as DietMessage docs
    7. Store the LLM response as a DietMessage doc with is_diet_plan=True
    8. Parse the response into a DietPlan object (best-effort)
    9. Update DietSession with current_diet_plan and current_provider
    10. Return SessionResponse with the full diet plan
    """
    settings = get_settings()
    router = get_llm_router()
    if router is None:
        raise RuntimeError("LLM router not initialized")

    # 1. Fetch patient context from MediVault backend
    context: PatientContext = await fetch_patient_context(patient_id, jwt_token)

    # 2. Build system prompt
    dietary_prefs = request.dietary_preferences or []
    prefs_str = ", ".join(dietary_prefs) if dietary_prefs else ""
    cuisine_pref = getattr(request, "cuisine_preference", "indian")
    regional_pref = getattr(request, "regional_preference", None)
    system_prompt = build_system_prompt(
        context, prefs_str, regional_preference=regional_pref
    )

    # 3. Determine initial provider
    available = await router.get_available_providers()
    initial_provider = available[0] if available else "unknown"

    # 4. Create session in MongoDB
    vector_chunks = context.vector_context_chunks or []
    session = await session_service.create_session(
        patient_id=patient_id,
        patient_jwt=jwt_token,
        patient_context=context.model_dump(mode="json"),
        provider=initial_provider,
        cuisine_preference=cuisine_pref,
        regional_preference=regional_pref,
    )
    session.vector_chunks_count = len(vector_chunks)
    session.vector_context_used = len(vector_chunks) > 0
    await session.save()
    session_id = str(session.id)

    # 5. Build initial messages
    pref_text = ""
    if dietary_prefs:
        pref_text = f" My dietary preferences: {', '.join(dietary_prefs)}."
    if (
        request.meal_count_preference
        and request.meal_count_preference != 5
    ):
        pref_text += (
            f" I prefer {request.meal_count_preference} meals per day."
        )
    if (
        request.language_preference
        and request.language_preference.lower() != "english"
    ):
        pref_text += f" Please respond in {request.language_preference}."

    initial_user_msg = (
        "Please generate a personalised diet plan for me based on my "
        f"medical records.{pref_text}"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_msg},
    ]

    # 6. Store system prompt and user message
    system_msg_doc = DietMessage(
        session_id=session_id,
        role="system",
        content=system_prompt,
    )
    await system_msg_doc.insert()

    user_msg_doc = DietMessage(
        session_id=session_id,
        role="user",
        content=initial_user_msg,
    )
    await user_msg_doc.insert()

    # 7. Call LLM router
    try:
        response_text, provider_used, switched, prev_provider = (
            await router.chat(
                session_id=session_id,
                messages=messages,
                session_exhausted_providers=list(session.exhausted_providers),
                preferred_provider=initial_provider,
                temperature=0.3,
                max_tokens=4000,
            )
        )
    except AllProvidersExhaustedException:
        session.status = "error"
        await session.save()
        raise

    # 8. Store assistant response
    assistant_msg_doc = DietMessage(
        session_id=session_id,
        role="assistant",
        content=response_text,
        provider_used=provider_used,
        model_used=settings.provider_models.get(provider_used, ""),
        is_diet_plan=True,
    )
    await assistant_msg_doc.insert()

    # 9. Parse diet plan (best-effort — returns a dict)
    plan_dict = parse_diet_plan(response_text)

    # 10. Update session
    await session_service.update_session_plan(
        session, plan_dict, provider_used
    )
    await session_service.increment_message_count(
        session, 2
    )  # user + assistant

    if switched and prev_provider:
        await session_service.update_session_provider(
            session, provider_used, prev_provider
        )

    logger.info(
        "diet_plan_generated",
        session_id=session_id,
        patient_id=patient_id,
        provider=provider_used,
        switched=switched,
    )

    # Build DietPlan from parsed dict if available
    diet_plan_obj = plan_dict if plan_dict else None

    return SessionResponse(
        session_id=session_id,
        status=session.status,
        diet_plan=diet_plan_obj,
        current_provider=provider_used,
        message_count=session.message_count,
        created_at=session.created_at,
    )


async def process_chat_message(
    session_id: str,
    patient_id: str,
    message: str,
) -> ChatResponse:
    """
    Process a follow-up chat message within an existing session.

    Steps:
    1. Fetch DietSession from MongoDB — verify it belongs to this patient_id
    2. Check session status (must be "active") and not expired
    3. Check message_count against SESSION_MAX_MESSAGES limit
    4. Fetch ALL DietMessages for this session ordered by created_at
    5. Convert to OpenAI-style message list
    6. Append the new user message
    7. Store the new user message as DietMessage
    8. Call LLMRouter.chat() with full messages list
    9. Handle provider switch if it occurred
    10. Store LLM response as DietMessage
    11. Increment session.message_count
    12. Parse response for updated diet plan if present
    13. Return ChatResponse with switch metadata
    """
    settings = get_settings()
    router = get_llm_router()
    if router is None:
        raise RuntimeError("LLM router not initialized")

    # 1. Fetch and validate session
    session = await session_service.get_session_for_patient(
        session_id, patient_id
    )
    if session is None:
        raise ValueError("Session not found or access denied")

    # 2. Check session status
    if session.is_expired():
        await session_service.expire_session(session)
        raise ValueError("Session has expired")

    if session.status != "active":
        raise ValueError(f"Session is not active (status: {session.status})")

    # 3. Check message limit
    if session.message_count >= settings.session_max_messages:
        raise ValueError(
            f"Session has reached the maximum of "
            f"{settings.session_max_messages} messages"
        )

    # 3b. Detect mid-session cuisine switch
    new_regional = _detect_cuisine_switch(message)
    if new_regional:
        session.regional_preference = new_regional
        session.cuisine_preference = "indian"
        session.touch()
        await session.save()
        logger.info(
            "cuisine_switch_detected",
            session_id=session_id,
            new_regional=new_regional,
        )

    # 4. Fetch ALL messages for this session
    all_messages = (
        await DietMessage.find(DietMessage.session_id == session_id)
        .sort("+created_at")
        .to_list()
    )

    # 5. Convert to OpenAI-style message list
    #    If a cuisine switch was detected, rebuild the system prompt
    messages: List[Dict[str, str]] = []
    if new_regional and all_messages and all_messages[0].role == "system":
        # Rebuild system prompt with new region
        from app.schemas.patient import PatientContext as _PC

        _ctx = _PC.model_validate(session.patient_context)
        _prefs = ", ".join(
            session.patient_context.get("dietary_restrictions", [])
        )
        rebuilt_prompt = build_system_prompt(
            _ctx, _prefs, regional_preference=new_regional
        )
        messages.append({"role": "system", "content": rebuilt_prompt})
        # Update stored system message
        all_messages[0].content = rebuilt_prompt
        await all_messages[0].save()
        # Skip the original system message
        for msg in all_messages[1:]:
            messages.append({"role": msg.role, "content": msg.content})
    else:
        for msg in all_messages:
            messages.append({"role": msg.role, "content": msg.content})

    # 6. Append the new user message
    messages.append({"role": "user", "content": message})

    # 7. Store new user message
    user_msg_doc = DietMessage(
        session_id=session_id,
        role="user",
        content=message,
    )
    await user_msg_doc.insert()

    # 8. Call LLM router with full history
    try:
        response_text, provider_used, switched, prev_provider = (
            await router.chat(
                session_id=session_id,
                messages=messages,
                session_exhausted_providers=list(
                    session.exhausted_providers
                ),
                preferred_provider=session.current_provider,
                temperature=0.3,
                max_tokens=4000,
            )
        )
    except AllProvidersExhaustedException:
        session.status = "error"
        await session.save()
        raise

    # 9. Handle provider switch
    if switched and prev_provider:
        await session_service.update_session_provider(
            session, provider_used, prev_provider
        )

    # 10. Store assistant response
    is_plan = _response_contains_plan(response_text)
    assistant_msg_doc = DietMessage(
        session_id=session_id,
        role="assistant",
        content=response_text,
        provider_used=provider_used,
        model_used=settings.provider_models.get(provider_used, ""),
        is_diet_plan=is_plan,
    )
    await assistant_msg_doc.insert()

    # 11. Increment message count (user + assistant)
    await session_service.increment_message_count(session, 2)

    # 12. Parse updated diet plan if present
    if is_plan:
        plan_dict = parse_diet_plan(response_text)
        await session_service.update_session_plan(
            session, plan_dict, provider_used
        )

    logger.info(
        "chat_message_processed",
        session_id=session_id,
        patient_id=patient_id,
        provider=provider_used,
        switched=switched,
        is_plan=is_plan,
    )

    return ChatResponse(
        session_id=session_id,
        message_id=str(assistant_msg_doc.id),
        role="assistant",
        content=response_text,
        diet_plan=None,
        provider_used=provider_used,
        provider_switched=switched,
        previous_provider=prev_provider if switched else None,
    )


def _response_contains_plan(text: str) -> bool:
    """Quick heuristic check if the response contains a meal table."""
    indicators = [
        "breakfast",
        "lunch",
        "dinner",
        "| time",
        "| meal",
        "diet plan",
    ]
    text_lower = text.lower()
    return sum(1 for i in indicators if i in text_lower) >= 3
