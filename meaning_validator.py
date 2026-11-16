import json
import re
import urllib.request
import urllib.error
from collections import Counter
from typing import Any


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

PROTECTED_ENTITY_LABELS = {
    "PER", "PERSON", "ORG", "LOC", "GPE", "FAC",
    "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY",
    "CARDINAL", "ORDINAL",
}

FRENCH_MONTHS = (
    "janvier|février|fevrier|mars|avril|mai|juin|"
    "juillet|août|aout|septembre|octobre|"
    "novembre|décembre|decembre"
)

# Written French dates and numeric dates.
DATE_PATTERN = re.compile(
    rf"(?<!\w)(?:"
    rf"\d{{1,2}}\s+(?:{FRENCH_MONTHS})(?:\s+\d{{4}})?"
    rf"|\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"
    rf")(?!\w)",
    flags=re.IGNORECASE,
)

AMOUNT_PATTERN = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s?(?:%|€|euros?|dollars?)(?!\w)",
    flags=re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)"
)


# "personne" is deliberately excluded.
# It can refer to a person, not necessarily a negation.
NEGATION_PATTERN = re.compile(
    r"(?<!\w)(?:ne|n['’]|pas|jamais|plus|aucun|aucune|"
    r"rien|sans|ni)(?!\w)",
    flags=re.IGNORECASE,
)


# Expressions that should not normally appear in the rewritten content.
META_COMMENTARY_PATTERNS = [
    re.compile(r"\bnote\s*:", re.IGNORECASE),
    re.compile(r"\bthe rewrite above\b", re.IGNORECASE),
    re.compile(r"\bthis rewrite\b", re.IGNORECASE),
    re.compile(r"\bthis example\b", re.IGNORECASE),
    re.compile(r"\badjustments may be needed\b", re.IGNORECASE),
    re.compile(r"\bremarque\s*:", re.IGNORECASE),
    re.compile(
        r"\bvoici la (?:version|réécriture|reformulation)\b",
        re.IGNORECASE,
    ),
]


# Small, curated vocabulary groups for important concepts.
# This is not a general French semantic dictionary.
CONTENT_ANCHORS = {
    "municipal_authority": {
        "municipalité", "mairie", "commune", "ville"
    },
    "target_audience": {
        "personne âgée", "personnes âgées", "senior", "seniors"
    },
    "personalized_support": {
        "personnalisé", "personnalisée",
        "personnalisés", "personnalisées",
        "adapté", "adaptée", "adaptés", "adaptées",
    },
    "support": {
        "accompagnement", "assistance", "aide", "aider"
    },
    "administrative_context": {
        "démarche", "démarches",
        "administratif", "administrative",
        "administratifs", "administratives",
        "procédure", "procédures",
    },
}


def _normalise(value: str) -> str:
    """Normalize whitespace and case."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def _unique(items):
    """Remove duplicates while preserving order."""
    seen = set()
    output = []

    for item in items:
        key = _normalise(item)

        if key and key not in seen:
            seen.add(key)
            output.append(item)

    return output


def _extract_numeric_temporal_values(text: str) -> list[str]:
    """
    Extract dates, amounts and numbers.

    Dates are extracted first to avoid separately extracting
    the numbers inside a date such as '15 mars 2026'.
    """
    matches = []
    occupied_spans = []

    for pattern in (DATE_PATTERN, AMOUNT_PATTERN):
        for match in pattern.finditer(text):
            matches.append(match.group(0).strip())
            occupied_spans.append(match.span())

    for match in NUMBER_PATTERN.finditer(text):
        start, end = match.span()

        overlaps = any(
            start < old_end and end > old_start
            for old_start, old_end in occupied_spans
        )

        if not overlaps:
            matches.append(match.group(0).strip())

    return _unique(matches)


def extract_protected_elements(text: str, nlp) -> dict[str, list[str]]:
    """Extract named entities and numeric/temporal values."""
    doc = nlp(text)

    entities = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in PROTECTED_ENTITY_LABELS
        and ent.text.strip()
    ]

    return {
        "entities": _unique(entities),
        "numbers_dates_amounts": _extract_numeric_temporal_values(text),
    }


def _sentence_list(text: str) -> list[str]:
    """Split text into normalized candidate sentences."""
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)

    return [
        _normalise(chunk.strip(" \t-•"))
        for chunk in chunks
        if chunk.strip(" \t-•")
    ]


def _check_repetition(text: str) -> dict[str, Any]:
    """Detect duplicate sentences and repeated sentence openings."""
    sentences = _sentence_list(text)

    counts = Counter(sentences)

    duplicates = [
        sentence
        for sentence, count in counts.items()
        if count > 1
    ]

    prefixes = []

    for sentence in sentences:
        words = sentence.split()

        if len(words) >= 4:
            prefixes.append(" ".join(words[:4]))

    prefix_counts = Counter(prefixes)

    repeated_prefixes = [
        prefix
        for prefix, count in prefix_counts.items()
        if count >= 3
    ]

    warnings = []

    if duplicates:
        warnings.append(
            "One or more sentences are repeated verbatim."
        )

    if repeated_prefixes:
        warnings.append(
            "Several sentences repeat the same opening phrase."
        )

    return {
        "duplicate_sentences": duplicates,
        "repeated_sentence_openings": repeated_prefixes,
        "warnings": warnings,
    }


def _check_content_anchors(
    original_text: str,
    simplified_text: str,
) -> dict[str, Any]:
    """
    Check whether selected concepts in the original remain
    represented in the simplified text.

    These checks are heuristic and may produce false positives.
    """
    original = _normalise(original_text)
    simplified = _normalise(simplified_text)

    missing = []

    for concept, variants in CONTENT_ANCHORS.items():

        source_contains_concept = any(
            _normalise(variant) in original
            for variant in variants
        )

        output_contains_concept = any(
            _normalise(variant) in simplified
            for variant in variants
        )

        if source_contains_concept and not output_contains_concept:
            missing.append(concept)

    warnings = []

    if missing:
        warnings.append(
            "Potentially missing content concepts: "
            + ", ".join(missing)
        )

    return {
        "potentially_missing_content_anchors": missing,
        "warnings": warnings,
    }


def _check_actor_change(
    original_text: str,
    simplified_text: str,
) -> list[str]:
    """
    Detect one specific actor reversal pattern.

    This is a narrow heuristic, not general role or action extraction.
    """
    original = _normalise(original_text)
    simplified = _normalise(simplified_text)

    original_municipal_decision = re.search(
        r"\b(?:la municipalité|la mairie|la commune|la ville)"
        r"\s+a décidé\b",
        original,
    )

    changed_elderly_actor = re.search(
        r"\b(?:les personnes âgées|les seniors)"
        r"\s+ont décidé\b",
        simplified,
    )

    if original_municipal_decision and changed_elderly_actor:
        return [
            "The decision-maker may have changed from the municipality "
            "to the elderly people."
        ]

    return []


def check_protected_elements(
    original_text: str,
    simplified_text: str,
    nlp,
) -> dict[str, Any]:
    """
    Run deterministic and heuristic checks.

    Any detected issue requires contextual human review.
    """
    original_items = extract_protected_elements(
        original_text, nlp
    )

    simplified_items = extract_protected_elements(
        simplified_text, nlp
    )

    original_values = {
        _normalise(value)
        for value in original_items["numbers_dates_amounts"]
    }

    missing_entities = [
        item
        for item in original_items["entities"]
        if _normalise(item) not in _normalise(simplified_text)
    ]

    missing_values = [
        item
        for item in original_items["numbers_dates_amounts"]
        if _normalise(item) not in _normalise(simplified_text)
    ]

    unsupported_values = [
        item
        for item in simplified_items["numbers_dates_amounts"]
        if _normalise(item) not in original_values
    ]

    original_negations = NEGATION_PATTERN.findall(original_text)
    simplified_negations = NEGATION_PATTERN.findall(simplified_text)

    anchor_check = _check_content_anchors(
        original_text, simplified_text
    )

    repetition_check = _check_repetition(simplified_text)

    actor_warnings = _check_actor_change(
        original_text, simplified_text
    )

    meta_commentary = [
        pattern.pattern
        for pattern in META_COMMENTARY_PATTERNS
        if pattern.search(simplified_text)
    ]

    warnings = []

    if missing_entities:
        warnings.append(
            "Some original named entities are missing from the generated text."
        )

    if missing_values:
        warnings.append(
            "Some original numbers, dates or amounts are missing."
        )

    if unsupported_values:
        warnings.append(
            "The generated text contains numbers, dates or amounts "
            "that were not present in the original."
        )

    if len(simplified_negations) < len(original_negations):
        warnings.append(
            "The generated text contains fewer explicit negation markers."
        )

    warnings.extend(anchor_check["warnings"])
    warnings.extend(repetition_check["warnings"])
    warnings.extend(actor_warnings)

    if meta_commentary:
        warnings.append(
            "The generated text may contain unwanted meta-commentary."
        )

    warnings = _unique(warnings)

    return {
        "original_protected_elements": original_items,
        "simplified_protected_elements": simplified_items,
        "potentially_missing_entities": missing_entities,
        "potentially_missing_values": missing_values,
        "potentially_unsupported_values": unsupported_values,
        "original_negation_marker_count": len(original_negations),
        "simplified_negation_marker_count": len(simplified_negations),
        "content_anchor_check": anchor_check,
        "actor_change_warnings": actor_warnings,
        "repetition_check": repetition_check,
        "potential_meta_commentary_patterns": meta_commentary,
        "warnings": warnings,
        "passed": not warnings,
        "note": (
            "These checks are heuristic. A valid paraphrase may trigger "
            "a warning. Review all detected issues in context."
        ),
    }



def review_semantic_preservation(
    original_text: str,
    simplified_text: str,
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 180,
) -> dict[str, Any]:
    """
    Ask a local Ollama model to review semantic preservation.

    Advisory review only. Not independent human validation.
    """

    prompt = f"""
You are reviewing a French FALC rewrite for meaning preservation.

Compare the ORIGINAL and REWRITTEN texts carefully.

Check whether the rewrite:
- preserves every important fact and detail;
- preserves people, organisations, places, dates and numbers;
- preserves who does what, to whom, when, where and why;
- preserves negation, conditions and obligations;
- preserves the original purpose and intended meaning;
- avoids adding unsupported information.

A synonym is acceptable only if it preserves the meaning.

Pay particular attention to:
- Decisions versus completed actions.
- Future, present and past actions.
- The difference between deciding to do something and
  actually doing it.
- The relationship between an organisation, its actions
  and the people receiving support.
- Personalized or adapted support.

IMPORTANT:
- Report only genuine differences.
- Distinguish semantic changes from stylistic paraphrases.
- Identify the original information and its rewritten equivalent.
- If uncertain, use "uncertain".
- Do not assume that shorter text automatically preserves meaning.

Return ONLY valid JSON with this structure:
{{
  "overall": "preserved | partially_preserved | not_preserved | uncertain",
  "missing_information": [],
  "changed_information": [],
  "unsupported_additions": [],
  "explanation": "A short explanation in English."
}}

ORIGINAL TEXT:
{original_text}

REWRITTEN TEXT:
{simplified_text}
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }

    # Accept either the Ollama base URL or its generate endpoint.
    endpoint = ollama_url.rstrip("/")

    if not endpoint.endswith("/api/generate"):
        endpoint += "/api/generate"

    try:
        data = json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8")

        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=timeout
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        raw_response = result.get("response", "")

        if isinstance(raw_response, str):
            review = json.loads(raw_response)
        else:
            review = raw_response

        if not isinstance(review, dict):
            raise ValueError("Expected a JSON object.")

        required = {"overall", "explanation"}

        if not required.issubset(review):
            raise ValueError(
                f"Missing required fields: "
                f"{required - set(review.keys())}"
            )

        allowed_statuses = {
            "preserved",
            "partially_preserved",
            "not_preserved",
            "uncertain",
        }

        if review["overall"] not in allowed_statuses:
            review["overall"] = "uncertain"

        list_fields = (
            "missing_information",
            "changed_information",
            "unsupported_additions",
        )

        for field in list_fields:
            value = review.get(field, [])

            if not isinstance(value, list):
                value = [value]

            review[field] = [str(item) for item in value]

        review["explanation"] = str(review["explanation"])
        review["reviewer_model"] = model
        review["available"] = True

        review["note"] = (
            "This review was generated by an LLM. It can miss errors "
            "or report false positives; it is not proof of semantic equivalence."
        )

        return review

    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as error:

        return {
            "available": False,
            "overall": "uncertain",
            "missing_information": [],
            "changed_information": [],
            "unsupported_additions": [],
            "explanation": f"Semantic review unavailable: {error}",
            "reviewer_model": model,
            "note": "Only deterministic checks are available.",
        }


def validate_meaning(
    original_text: str,
    simplified_text: str,
    nlp,
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    """
    Run deterministic checks and the advisory LLM review.

    Semantic concerns and deterministic warnings require human review.
    """
    protected_check = check_protected_elements(
        original_text,
        simplified_text,
        nlp,
    )

    semantic_review = review_semantic_preservation(
        original_text,
        simplified_text,
        model,
        ollama_url,
    )

    # A changed-information observation alone does not trigger
    # the warning if the reviewer assesses the meaning as preserved.
    # Missing information and unsupported additions still require review.
    semantic_concerns = (
    semantic_review.get("overall") in {
        "partially_preserved",
        "not_preserved",
        "uncertain",
    }
    or bool(semantic_review.get("missing_information"))
    or bool(semantic_review.get("unsupported_additions"))
)

    needs_review = (
        not protected_check["passed"]
        or semantic_concerns
        or not semantic_review.get("available", False)
    )

    return {
        "status": (
            "needs_human_review"
            if needs_review
            else "no_issue_detected"
        ),
        "protected_elements": protected_check,
        "semantic_review": semantic_review,
        "note": (
            "No automated check can guarantee meaning preservation. "
            "Human validation remains necessary."
        ),
    }
