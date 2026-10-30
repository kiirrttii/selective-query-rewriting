"""Pre-retrieval feature extraction for query gating."""

import re

PRONOUNS = {
    "he", "she", "it", "they", "them", "his", "her",
    "their", "this", "that", "these", "those", "there",
    "him", "its", "who", "which"
}

ELLIPSIS_PHRASES = {
    "what about", "and what", "how about", "tell me more",
    "what else", "and the", "what happened next"
}


def tokenize(text):
    return set(re.findall(r"\b\w+\b", text.lower()))


def extract_features(question, history, turn_number):
    """Extract inexpensive features available before rewriting."""
    question = question or ""
    history = history or []

    tokens = tokenize(question)
    history_tokens = tokenize(" ".join(history))

    overlap = (
        len(tokens & history_tokens) / len(tokens | history_tokens)
        if tokens | history_tokens else 0.0
    )

    lower_question = question.lower()

    return {
        "query_length": len(question.split()),
        "has_pronoun": int(bool(tokens & PRONOUNS)),
        "has_ellipsis": int(
            any(phrase in lower_question for phrase in ELLIPSIS_PHRASES)
        ),
        "turn_position": int(turn_number),
        "lexical_overlap": overlap,
    }