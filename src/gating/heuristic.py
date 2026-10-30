"""Simple pre-retrieval heuristic baseline."""


def should_rewrite(features, short_query_threshold=5):
    """Rewrite if the question is short or context-dependent."""
    return bool(
        features["has_pronoun"]
        or features["has_ellipsis"]
        or features["query_length"] <= short_query_threshold
    )