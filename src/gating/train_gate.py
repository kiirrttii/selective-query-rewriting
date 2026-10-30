"""Lightweight logistic-regression gate."""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "query_length",
    "has_pronoun",
    "has_ellipsis",
    "turn_position",
    "lexical_overlap",
]


def train_model(X_train, y_train, seed=42):
    """Train a balanced logistic-regression classifier."""
    if len(set(y_train)) < 2:
        raise ValueError(
            "Training split contains only one class. "
            "A binary classifier cannot be trained."
        )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed
        ))
    ])

    model.fit(X_train, y_train)
    return model


def select_threshold(probabilities, keep_scores, rewrite_scores):
    """Choose threshold maximizing validation retrieval quality.

    Ties are broken in favour of the lower rewrite rate.
    """
    probabilities = np.asarray(probabilities)

    candidates = sorted(set(
        [0.0, 1.0000001] + probabilities.tolist()
    ))

    best_threshold = None
    best_score = float("-inf")
    best_rate = float("inf")

    for threshold in candidates:
        decisions = probabilities >= threshold

        scores = [
            rewrite_scores[i] if decisions[i] else keep_scores[i]
            for i in range(len(decisions))
        ]

        mean_score = float(np.mean(scores))
        rewrite_rate = float(np.mean(decisions))

        if (
            mean_score > best_score + 1e-12
            or (
                abs(mean_score - best_score) <= 1e-12
                and rewrite_rate < best_rate
            )
        ):
            best_score = mean_score
            best_rate = rewrite_rate
            best_threshold = float(threshold)

    return best_threshold, best_score, best_rate