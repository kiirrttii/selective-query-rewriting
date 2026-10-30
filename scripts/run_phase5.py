
"""Run Phase 5: heuristic, random and budget-constrained learned gates."""

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit

from src.gating.features import extract_features
from src.gating.heuristic import should_rewrite
from src.gating.random_gate import select_random
from src.gating.train_gate import FEATURE_NAMES, train_model


ROOT = Path(__file__).resolve().parents[1]

PILOT_FILE = ROOT / "outputs/results/formulation_pilot.jsonl"
OUTCOME_FILE = ROOT / "outputs/results/phase4/outcome_results.jsonl"
METRIC_FILE = ROOT / "outputs/results/phase4/metric_results.jsonl"
CONFIG_FILE = ROOT / "configs/experiment.yaml"

OUTPUT_DIR = ROOT / "outputs/results/phase5"

REWRITE_BUDGETS = [0.25, 0.50, 0.75, 1.00]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def make_group_splits(groups, seed=42):
    """Create approximately 60/20/20 conversation-level splits."""
    indices = np.arange(len(groups))
    groups = np.asarray(groups)

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed
    )

    train_val_idx, test_idx = next(
        first_split.split(indices, groups=groups)
    )

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=seed
    )

    train_rel, val_rel = next(
        second_split.split(
            train_val_idx,
            groups=groups[train_val_idx]
        )
    )

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    return train_idx, val_idx, test_idx


def evaluate_policy(name, decisions, keep_scores, rewrite_scores):
    """Evaluate a policy using per-query KEEP/REWRITE scores."""
    decisions = np.asarray(decisions, dtype=bool)

    scores = [
        rewrite_scores[i] if decisions[i] else keep_scores[i]
        for i in range(len(decisions))
    ]

    rewrite_count = int(decisions.sum())
    total = len(decisions)

    return {
        "policy": name,
        "mean_primary_metric": float(np.mean(scores)),
        "rewrite_count": rewrite_count,
        "total_queries": total,
        "rewrite_rate": rewrite_count / total if total else 0.0,
        "llm_calls_per_query": rewrite_count / total if total else 0.0,
    }


def select_threshold_with_budget(
    probabilities,
    keep_scores,
    rewrite_scores,
    budget
):
    """
    Select the best validation threshold subject to a maximum
    rewrite budget.

    The validation retrieval score is maximized.
    Ties prefer fewer rewrites.

    Test scores are never used here.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    keep_scores = np.asarray(keep_scores, dtype=float)
    rewrite_scores = np.asarray(rewrite_scores, dtype=float)

    if len(probabilities) == 0:
        raise ValueError("Validation probabilities cannot be empty.")

    if not (0.0 <= budget <= 1.0):
        raise ValueError("Budget must be between 0 and 1.")

    max_rewrites = int(np.floor(budget * len(probabilities)))

    # Each candidate produces a threshold-based decision.
    # The extra threshold above the maximum selects zero queries.
    candidates = sorted(set(probabilities.tolist()))

    candidates.append(
        float(np.nextafter(np.max(probabilities), np.inf))
    )

    best = None

    for threshold in candidates:
        decisions = probabilities >= threshold
        rewrite_count = int(decisions.sum())

        if rewrite_count > max_rewrites:
            continue

        scores = np.where(
            decisions,
            rewrite_scores,
            keep_scores
        )

        mean_score = float(np.mean(scores))
        rewrite_rate = float(np.mean(decisions))

        candidate = {
            "threshold": float(threshold),
            "validation_score": mean_score,
            "validation_rewrite_count": rewrite_count,
            "validation_max_rewrites": max_rewrites,
            "validation_rewrite_rate": rewrite_rate,
        }

        if best is None:
            best = candidate
        elif mean_score > best["validation_score"] + 1e-12:
            best = candidate
        elif (
            abs(mean_score - best["validation_score"]) <= 1e-12
            and rewrite_count < best["validation_rewrite_count"]
        ):
            best = candidate

    if best is None:
        raise RuntimeError("Could not select a valid threshold.")

    return best


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    primary_metric = config["evaluation"]["primary_metric"]
    seed = int(config["gating"]["random_seed"])

    print("PHASE 5: BUDGET-CONSTRAINED GATING EXPERIMENT")
    print("Primary metric:", primary_metric)
    print("Random seed:", seed)
    print("Rewrite budgets:", REWRITE_BUDGETS)

    pilot = read_jsonl(PILOT_FILE)
    outcomes = read_jsonl(OUTCOME_FILE)
    metric_rows = read_jsonl(METRIC_FILE)

    # Build lookup for CONCAT scores.
    concat_scores = {}

    for row in metric_rows:
        if (
            row.get("status") == "success"
            and row["action"].lower() == "concat"
        ):
            concat_scores[row["example_id"]] = (
                row["metrics"][primary_metric]
            )

    # Align pilot features, outcomes and formulation scores.
    pilot_by_id = {row["turn_id"]: row for row in pilot}

    records = []

    for outcome in outcomes:
        uid = outcome["example_id"]

        if uid not in pilot_by_id or uid not in concat_scores:
            continue

        turn = pilot_by_id[uid]

        features = extract_features(
            question=turn["question"],
            history=turn.get("history", []),
            turn_number=turn["turn_number"],
        )

        records.append({
            "example_id": uid,
            "conversation_id": str(turn["conversation_id"]),
            "features": features,
            "label": int(outcome["outcome"] == "helpful"),
            "outcome": outcome["outcome"],
            "keep_score": float(outcome["keep_score"]),
            "rewrite_score": float(outcome["rewrite_score"]),
            "concat_score": float(concat_scores[uid]),
        })

    if len(records) < 20:
        raise RuntimeError(
            f"Only {len(records)} aligned records. "
            "Not enough data for this pilot."
        )

    print("\nAligned records:", len(records))
    print(
        "Conversations:",
        len(set(row["conversation_id"] for row in records))
    )

    # Prepare feature matrix and labels.
    X = np.array([
        [row["features"][name] for name in FEATURE_NAMES]
        for row in records
    ], dtype=float)

    y = np.array([row["label"] for row in records])
    groups = np.array([row["conversation_id"] for row in records])

    train_idx, val_idx, test_idx = make_group_splits(
        groups,
        seed=seed
    )

    # Verify conversation-level separation.
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])

    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)

    print("\nDATA SPLIT")

    for name, idx in [
        ("Train", train_idx),
        ("Validation", val_idx),
        ("Test", test_idx),
    ]:
        print(
            f"{name}: {len(idx)} turns, "
            f"{len(set(groups[idx]))} conversations, "
            f"{int(y[idx].sum())} helpful labels"
        )

    # Train only on training conversations.
    model = train_model(
        X[train_idx],
        y[train_idx],
        seed=seed
    )

    positive_class_index = list(model.classes_).index(1)

    # Validation probabilities.
    val_probabilities = model.predict_proba(
        X[val_idx]
    )[:, positive_class_index]

    val_keep = np.array([
        records[i]["keep_score"] for i in val_idx
    ])

    val_rewrite = np.array([
        records[i]["rewrite_score"] for i in val_idx
    ])

    # Test probabilities. No test scores are used for threshold selection.
    test_probabilities = model.predict_proba(
        X[test_idx]
    )[:, positive_class_index]

    test_records = [records[i] for i in test_idx]

    keep_scores = np.array([
        r["keep_score"] for r in test_records
    ])

    rewrite_scores = np.array([
        r["rewrite_score"] for r in test_records
    ])

    concat_test_scores = np.array([
        r["concat_score"] for r in test_records
    ])

    # Fixed baselines.
    always_keep = np.zeros(len(test_idx), dtype=bool)
    always_rewrite = np.ones(len(test_idx), dtype=bool)

    results = [
        evaluate_policy(
            "KEEP",
            always_keep,
            keep_scores,
            rewrite_scores
        ),
        {
            "policy": "CONCAT",
            "mean_primary_metric": float(np.mean(concat_test_scores)),
            "rewrite_count": 0,
            "total_queries": len(test_idx),
            "rewrite_rate": 0.0,
            "llm_calls_per_query": 0.0,
        },
        evaluate_policy(
            "REWRITE",
            always_rewrite,
            keep_scores,
            rewrite_scores
        ),
    ]

    # Heuristic baseline.
    heuristic_decisions = np.array([
        should_rewrite(record["features"])
        for record in test_records
    ], dtype=bool)

    results.append(
        evaluate_policy(
            "HEURISTIC",
            heuristic_decisions,
            keep_scores,
            rewrite_scores
        )
    )

    # Save decisions and validation details for every budget.
    budget_summaries = []
    prediction_rows = []

    for budget_index, budget in enumerate(REWRITE_BUDGETS):

        selection = select_threshold_with_budget(
            val_probabilities,
            val_keep,
            val_rewrite,
            budget
        )

        threshold = selection["threshold"]

        # Apply the validation-selected threshold to test.
        learned_decisions = test_probabilities >= threshold

        # Match random baseline to the learned test rewrite count.
        random_count = int(learned_decisions.sum())

        random_indices = select_random(
            rewrite_count=random_count,
            total_count=len(test_idx),
            seed=seed + budget_index
        )

        random_decisions = np.array([
            i in random_indices
            for i in range(len(test_idx))
        ], dtype=bool)

        budget_label = int(budget * 100)

        learned_name = f"LEARNED_{budget_label}PCT"
        random_name = f"RANDOM_MATCHED_{budget_label}PCT"

        learned_result = evaluate_policy(
            learned_name,
            learned_decisions,
            keep_scores,
            rewrite_scores
        )

        random_result = evaluate_policy(
            random_name,
            random_decisions,
            keep_scores,
            rewrite_scores
        )

        learned_result.update({
            "target_validation_budget": budget,
            "validation_threshold": threshold,
            "validation_score": selection["validation_score"],
            "validation_rewrite_rate": selection[
                "validation_rewrite_rate"
            ],
            "validation_rewrite_count": selection[
                "validation_rewrite_count"
            ],
            "validation_max_rewrites": selection[
                "validation_max_rewrites"
            ],
        })

        random_result["target_validation_budget"] = budget

        results.extend([learned_result, random_result])

        budget_summaries.append({
            "budget": budget,
            **selection,
            "test_threshold": threshold,
            "test_rewrite_count": random_count,
            "test_rewrite_rate": random_count / len(test_idx),
        })

        for i, record in enumerate(test_records):
            prediction_rows.append({
                "example_id": record["example_id"],
                "conversation_id": record["conversation_id"],
                "outcome": record["outcome"],
                "helpful_probability": float(test_probabilities[i]),
                "budget": budget,
                "threshold": threshold,
                "learned_rewrite": bool(learned_decisions[i]),
                "random_rewrite": bool(random_decisions[i]),
                "keep_score": record["keep_score"],
                "rewrite_score": record["rewrite_score"],
                "concat_score": record["concat_score"],
            })

    print("\nVALIDATION BUDGET SELECTION")
    print("-" * 75)

    for item in budget_summaries:
        print(
            f"Budget={item['budget']:.0%}  "
            f"threshold={item['threshold']:.6f}  "
            f"validation_score={item['validation_score']:.4f}  "
            f"validation_rewrite_rate="
            f"{item['validation_rewrite_rate']:.2%}  "
            f"test_rewrite_rate={item['test_rewrite_rate']:.2%}"
        )

    print("\nTEST RESULTS")
    print("-" * 90)

    for result in results:
        print(
            f"{result['policy']:24s} "
            f"score={result['mean_primary_metric']:.4f}  "
            f"rewrite_rate={result['rewrite_rate']:.2%}  "
            f"LLM_calls={result['rewrite_count']}"
        )

    # Save test predictions.
    with (OUTPUT_DIR / "test_predictions.jsonl").open(
        "w",
        encoding="utf-8"
    ) as file:
        for row in prediction_rows:
            file.write(json.dumps(row) + "\n")

    summary = {
        "primary_metric": primary_metric,
        "seed": seed,
        "feature_names": FEATURE_NAMES,
        "rewrite_budgets": REWRITE_BUDGETS,
        "budget_selections": budget_summaries,
        "split_sizes": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "split_conversations": {
            "train": sorted(train_groups),
            "validation": sorted(val_groups),
            "test": sorted(test_groups),
        },
        "test_results": results,
    }

    save_json(
        OUTPUT_DIR / "phase5_summary.json",
        summary
    )

    print("\nSaved:")
    print(OUTPUT_DIR / "phase5_summary.json")
    print(OUTPUT_DIR / "test_predictions.jsonl")
    print("\nPHASE 5 COMPLETE")


if __name__ == "__main__":
    main()
