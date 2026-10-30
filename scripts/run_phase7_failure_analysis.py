"""Phase 7: Failure analysis for selective conversational query rewriting.

Uses:
- All 134 comparable turns for outcome/feature analysis.
- The 24 held-out test turns for gate-selection analysis.

Does not modify previous experiment outputs.
"""

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.gating.features import extract_features


ROOT = Path(__file__).resolve().parents[1]

PILOT_FILE = ROOT / "outputs/results/formulation_pilot.jsonl"
OUTCOME_FILE = ROOT / "outputs/results/phase4/outcome_results.jsonl"
PREDICTION_FILE = ROOT / "outputs/results/phase5/test_predictions.jsonl"

OUTPUT_DIR = ROOT / "outputs/results/phase7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return statistics.mean(values) if values else None


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_outcome(value):
    return str(value).strip().lower()


def query_length_category(length):
    if length <= 5:
        return "short (<=5 words)"
    if length <= 10:
        return "medium (6-10 words)"
    return "long (>10 words)"


def overlap_category(value):
    if value < 0.10:
        return "low (<0.10)"
    if value < 0.30:
        return "medium (0.10-0.29)"
    return "high (>=0.30)"


def summarize_outcomes(records):
    counts = Counter(record["outcome"] for record in records)
    total = len(records)

    return {
        "total_turns": total,
        "counts": dict(counts),
        "percentages": {
            outcome: round(count / total * 100, 2)
            for outcome, count in counts.items()
        },
    }


def build_feature_analysis(pilot_by_id, outcomes):
    """Join outcomes to pilot questions and extract the original gate features."""

    joined = []

    for outcome in outcomes:
        example_id = outcome["example_id"]
        pilot = pilot_by_id.get(example_id)

        if pilot is None:
            continue

        features = extract_features(
            question=pilot.get("question", ""),
            history=pilot.get("history", []),
            turn_number=pilot.get("turn_number", 0),
        )

        joined.append({
            "example_id": example_id,
            "conversation_id": pilot.get("conversation_id"),
            "turn_number": pilot.get("turn_number"),
            "question": pilot.get("question", ""),
            "history": pilot.get("history", []),
            "outcome": normalize_outcome(outcome["outcome"]),
            "keep_score": safe_float(outcome.get("keep_score")),
            "rewrite_score": safe_float(outcome.get("rewrite_score")),
            "delta": safe_float(outcome.get("delta")),
            **features,
        })

    # Feature averages for each outcome class.
    grouped = defaultdict(list)
    for row in joined:
        grouped[row["outcome"]].append(row)

    feature_names = [
        "query_length",
        "has_pronoun",
        "has_ellipsis",
        "turn_position",
        "lexical_overlap",
    ]

    feature_summary = []

    for outcome in ["helpful", "tied", "harmful"]:
        rows = grouped.get(outcome, [])

        result = {
            "outcome": outcome,
            "count": len(rows),
        }

        for feature in feature_names:
            result[f"mean_{feature}"] = round(
                mean([row[feature] for row in rows]), 4
            ) if rows else None

        result["pronoun_rate_percent"] = round(
            mean([row["has_pronoun"] for row in rows]) * 100, 2
        ) if rows else None

        result["ellipsis_rate_percent"] = round(
            mean([row["has_ellipsis"] for row in rows]) * 100, 2
        ) if rows else None

        feature_summary.append(result)

    # Outcome counts for interpretable feature categories.
    category_definitions = {
        "query_length_category": lambda row: query_length_category(
            row["query_length"]
        ),
        "pronoun": lambda row: (
            "contains pronoun" if row["has_pronoun"]
            else "no pronoun"
        ),
        "ellipsis": lambda row: (
            "contains ellipsis phrase" if row["has_ellipsis"]
            else "no ellipsis phrase"
        ),
        "lexical_overlap_category": lambda row: overlap_category(
            row["lexical_overlap"]
        ),
    }

    category_rows = []

    for feature_name, category_function in category_definitions.items():
        category_groups = defaultdict(list)

        for row in joined:
            category_groups[category_function(row)].append(row)

        for category, rows in sorted(category_groups.items()):
            counts = Counter(row["outcome"] for row in rows)
            total = len(rows)

            category_rows.append({
                "feature": feature_name,
                "category": category,
                "count": total,
                "helpful": counts.get("helpful", 0),
                "tied": counts.get("tied", 0),
                "harmful": counts.get("harmful", 0),
                "helpful_percent": round(
                    counts.get("helpful", 0) / total * 100, 2
                ),
                "tied_percent": round(
                    counts.get("tied", 0) / total * 100, 2
                ),
                "harmful_percent": round(
                    counts.get("harmful", 0) / total * 100, 2
                ),
            })

    return joined, feature_summary, category_rows


def confusion_statistics(rows, selection_field):
    """Compare a gate's rewrite selections with helpful/tied/harmful labels."""

    selected = [row for row in rows if row[selection_field]]
    not_selected = [row for row in rows if not row[selection_field]]

    helpful = [row for row in rows if row["outcome"] == "helpful"]
    harmful = [row for row in rows if row["outcome"] == "harmful"]

    selected_helpful = [
        row for row in selected if row["outcome"] == "helpful"
    ]
    selected_harmful = [
        row for row in selected if row["outcome"] == "harmful"
    ]

    return {
        "selected_count": len(selected),
        "not_selected_count": len(not_selected),
        "selected_helpful": len(selected_helpful),
        "selected_tied": sum(
            row["outcome"] == "tied" for row in selected
        ),
        "selected_harmful": len(selected_harmful),
        "missed_helpful": sum(
            row["outcome"] == "helpful" for row in not_selected
        ),
        "helpful_recall_percent": round(
            len(selected_helpful) / len(helpful) * 100, 2
        ) if helpful else None,
        "selected_rewrite_helpful_percent": round(
            len(selected_helpful) / len(selected) * 100, 2
        ) if selected else None,
        "harmful_rewrite_selection_percent": round(
            len(selected_harmful) / len(harmful) * 100, 2
        ) if harmful else None,
    }


def selected_score(rows, selection_field):
    scores = []

    for row in rows:
        field = "rewrite_score" if row[selection_field] else "keep_score"
        score = safe_float(row.get(field))

        if score is not None:
            scores.append(score)

    return mean(scores)


def build_test_gate_analysis(predictions):
    """Evaluate learned and matched-random selections on held-out test turns."""

    by_budget = defaultdict(list)

    for prediction in predictions:
        row = dict(prediction)
        row["outcome"] = normalize_outcome(row["outcome"])
        row["learned_rewrite"] = bool(row["learned_rewrite"])
        row["random_rewrite"] = bool(row["random_rewrite"])
        by_budget[float(row["budget"])].append(row)

    analysis_rows = []
    detailed_rows = []

    for budget in sorted(by_budget):
        rows = by_budget[budget]

        learned_stats = confusion_statistics(rows, "learned_rewrite")
        random_stats = confusion_statistics(rows, "random_rewrite")

        learned_score = selected_score(rows, "learned_rewrite")
        random_score = selected_score(rows, "random_rewrite")

        rewrite_rate = (
            learned_stats["selected_count"] / len(rows) * 100
            if rows else 0
        )

        analysis_rows.append({
            "budget": budget,
            "test_turns": len(rows),
            "learned_rewrite_count": learned_stats["selected_count"],
            "random_rewrite_count": random_stats["selected_count"],
            "learned_actual_rewrite_rate_percent": round(rewrite_rate, 2),
            "learned_selected_helpful": learned_stats["selected_helpful"],
            "learned_selected_tied": learned_stats["selected_tied"],
            "learned_selected_harmful": learned_stats["selected_harmful"],
            "learned_missed_helpful": learned_stats["missed_helpful"],
            "learned_helpful_recall_percent":
                learned_stats["helpful_recall_percent"],
            "learned_selected_rewrite_helpful_percent":
                learned_stats["selected_rewrite_helpful_percent"],
            "learned_harmful_selection_percent":
                learned_stats["harmful_rewrite_selection_percent"],
            "learned_realized_ndcg_at_10": round(learned_score, 4)
                if learned_score is not None else None,
            "random_realized_ndcg_at_10": round(random_score, 4)
                if random_score is not None else None,
            "learned_minus_random_ndcg": round(
                learned_score - random_score, 4
            ) if learned_score is not None and random_score is not None
            else None,
        })

        # Detailed records: useful for inspecting individual test examples.
        for row in rows:
            detailed_rows.append({
                "budget": budget,
                "example_id": row["example_id"],
                "conversation_id": row["conversation_id"],
                "outcome": row["outcome"],
                "helpful_probability": row.get("helpful_probability"),
                "learned_rewrite": row["learned_rewrite"],
                "random_rewrite": row["random_rewrite"],
                "keep_score": row.get("keep_score"),
                "rewrite_score": row.get("rewrite_score"),
                "concat_score": row.get("concat_score"),
                "learned_selected_score": (
                    row["rewrite_score"]
                    if row["learned_rewrite"]
                    else row["keep_score"]
                ),
                "random_selected_score": (
                    row["rewrite_score"]
                    if row["random_rewrite"]
                    else row["keep_score"]
                ),
            })

    return analysis_rows, detailed_rows


def build_examples(joined, test_detail, pilot_by_id):
    """Select representative helpful, tied, harmful and gate-error examples."""

    examples = {
        "largest_rewrite_improvements": [],
        "largest_rewrite_degradations": [],
        "representative_tied_examples": [],
        "test_gate_missed_helpful": [],
        "test_gate_selected_harmful": [],
    }

    def make_example(row):
        pilot = pilot_by_id[row["example_id"]]

        return {
            "example_id": row["example_id"],
            "conversation_id": row.get("conversation_id"),
            "turn_number": row.get("turn_number"),
            "question": pilot.get("question"),
            "history": pilot.get("history", []),
            "keep_formulation": pilot.get("formulations", {}).get("keep"),
            "concat_formulation": pilot.get("formulations", {}).get("concat"),
            "rewrite_formulation": pilot.get("formulations", {}).get("rewrite"),
            "keep_score": row.get("keep_score"),
            "rewrite_score": row.get("rewrite_score"),
            "delta": row.get("delta"),
            "outcome": row.get("outcome"),
        }

    helpful = sorted(
        [row for row in joined if row["outcome"] == "helpful"],
        key=lambda row: row["delta"] if row["delta"] is not None else 0,
        reverse=True,
    )

    harmful = sorted(
        [row for row in joined if row["outcome"] == "harmful"],
        key=lambda row: row["delta"] if row["delta"] is not None else 0,
    )

    tied = [
        row for row in joined if row["outcome"] == "tied"
    ]

    examples["largest_rewrite_improvements"] = [
        make_example(row) for row in helpful[:5]
    ]

    examples["largest_rewrite_degradations"] = [
        make_example(row) for row in harmful[:5]
    ]

    examples["representative_tied_examples"] = [
        make_example(row) for row in tied[:5]
    ]

    # Use the 50% budget as a representative operating point.
    # The other budgets remain available in the detailed CSV.
    for row in test_detail:
        if row["budget"] != 0.5:
            continue

        pilot = pilot_by_id[row["example_id"]]

        if row["outcome"] == "helpful" and not row["learned_rewrite"]:
            examples["test_gate_missed_helpful"].append({
                **row,
                "question": pilot.get("question"),
                "history": pilot.get("history", []),
                "rewrite_formulation":
                    pilot.get("formulations", {}).get("rewrite"),
            })

        if row["outcome"] == "harmful" and row["learned_rewrite"]:
            examples["test_gate_selected_harmful"].append({
                **row,
                "question": pilot.get("question"),
                "history": pilot.get("history", []),
                "rewrite_formulation":
                    pilot.get("formulations", {}).get("rewrite"),
            })

    return examples


def main():
    print("Loading Phase 7 analysis inputs...")

    pilot = read_jsonl(PILOT_FILE)
    outcomes = read_jsonl(OUTCOME_FILE)
    predictions = read_jsonl(PREDICTION_FILE)

    pilot_by_id = {row["turn_id"]: row for row in pilot}

    print(f"Pilot turns: {len(pilot)}")
    print(f"Comparable outcome records: {len(outcomes)}")
    print(f"Test prediction records: {len(predictions)}")

    # 1. Overall outcome distribution.
    outcome_summary = summarize_outcomes(outcomes)

    # 2. Feature/outcome analysis over all comparable turns.
    joined, feature_summary, category_rows = build_feature_analysis(
        pilot_by_id, outcomes
    )

    # 3. Held-out gate analysis.
    test_analysis, test_detail = build_test_gate_analysis(predictions)

    # 4. Representative examples.
    examples = build_examples(joined, test_detail, pilot_by_id)

    # Save machine-readable outputs.
    write_json(
        OUTPUT_DIR / "phase7_outcome_summary.json",
        outcome_summary,
    )

    write_csv(
        OUTPUT_DIR / "feature_outcome_summary.csv",
        feature_summary,
        [
            "outcome",
            "count",
            "mean_query_length",
            "mean_has_pronoun",
            "mean_has_ellipsis",
            "mean_turn_position",
            "mean_lexical_overlap",
            "pronoun_rate_percent",
            "ellipsis_rate_percent",
        ],
    )

    write_csv(
        OUTPUT_DIR / "feature_category_outcomes.csv",
        category_rows,
        [
            "feature",
            "category",
            "count",
            "helpful",
            "tied",
            "harmful",
            "helpful_percent",
            "tied_percent",
            "harmful_percent",
        ],
    )

    write_csv(
        OUTPUT_DIR / "test_gate_analysis.csv",
        test_analysis,
        list(test_analysis[0].keys()) if test_analysis else [],
    )

    write_csv(
        OUTPUT_DIR / "test_gate_details.csv",
        test_detail,
        list(test_detail[0].keys()) if test_detail else [],
    )

    write_json(
        OUTPUT_DIR / "phase7_representative_examples.json",
        examples,
    )

    # Combined summary.
    combined = {
        "outcome_summary": outcome_summary,
        "feature_outcome_summary": feature_summary,
        "test_gate_analysis": test_analysis,
        "notes": [
            "Feature analysis uses all comparable pilot turns.",
            "Gate evaluation uses only held-out test turns.",
            "The learned gate chooses between KEEP and REWRITE.",
            "CONCAT is analysed as a separate retrieval baseline.",
            "Associations between features and outcomes are descriptive, "
            "not causal evidence.",
            "The held-out test set contains only 24 turns, so percentages "
            "and score differences are exploratory.",
        ],
    }

    write_json(
        OUTPUT_DIR / "phase7_failure_analysis.json",
        combined,
    )

    # Print concise results.
    print("\n" + "=" * 70)
    print("PHASE 7 — OUTCOME DISTRIBUTION")
    print("=" * 70)

    for outcome, count in outcome_summary["counts"].items():
        percentage = outcome_summary["percentages"][outcome]
        print(f"{outcome.upper():10} {count:3} ({percentage:5.2f}%)")

    print("\n" + "=" * 70)
    print("FEATURE AVERAGES BY OUTCOME")
    print("=" * 70)

    for row in feature_summary:
        print(f"\n{row['outcome'].upper()} — {row['count']} turns")

        for key, value in row.items():
            if key.startswith("mean_") or key.endswith("_rate_percent"):
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("FEATURE CATEGORY OUTCOMES")
    print("=" * 70)

    for row in category_rows:
        print(
            f"{row['feature']} | {row['category']} | "
            f"n={row['count']} | "
            f"helpful={row['helpful_percent']}% | "
            f"tied={row['tied_percent']}% | "
            f"harmful={row['harmful_percent']}%"
        )

    print("\n" + "=" * 70)
    print("HELD-OUT TEST: LEARNED GATE VS MATCHED RANDOM")
    print("=" * 70)

    for row in test_analysis:
        print(f"\nBudget: {row['budget']:.0%}")
        print(
            f"  Learned rewrites: {row['learned_rewrite_count']}/"
            f"{row['test_turns']} "
            f"({row['learned_actual_rewrite_rate_percent']}%)"
        )
        print(
            f"  Selected helpful/tied/harmful: "
            f"{row['learned_selected_helpful']}/"
            f"{row['learned_selected_tied']}/"
            f"{row['learned_selected_harmful']}"
        )
        print(
            f"  Missed helpful rewrites: "
            f"{row['learned_missed_helpful']}"
        )
        print(
            f"  Helpful recall: "
            f"{row['learned_helpful_recall_percent']}%"
        )
        print(
            f"  Learned realized nDCG@10: "
            f"{row['learned_realized_ndcg_at_10']}"
        )
        print(
            f"  Random realized nDCG@10: "
            f"{row['random_realized_ndcg_at_10']}"
        )
        print(
            f"  Learned - random: "
            f"{row['learned_minus_random_ndcg']}"
        )

    print("\n" + "=" * 70)
    print("REPRESENTATIVE EXAMPLE COUNTS")
    print("=" * 70)

    for name, rows in examples.items():
        print(f"{name}: {len(rows)}")

    print("\nSaved Phase 7 failure-analysis files in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()