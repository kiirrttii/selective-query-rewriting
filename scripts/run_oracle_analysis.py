
import json
import yaml
from pathlib import Path

from src.evaluation.oracle import (
    oracle_score_per_turn,
    mean_oracle_score,
    oracle_headroom_summary,
)

METRIC_FILE = Path("outputs/results/phase4/metric_results.jsonl")
CONFIG_FILE = Path("configs/experiment.yaml")

OUTPUT_DIR = Path("outputs/results/phase4")
ORACLE_FILE = OUTPUT_DIR / "oracle_results.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "oracle_summary.json"

ACTIONS = ["keep", "concat", "rewrite"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    primary_metric = config["evaluation"]["primary_metric"]

    if primary_metric == "UNRESOLVED":
        raise ValueError("Primary metric has not been selected.")

    # Collect successful scores for each formulation
    score_sets = {action: {} for action in ACTIONS}

    with METRIC_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record["status"] != "success":
                continue

            action = record["action"].lower()

            if action not in ACTIONS:
                continue

            score_sets[action][record["example_id"]] = (
                record["metrics"][primary_metric]
            )

    # Use only turns where all three formulations succeeded
    common_ids = set.intersection(
        *(set(scores.keys()) for scores in score_sets.values())
    )

    if not common_ids:
        raise ValueError("No common turns available for oracle analysis.")

    common_scores = {
        action: {
            uid: scores[uid]
            for uid in common_ids
        }
        for action, scores in score_sets.items()
    }

    # Calculate oracle score for every common turn
    per_turn_oracle = oracle_score_per_turn(common_scores)

    # Calculate overall oracle average
    oracle_mean = mean_oracle_score(common_scores)

    # Compare oracle against each formulation
    headroom = {
        action: oracle_headroom_summary(
            common_scores,
            baseline_name=action,
        )
        for action in ACTIONS
    }

    # Identify which formulation the oracle selects per turn
    per_turn_results = []

    for uid in sorted(common_ids):
        scores = {
            action: common_scores[action][uid]
            for action in ACTIONS
        }

        best_score = per_turn_oracle[uid]

        best_actions = [
            action.upper()
            for action, score in scores.items()
            if score == best_score
        ]

        per_turn_results.append({
            "example_id": uid,
            "scores": scores,
            "oracle_score": best_score,
            "oracle_best_actions": best_actions,
        })

    # Save per-turn oracle results
    with ORACLE_FILE.open("w", encoding="utf-8") as f:
        for result in per_turn_results:
            f.write(json.dumps(result) + "\n")

    summary = {
        "primary_metric": primary_metric,
        "total_common_turns": len(common_ids),
        "oracle_mean": oracle_mean,
        "baselines": headroom,
        "oracle_action_counts": {
            action: sum(
                action.upper() in result["oracle_best_actions"]
                for result in per_turn_results
            )
            for action in ACTIONS
        },
        "note": (
            "Oracle is a non-deployable upper bound because it uses "
            "observed per-turn retrieval outcomes."
        ),
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nORACLE ANALYSIS COMPLETE")
    print("-" * 45)
    print(f"Primary metric: {primary_metric}")
    print(f"Common turns: {len(common_ids)}")
    print(f"Oracle mean: {oracle_mean:.4f}")

    print("\nORACLE HEADROOM")
    for action, values in headroom.items():
        print(f"\nCompared with {action.upper()}:")
        print(f"  Baseline: {values['baseline_mean']:.4f}")
        print(f"  Oracle:   {values['oracle_mean']:.4f}")
        print(f"  Headroom: {values['headroom']:.4f}")

    print("\nORACLE ACTION COUNTS")
    for action, count in summary["oracle_action_counts"].items():
        print(f"{action.upper()}: {count}")

    print("\nSaved files:")
    print(ORACLE_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
