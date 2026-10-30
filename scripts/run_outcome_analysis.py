
import json
import yaml
from pathlib import Path

from src.evaluation.outcomes import (
    label_all_outcomes,
    summarise_outcomes,
)

BASE_DIR = Path(".")
METRIC_FILE = BASE_DIR / "outputs/results/phase4/metric_results.jsonl"
CONFIG_FILE = BASE_DIR / "configs/experiment.yaml"

OUTPUT_DIR = BASE_DIR / "outputs/results/phase4"
OUTCOME_FILE = OUTPUT_DIR / "outcome_results.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "outcome_summary.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read primary metric and tie margin from experiment config
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    primary_metric = config["evaluation"]["primary_metric"]
    tie_margin = float(config["evaluation"]["tie_margin"])

    if primary_metric == "UNRESOLVED":
        raise ValueError("Primary metric must be selected before analysis.")

    print(f"Primary metric: {primary_metric}")
    print(f"Tie margin: {tie_margin}")

    # Load metric results
    keep_scores = {}
    rewrite_scores = {}

    with METRIC_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record["status"] != "success":
                continue

            example_id = record["example_id"]
            action = record["action"].lower()
            score = record["metrics"][primary_metric]

            if action == "keep":
                keep_scores[example_id] = score
            elif action == "rewrite":
                rewrite_scores[example_id] = score

    # Identify missing comparisons
    keep_ids = set(keep_scores)
    rewrite_ids = set(rewrite_scores)

    common_ids = keep_ids & rewrite_ids

    missing_rewrite_ids = sorted(keep_ids - rewrite_ids)
    missing_keep_ids = sorted(rewrite_ids - keep_ids)

    # Label outcomes using existing implementation
    outcomes = label_all_outcomes(
        keep_scores=keep_scores,
        rewrite_scores=rewrite_scores,
        tie_margin=tie_margin,
    )

    # Save individual outcomes
    with OUTCOME_FILE.open("w", encoding="utf-8") as f:
        for item in outcomes:
            result = {
                "example_id": item.unique_id,
                "keep_score": item.keep_score,
                "rewrite_score": item.rewrite_score,
                "delta": item.delta,
                "outcome": item.outcome.value,
            }
            f.write(json.dumps(result) + "\n")

    # Calculate outcome percentages
    statistics = summarise_outcomes(outcomes)

    summary = {
        "primary_metric": primary_metric,
        "tie_margin": tie_margin,
        "total_keep_successful": len(keep_ids),
        "total_rewrite_successful": len(rewrite_ids),
        "common_turns": len(common_ids),
        "missing_rewrite_ids": missing_rewrite_ids,
        "missing_keep_ids": missing_keep_ids,
        "outcome_summary": statistics,
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nOUTCOME ANALYSIS COMPLETE")
    print("-" * 40)

    print(f"Comparable turns: {len(common_ids)}")
    print(f"Helpful: {statistics['helpful_pct']:.2f}%")
    print(f"Tied: {statistics['tied_pct']:.2f}%")
    print(f"Harmful: {statistics['harmful_pct']:.2f}%")

    print(f"\nMissing REWRITE results: {len(missing_rewrite_ids)}")
    print("\nSaved files:")
    print(OUTCOME_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()

