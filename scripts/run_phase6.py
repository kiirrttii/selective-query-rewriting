
"""Phase 6: Retrieval quality, LLM usage and efficiency analysis."""

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from src.gating.features import extract_features
from src.gating.heuristic import should_rewrite


ROOT = Path(__file__).resolve().parents[1]

PILOT_FILE = ROOT / "outputs/results/formulation_pilot.jsonl"
METRIC_FILE = ROOT / "outputs/results/phase4/metric_results.jsonl"
PREDICTION_FILE = ROOT / "outputs/results/phase5/test_predictions.jsonl"
CACHE_FILE = ROOT / ".rewrite_cache/rewrites.jsonl"
CONFIG_FILE = ROOT / "configs/experiment.yaml"

OUTPUT_DIR = ROOT / "outputs/results/phase6"

METRICS = ["mrr@10", "recall@10", "recall@100", "ndcg@10"]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_data():
    pilot = read_jsonl(PILOT_FILE)
    metric_rows = read_jsonl(METRIC_FILE)
    predictions = read_jsonl(PREDICTION_FILE)
    cache = read_jsonl(CACHE_FILE)

    pilot_by_id = {row["turn_id"]: row for row in pilot}

    cache_by_id = {
        row["turn_id"]: row
        for row in cache
        if row["turn_id"] in pilot_by_id
    }

    return pilot, pilot_by_id, metric_rows, predictions, cache_by_id


def build_metric_lookup(metric_rows):
    lookup = {}

    for row in metric_rows:
        if row.get("status") != "success":
            continue

        action = row["action"].lower()
        example_id = row["example_id"]

        lookup[(example_id, action)] = row["metrics"]

    return lookup


def validate_cache(pilot, cache_by_id):
    successful_ids = {
        row["turn_id"]
        for row in pilot
        if row.get("rewrite_status") == "success"
    }

    missing = successful_ids - set(cache_by_id)

    if missing:
        raise RuntimeError(
            f"Missing cache records for successful rewrites: {sorted(missing)}"
        )

    return successful_ids


def calculate_cost(example_ids, decisions, cache_by_id):
    """Calculate observed rewrite usage for selected test queries."""

    selected_ids = [
        example_ids[i]
        for i, decision in enumerate(decisions)
        if decision
    ]

    entries = []

    for example_id in selected_ids:
        if example_id not in cache_by_id:
            raise RuntimeError(
                f"No cached usage record for selected rewrite: {example_id}"
            )

        entries.append(cache_by_id[example_id])

    prompt_tokens = sum(
        row["prompt_tokens"] for row in entries
    )

    completion_tokens = sum(
        row["completion_tokens"] for row in entries
    )

    total_tokens = sum(
        row["total_tokens"] for row in entries
    )

    latency_values = [
        row["latency_ms"] for row in entries
    ]

    total_latency = sum(latency_values)

    return {
        "rewrite_count": len(entries),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cumulative_api_latency_ms": round(total_latency, 2),
        "mean_latency_per_call_ms": (
            round(float(np.mean(latency_values)), 2)
            if latency_values else None
        ),
        "median_latency_per_call_ms": (
            round(float(np.median(latency_values)), 2)
            if latency_values else None
        ),
    }


def calculate_quality(example_ids, decisions, metric_lookup):
    """Calculate retrieval quality for a policy."""

    results = {}

    for metric in METRICS:
        scores = []

        for i, example_id in enumerate(example_ids):
            action = "rewrite" if decisions[i] else "keep"

            score = metric_lookup.get(
                (example_id, action), {}
            ).get(metric)

            if score is None:
                raise RuntimeError(
                    f"Missing {metric} for {example_id}, action={action}"
                )

            scores.append(float(score))

        results[metric] = float(np.mean(scores))

    return results


def evaluate_policy(
    name,
    example_ids,
    decisions,
    metric_lookup,
    cache_by_id,
    total_queries,
):
    decisions = np.asarray(decisions, dtype=bool)

    quality = calculate_quality(
        example_ids,
        decisions,
        metric_lookup
    )

    cost = calculate_cost(
        example_ids,
        decisions,
        cache_by_id
    )

    return {
        "policy": name,
        "total_queries": total_queries,
        "rewrite_rate": float(np.mean(decisions)),
        "llm_calls_per_query": (
            cost["rewrite_count"] / total_queries
        ),
        **quality,
        **cost,
    }


def evaluate_concat(example_ids, metric_lookup):
    """CONCAT is a separate zero-LLM-call baseline."""

    scores = {metric: [] for metric in METRICS}

    for example_id in example_ids:
        values = metric_lookup.get((example_id, "concat"))

        if values is None:
            raise RuntimeError(
                f"Missing CONCAT metrics for {example_id}"
            )

        for metric in METRICS:
            scores[metric].append(float(values[metric]))

    return {
        "policy": "CONCAT",
        "total_queries": len(example_ids),
        "rewrite_rate": 0.0,
        "llm_calls_per_query": 0.0,
        **{
            metric: float(np.mean(values))
            for metric, values in scores.items()
        },
        "rewrite_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cumulative_api_latency_ms": 0.0,
        "mean_latency_per_call_ms": None,
        "median_latency_per_call_ms": None,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    primary_metric = config["evaluation"]["primary_metric"]

    print("PHASE 6: COST AND EFFICIENCY ANALYSIS")
    print("Primary metric:", primary_metric)

    pilot, pilot_by_id, metric_rows, predictions, cache_by_id = (
        load_data()
    )

    successful_ids = validate_cache(pilot, cache_by_id)

    print("\nCACHE VALIDATION")
    print("Successful pilot rewrites:", len(successful_ids))
    print("Matching cache records:", len(successful_ids))

    metric_lookup = build_metric_lookup(metric_rows)

    # Use the held-out test IDs from Phase 5 predictions.
    test_ids = sorted({
        row["example_id"] for row in predictions
    })

    if not test_ids:
        raise RuntimeError("No Phase 5 test predictions found.")

    # All test queries must have successful cached rewrites.
    missing_test_rewrites = set(test_ids) - successful_ids

    if missing_test_rewrites:
        raise RuntimeError(
            f"Test rewrites missing from cache: "
            f"{sorted(missing_test_rewrites)}"
        )

    # Ensure every test query has all three retrieval formulations.
    for example_id in test_ids:
        for action in ["keep", "concat", "rewrite"]:
            if (example_id, action) not in metric_lookup:
                raise RuntimeError(
                    f"Missing metric record: {example_id}, {action}"
                )

    print("\nTEST SET")
    print("Unique test queries:", len(test_ids))

    # Fixed baselines.
    no_rewrite = np.zeros(len(test_ids), dtype=bool)
    all_rewrite = np.ones(len(test_ids), dtype=bool)

    results = []

    results.append(
        evaluate_policy(
            "KEEP",
            test_ids,
            no_rewrite,
            metric_lookup,
            cache_by_id,
            len(test_ids),
        )
    )

    results.append(
        evaluate_concat(test_ids, metric_lookup)
    )

    results.append(
        evaluate_policy(
            "REWRITE",
            test_ids,
            all_rewrite,
            metric_lookup,
            cache_by_id,
            len(test_ids),
        )
    )

    # Reconstruct heuristic decisions from original query features.
    heuristic_decisions = []

    for example_id in test_ids:
        turn = pilot_by_id[example_id]

        features = extract_features(
            question=turn["question"],
            history=turn.get("history", []),
            turn_number=turn["turn_number"],
        )

        heuristic_decisions.append(
            should_rewrite(features)
        )

    results.append(
        evaluate_policy(
            "HEURISTIC",
            test_ids,
            heuristic_decisions,
            metric_lookup,
            cache_by_id,
            len(test_ids),
        )
    )

    # Evaluate every learned and matched-random budget.
    prediction_lookup = {}

    for row in predictions:
        key = (row["example_id"], float(row["budget"]))
        prediction_lookup[key] = row

    budgets = sorted({
        float(row["budget"]) for row in predictions
    })

    prediction_details = []

    for budget in budgets:
        learned = []
        random = []

        for example_id in test_ids:
            row = prediction_lookup.get((example_id, budget))

            if row is None:
                raise RuntimeError(
                    f"Missing prediction for {example_id}, "
                    f"budget={budget}"
                )

            learned.append(bool(row["learned_rewrite"]))
            random.append(bool(row["random_rewrite"]))

            prediction_details.append({
                "example_id": example_id,
                "budget": budget,
                "learned_rewrite": bool(row["learned_rewrite"]),
                "random_rewrite": bool(row["random_rewrite"]),
            })

        label = int(budget * 100)

        results.append(
            evaluate_policy(
                f"LEARNED_{label}PCT",
                test_ids,
                learned,
                metric_lookup,
                cache_by_id,
                len(test_ids),
            )
        )

        results.append(
            evaluate_policy(
                f"RANDOM_MATCHED_{label}PCT",
                test_ids,
                random,
                metric_lookup,
                cache_by_id,
                len(test_ids),
            )
        )

    # Calculate savings relative to always-REWRITE.
    rewrite_baseline = next(
        row for row in results
        if row["policy"] == "REWRITE"
    )

    baseline_calls = rewrite_baseline["rewrite_count"]
    baseline_tokens = rewrite_baseline["total_tokens"]
    baseline_latency = rewrite_baseline["cumulative_api_latency_ms"]

    for row in results:
        row["llm_call_reduction_pct"] = (
            100 * (
                1 - row["rewrite_count"] / baseline_calls
            )
            if baseline_calls else 0.0
        )

        row["token_reduction_pct"] = (
            100 * (
                1 - row["total_tokens"] / baseline_tokens
            )
            if baseline_tokens else 0.0
        )

        row["latency_reduction_pct"] = (
            100 * (
                1 - row["cumulative_api_latency_ms"] / baseline_latency
            )
            if baseline_latency else 0.0
        )

    # Overall successful pilot rewrite usage, excluding manual entries.
    pilot_cache_entries = [
        cache_by_id[example_id]
        for example_id in sorted(successful_ids)
    ]

    pilot_usage = calculate_cost(
        sorted(successful_ids),
        [True] * len(successful_ids),
        cache_by_id
    )

    pilot_usage["successful_pilot_rewrites"] = len(successful_ids)
    pilot_usage["extra_manual_cache_entries_excluded"] = 1

    # Save detailed outputs.
    save_json(
        OUTPUT_DIR / "phase6_summary.json",
        {
            "primary_metric": primary_metric,
            "test_query_count": len(test_ids),
            "pilot_successful_rewrite_count": len(successful_ids),
            "test_ids": test_ids,
            "policies": results,
            "pilot_rewrite_usage": pilot_usage,
            "measurement_notes": [
                "Token usage and latency come from actual cached Groq API responses.",
                "The manual-pilot_2 cache entry is excluded.",
                "Policy costs are retrospective estimates using observed per-query rewrite usage.",
                "Cumulative API latency is the sum of individual request latencies, not parallel deployment wall-clock time.",
                "CONCAT is a retrieval baseline and does not invoke the rewriting LLM.",
                "No monetary API cost is calculated because pricing was not supplied."
            ],
        }
    )

    save_json(
        OUTPUT_DIR / "pilot_rewrite_usage.json",
        pilot_usage
    )

    with (OUTPUT_DIR / "policy_cost_quality.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    with (OUTPUT_DIR / "policy_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for row in prediction_details:
            file.write(json.dumps(row) + "\n")

    # Print concise results.
    print("\nPILOT REWRITE USAGE")
    print("Prompt tokens:", pilot_usage["prompt_tokens"])
    print("Completion tokens:", pilot_usage["completion_tokens"])
    print("Total tokens:", pilot_usage["total_tokens"])
    print(
        "Cumulative API latency (ms):",
        pilot_usage["cumulative_api_latency_ms"]
    )
    print(
        "Mean latency per call (ms):",
        pilot_usage["mean_latency_per_call_ms"]
    )

    print("\nTEST POLICY COMPARISON")
    print("-" * 110)

    for row in results:
        print(
            f"{row['policy']:24s} "
            f"nDCG@10={row['ndcg@10']:.4f}  "
            f"calls={row['rewrite_count']:2d}  "
            f"tokens={row['total_tokens']:6d}  "
            f"latency_ms={row['cumulative_api_latency_ms']:10.2f}  "
            f"calls_saved={row['llm_call_reduction_pct']:6.2f}%"
        )

    print("\nSaved:")
    print(OUTPUT_DIR / "phase6_summary.json")
    print(OUTPUT_DIR / "pilot_rewrite_usage.json")
    print(OUTPUT_DIR / "policy_cost_quality.csv")
    print(OUTPUT_DIR / "policy_predictions.jsonl")

    print("\nPHASE 6 COMPLETE")


if __name__ == "__main__":
    main()
