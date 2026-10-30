
import json
from pathlib import Path
from collections import Counter
from statistics import fmean

from src.evaluation.metrics import compute_metrics


INPUT_FILE = Path(
    "outputs/results/phase4/retrieval_results.jsonl"
)

OUTPUT_DIR = Path("outputs/results/phase4")

METRIC_FILE = OUTPUT_DIR / "metric_results.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "metric_summary.json"

ACTIONS = ["keep", "concat", "rewrite"]

# Metrics specified in our experiment configuration
METRIC_NAMES = [
    "mrr@10",
    "recall@10",
    "recall@100",
    "ndcg@10",
]


def load_records():
    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def evaluate_record(record, action):
    retrieval = record.get("retrieval", {}).get(action, {})
    status = retrieval.get("status", "missing")

    result = {
        "example_id": record["example_id"],
        "conversation_id": record.get("conversation_id"),
        "turn_number": record.get("turn_number"),
        "action": action.upper(),
        "status": status,
        "metrics": None,
    }

    if status != "success":
        return result

    ranked_results = retrieval.get("ranked_results", [])

    retrieved_ids = [
        item["passage_id"]
        for item in ranked_results
    ]

    relevant_ids = set(
        record.get("relevant_passage_ids", [])
    )

    metrics = compute_metrics(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        ks=[10, 100],
    )

    result["metrics"] = {
        name: metrics[name]
        for name in METRIC_NAMES
    }

    return result


def calculate_average(results):
    valid_results = [
        result for result in results
        if result["status"] == "success"
        and result["metrics"] is not None
    ]

    if not valid_results:
        return None

    return {
        metric: fmean(
            result["metrics"][metric]
            for result in valid_results
        )
        for metric in METRIC_NAMES
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records()

    print(f"Loaded {len(records)} turns.")
    print("Calculating retrieval metrics...\n")

    all_results = []
    results_by_action = {
        action: [] for action in ACTIONS
    }

    status_counts = Counter()

    for record in records:
        for action in ACTIONS:
            result = evaluate_record(record, action)

            all_results.append(result)
            results_by_action[action].append(result)

            status_counts[
                (action, result["status"])
            ] += 1

    # Save individual turn-level metrics
    with METRIC_FILE.open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    # Find turns where all three formulations succeeded
    successful_ids = {}

    for action in ACTIONS:
        successful_ids[action] = {
            result["example_id"]
            for result in results_by_action[action]
            if result["status"] == "success"
        }

    common_ids = set.intersection(
        *successful_ids.values()
    )

    common_results = {
        action: [
            result
            for result in results_by_action[action]
            if result["example_id"] in common_ids
        ]
        for action in ACTIONS
    }

    # Calculate averages using all available results
    all_available_summary = {
        action.upper(): {
            "successful_turns": len([
                r for r in results_by_action[action]
                if r["status"] == "success"
            ]),
            "average_metrics": calculate_average(
                results_by_action[action]
            ),
        }
        for action in ACTIONS
    }

    # Calculate averages on identical turns for fair comparison
    common_summary = {
        action.upper(): {
            "successful_turns": len(common_results[action]),
            "average_metrics": calculate_average(
                common_results[action]
            ),
        }
        for action in ACTIONS
    }

    summary = {
        "total_input_turns": len(records),
        "common_successful_turns": len(common_ids),
        "status_counts": {
            f"{action}_{status}": count
            for (action, status), count
            in status_counts.items()
        },
        "all_available_results": all_available_summary,
        "common_turn_comparison": common_summary,
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("METRIC EVALUATION COMPLETE")
    print("-" * 45)

    for action in ACTIONS:
        success_count = successful_ids[action].__len__()

        print(
            f"{action.upper()}: "
            f"{success_count}/{len(records)} successful"
        )

    print(f"\nCommon turns for fair comparison: {len(common_ids)}")

    print("\nAVERAGE METRICS (COMMON TURNS)")
    print("-" * 45)

    for action, values in common_summary.items():
        print(f"\n{action}")

        for metric, value in values["average_metrics"].items():
            print(f"  {metric}: {value:.4f}")

    print("\nFiles saved:")
    print(METRIC_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()

