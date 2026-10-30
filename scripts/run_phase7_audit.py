
"""Phase 7: integrity audit of Phases 4, 5 and 6."""

import json
from pathlib import Path

import numpy as np

from src.gating.features import extract_features
from src.gating.heuristic import should_rewrite


ROOT = Path(__file__).resolve().parents[1]

PILOT = ROOT / "outputs/results/formulation_pilot.jsonl"
RETRIEVAL = ROOT / "outputs/results/phase4/retrieval_results.jsonl"
METRICS = ROOT / "outputs/results/phase4/metric_results.jsonl"
OUTCOMES = ROOT / "outputs/results/phase4/outcome_results.jsonl"
ORACLE = ROOT / "outputs/results/phase4/oracle_results.jsonl"

CACHE = ROOT / ".rewrite_cache/rewrites.jsonl"

PHASE5_SUMMARY = ROOT / "outputs/results/phase5/phase5_summary.json"
PHASE5_PREDICTIONS = ROOT / "outputs/results/phase5/test_predictions.jsonl"

PHASE6_SUMMARY = ROOT / "outputs/results/phase6/phase6_summary.json"

OUTPUT_DIR = ROOT / "outputs/results/phase7"

METRIC_NAMES = [
    "mrr@10",
    "recall@10",
    "recall@100",
    "ndcg@10",
]

BUDGETS = [0.25, 0.50, 0.75, 1.00]


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def close(a, b, tolerance=1e-9):
    return abs(float(a) - float(b)) <= tolerance


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checks = []
    errors = []

    def record_check(name, passed, details=None):
        item = {
            "check": name,
            "passed": bool(passed),
            "details": details,
        }
        checks.append(item)

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")

        if details:
            print("       ", details)

        if not passed:
            errors.append(name)

    required_files = [
        PILOT, RETRIEVAL, METRICS, OUTCOMES, ORACLE,
        CACHE, PHASE5_SUMMARY, PHASE5_PREDICTIONS,
        PHASE6_SUMMARY,
    ]

    missing_files = [
        str(path.relative_to(ROOT))
        for path in required_files
        if not path.exists()
    ]

    record_check(
        "All required experiment files exist",
        not missing_files,
        missing_files if missing_files else "All files found",
    )

    if missing_files:
        raise FileNotFoundError(
            f"Missing experiment files: {missing_files}"
        )

    pilot = read_jsonl(PILOT)
    retrieval = read_jsonl(RETRIEVAL)
    metric_rows = read_jsonl(METRICS)
    outcomes = read_jsonl(OUTCOMES)
    oracle = read_jsonl(ORACLE)
    cache = read_jsonl(CACHE)

    phase5 = read_json(PHASE5_SUMMARY)
    predictions = read_jsonl(PHASE5_PREDICTIONS)
    phase6 = read_json(PHASE6_SUMMARY)

    # ---------------------------------------------------------
    # 1. Dataset and Phase 4 structural checks
    # ---------------------------------------------------------

    pilot_ids = [row["turn_id"] for row in pilot]
    pilot_id_set = set(pilot_ids)

    record_check(
        "Pilot turn IDs are unique",
        len(pilot_ids) == len(pilot_id_set),
        f"{len(pilot)} rows, {len(pilot_id_set)} unique IDs",
    )

    record_check(
        "Pilot contains 136 turns",
        len(pilot) == 136,
        f"Found {len(pilot)}",
    )

    retrieval_ids = [row["example_id"] for row in retrieval]

    record_check(
        "Retrieval file has one record per pilot turn",
        len(retrieval) == len(pilot)
        and set(retrieval_ids) == pilot_id_set,
        f"{len(retrieval)} retrieval records",
    )

    retrieval_statuses = {}

    for action in ["keep", "concat", "rewrite"]:
        statuses = [
            row["retrieval"][action]["status"]
            for row in retrieval
        ]

        retrieval_statuses[action] = {
            status: statuses.count(status)
            for status in sorted(set(statuses))
        }

    total_successful_retrievals = sum(
        counts.get("success", 0)
        for counts in retrieval_statuses.values()
    )

    record_check(
        "Phase 4 has 406 successful retrievals",
        total_successful_retrievals == 406,
        f"Found {total_successful_retrievals}; "
        f"statuses={retrieval_statuses}",
    )

    record_check(
        "KEEP and CONCAT succeeded for all 136 turns",
        retrieval_statuses["keep"].get("success", 0) == 136
        and retrieval_statuses["concat"].get("success", 0) == 136,
        "Expected 136 successful retrievals for each action",
    )

    record_check(
        "REWRITE has 134 successful retrievals",
        retrieval_statuses["rewrite"].get("success", 0) == 134,
        str(retrieval_statuses["rewrite"]),
    )

    metric_ids = {
        row["example_id"] for row in metric_rows
    }

    record_check(
        "Phase 4 has 408 metric records",
        len(metric_rows) == 408,
        f"Found {len(metric_rows)}",
    )

    record_check(
        "Metric records cover all pilot turns",
        metric_ids == pilot_id_set,
        f"{len(metric_ids)} distinct example IDs",
    )

    successful_metric_rows = [
        row for row in metric_rows
        if row.get("status") == "success"
    ]

    record_check(
        "Metric file has 406 successful records",
        len(successful_metric_rows) == 406,
        f"Found {len(successful_metric_rows)}",
    )

    metric_lookup = {}

    for row in successful_metric_rows:
        key = (
            row["example_id"],
            row["action"].lower()
        )
        metric_lookup[key] = row["metrics"]

    missing_metric_values = []

    for row in successful_metric_rows:
        for metric in METRIC_NAMES:
            value = row["metrics"].get(metric)

            if value is None or not np.isfinite(float(value)):
                missing_metric_values.append(
                    (row["example_id"], row["action"], metric)
                )

    record_check(
        "Successful retrieval metrics contain finite values",
        not missing_metric_values,
        f"{len(missing_metric_values)} invalid metric values",
    )

    out_ids = {row["example_id"] for row in outcomes}
    oracle_ids = {row["example_id"] for row in oracle}

    record_check(
        "Outcome records contain 134 unique turns",
        len(outcomes) == 134 and len(out_ids) == 134,
        f"Found {len(outcomes)} records and {len(out_ids)} unique IDs",
    )

    record_check(
        "Oracle records contain 134 turns",
        len(oracle) == 134 and len(oracle_ids) == 134,
        f"Found {len(oracle)} records",
    )

    record_check(
        "Outcome and oracle IDs match",
        out_ids == oracle_ids,
        "Comparing outcome and oracle example IDs",
    )

    record_check(
        "Outcome IDs have KEEP, CONCAT and REWRITE metrics",
        all(
            all(
                (uid, action) in metric_lookup
                for action in ["keep", "concat", "rewrite"]
            )
            for uid in out_ids
        ),
        "All outcome IDs checked against successful metrics",
    )

    # ---------------------------------------------------------
    # 2. Cache validation
    # ---------------------------------------------------------

    successful_pilot_ids = {
        row["turn_id"]
        for row in pilot
        if row.get("rewrite_status") == "success"
    }

    cache_by_id = {
        row["turn_id"]: row
        for row in cache
        if row["turn_id"] in pilot_id_set
    }

    cache_ids = [row["turn_id"] for row in cache]

    record_check(
        "Cache has 135 entries",
        len(cache) == 135,
        f"Found {len(cache)}",
    )

    record_check(
        "Cache turn IDs are unique",
        len(cache_ids) == len(set(cache_ids)),
        f"{len(set(cache_ids))} unique IDs",
    )

    record_check(
        "All 134 successful rewrites have cache records",
        successful_pilot_ids == set(cache_by_id),
        f"Missing: {sorted(successful_pilot_ids - set(cache_by_id))}",
    )

    extra_cache_ids = set(cache_ids) - pilot_id_set

    record_check(
        "Non-pilot cache entries are excluded",
        extra_cache_ids == {"manual-pilot_2"},
        f"Extra IDs: {sorted(extra_cache_ids)}",
    )

    cache_fields = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_ms",
    ]

    invalid_cache_records = []

    for uid in successful_pilot_ids:
        entry = cache_by_id.get(uid, {})

        for field in cache_fields:
            value = entry.get(field)

            if value is None or not np.isfinite(float(value)):
                invalid_cache_records.append((uid, field))

    record_check(
        "All pilot cache entries have valid usage measurements",
        not invalid_cache_records,
        f"{len(invalid_cache_records)} invalid fields",
    )

    # ---------------------------------------------------------
    # 3. Phase 5 split and prediction checks
    # ---------------------------------------------------------

    split = phase5["split_conversations"]

    train_groups = set(split["train"])
    val_groups = set(split["validation"])
    test_groups = set(split["test"])

    no_leakage = (
        not (train_groups & val_groups)
        and not (train_groups & test_groups)
        and not (val_groups & test_groups)
    )

    record_check(
        "No conversation leakage across data splits",
        no_leakage,
        (
            f"Train={len(train_groups)}, "
            f"Validation={len(val_groups)}, "
            f"Test={len(test_groups)} conversations"
        ),
    )

    test_ids = {
        row["example_id"] for row in predictions
    }

    record_check(
        "Phase 5 has 24 unique test queries",
        len(test_ids) == 24,
        f"Found {len(test_ids)}",
    )

    record_check(
        "All test queries belong to held-out conversations",
        all(
            str(row["conversation_id"]) in test_groups
            for row in predictions
        ),
        "Checking test prediction conversation IDs",
    )

    prediction_budgets = sorted({
        float(row["budget"]) for row in predictions
    })

    record_check(
        "All four rewrite budgets are represented",
        prediction_budgets == BUDGETS,
        f"Found {prediction_budgets}",
    )

    budget_groups = {}

    for row in predictions:
        budget = float(row["budget"])
        budget_groups.setdefault(budget, []).append(row)

    record_check(
        "Each budget has 24 test predictions",
        all(
            len(budget_groups.get(budget, [])) == 24
            for budget in BUDGETS
        ),
        {
            str(budget): len(budget_groups.get(budget, []))
            for budget in BUDGETS
        },
    )

    complete_budget_ids = all(
        {
            row["example_id"]
            for row in budget_groups.get(budget, [])
        } == test_ids
        for budget in BUDGETS
    )

    record_check(
        "Every budget covers the same test queries",
        complete_budget_ids,
        "Comparing example IDs across budgets",
    )

    random_matches = {}

    for budget in BUDGETS:
        rows = budget_groups.get(budget, [])

        learned_count = sum(
            bool(row["learned_rewrite"]) for row in rows
        )

        random_count = sum(
            bool(row["random_rewrite"]) for row in rows
        )

        random_matches[str(budget)] = {
            "learned": learned_count,
            "random": random_count,
        }

    record_check(
        "Matched-random counts equal learned-gate counts",
        all(
            item["learned"] == item["random"]
            for item in random_matches.values()
        ),
        random_matches,
    )

    # ---------------------------------------------------------
    # 4. Independently reproduce Phase 6 results
    # ---------------------------------------------------------

    phase6_policies = {
        row["policy"]: row
        for row in phase6["policies"]
    }

    test_ids_sorted = sorted(test_ids)

    prediction_lookup = {
        (row["example_id"], float(row["budget"])): row
        for row in predictions
    }

    def expected_cost(decisions):
        selected = [
            uid
            for uid, decision in zip(test_ids_sorted, decisions)
            if decision
        ]

        entries = [cache_by_id[uid] for uid in selected]

        return {
            "rewrite_count": len(entries),
            "prompt_tokens": sum(
                row["prompt_tokens"] for row in entries
            ),
            "completion_tokens": sum(
                row["completion_tokens"] for row in entries
            ),
            "total_tokens": sum(
                row["total_tokens"] for row in entries
            ),
            "cumulative_api_latency_ms": sum(
                row["latency_ms"] for row in entries
            ),
        }

    def expected_quality(decisions, action="keep"):
        values = {metric: [] for metric in METRIC_NAMES}

        for uid, decision in zip(test_ids_sorted, decisions):
            selected_action = (
                "rewrite" if decision else action
            )

            row = metric_lookup[(uid, selected_action)]

            for metric in METRIC_NAMES:
                values[metric].append(float(row[metric]))

        return {
            metric: float(np.mean(scores))
            for metric, scores in values.items()
        }

    expected_policies = {}

    expected_policies["KEEP"] = (
        np.zeros(len(test_ids_sorted), dtype=bool),
        "keep",
    )

    expected_policies["REWRITE"] = (
        np.ones(len(test_ids_sorted), dtype=bool),
        "keep",
    )

    # CONCAT uses its own fixed formulation.
    concat_values = {
        metric: float(np.mean([
            metric_lookup[(uid, "concat")][metric]
            for uid in test_ids_sorted
        ]))
        for metric in METRIC_NAMES
    }

    expected_policies["CONCAT"] = (None, "concat")

    pilot_by_id = {
        row["turn_id"]: row for row in pilot
    }

    heuristic_decisions = []

    for uid in test_ids_sorted:
        turn = pilot_by_id[uid]

        features = extract_features(
            question=turn["question"],
            history=turn.get("history", []),
            turn_number=turn["turn_number"],
        )

        heuristic_decisions.append(
            should_rewrite(features)
        )

    expected_policies["HEURISTIC"] = (
        np.asarray(heuristic_decisions, dtype=bool),
        "keep",
    )

    for budget in BUDGETS:
        label = int(budget * 100)

        learned = np.array([
            bool(prediction_lookup[(uid, budget)]["learned_rewrite"])
            for uid in test_ids_sorted
        ])

        random = np.array([
            bool(prediction_lookup[(uid, budget)]["random_rewrite"])
            for uid in test_ids_sorted
        ])

        expected_policies[f"LEARNED_{label}PCT"] = (
            learned, "keep"
        )

        expected_policies[f"RANDOM_MATCHED_{label}PCT"] = (
            random, "keep"
        )

    reproduction_errors = []

    for name, (decisions, action) in expected_policies.items():
        if name not in phase6_policies:
            reproduction_errors.append(f"{name}: missing policy")
            continue

        saved = phase6_policies[name]

        if decisions is None:
            quality = concat_values
            cost = {
                "rewrite_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cumulative_api_latency_ms": 0.0,
            }
        else:
            quality = expected_quality(decisions, action)
            cost = expected_cost(decisions)

        for metric, expected in quality.items():
            actual = saved.get(metric)

            if actual is None or not close(actual, expected):
                reproduction_errors.append(
                    f"{name}: {metric} saved={actual}, "
                    f"recomputed={expected}"
                )

        for field, expected in cost.items():
            actual = saved.get(field)

            if actual is None or not close(actual, expected):
                reproduction_errors.append(
                    f"{name}: {field} saved={actual}, "
                    f"recomputed={expected}"
                )

    record_check(
        "Phase 6 quality and cost results reproduce independently",
        not reproduction_errors,
        reproduction_errors[:12]
        if reproduction_errors
        else "All policy metrics, tokens, calls and latency match",
    )

    # ---------------------------------------------------------
    # 5. Phase 5 and Phase 6 agreement
    # ---------------------------------------------------------

    phase5_policies = {
        row["policy"]: row
        for row in phase5["test_results"]
    }

    phase5_phase6_errors = []

    for name, phase5_row in phase5_policies.items():
        phase6_row = phase6_policies.get(name)

        if phase6_row is None:
            phase5_phase6_errors.append(
                f"{name}: absent from Phase 6"
            )
            continue

        if not close(
            phase5_row["mean_primary_metric"],
            phase6_row["ndcg@10"],
        ):
            phase5_phase6_errors.append(
                f"{name}: Phase 5 and Phase 6 nDCG differ"
            )

        if (
            phase5_row["rewrite_count"]
            != phase6_row["rewrite_count"]
        ):
            phase5_phase6_errors.append(
                f"{name}: rewrite counts differ"
            )

    record_check(
        "Phase 5 and Phase 6 test results agree",
        not phase5_phase6_errors,
        phase5_phase6_errors
        if phase5_phase6_errors
        else "nDCG@10 and rewrite counts match",
    )

    # ---------------------------------------------------------
    # 6. Final report
    # ---------------------------------------------------------

    report = {
        "audit": "Phase 7 experimental integrity audit",
        "checks_passed": sum(
            item["passed"] for item in checks
        ),
        "checks_failed": len(errors),
        "overall_status": "PASS" if not errors else "REVIEW_REQUIRED",
        "checks": checks,
        "errors": errors,
        "pilot": {
            "turns": len(pilot),
            "successful_rewrites": len(successful_pilot_ids),
            "outcomes": len(outcomes),
            "oracle_records": len(oracle),
        },
        "test": {
            "queries": len(test_ids),
            "conversations": len(test_groups),
            "budget_counts": random_matches,
        },
        "retrieval_statuses": retrieval_statuses,
    }

    with (OUTPUT_DIR / "phase7_audit.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 70)
    print("PHASE 7 AUDIT SUMMARY")
    print("=" * 70)
    print("Checks passed:", report["checks_passed"])
    print("Checks failed:", report["checks_failed"])
    print("Overall status:", report["overall_status"])
    print("\nSaved:", OUTPUT_DIR / "phase7_audit.json")

    if errors:
        print("\nPlease investigate the failed checks before reporting results.")
    else:
        print("\nAll integrity checks passed.")


if __name__ == "__main__":
    main()
