
"""Phase 4: Retrieve passages for KEEP, CONCAT, and REWRITE."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.retrieval.retrieve import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Load experiment configuration."""
    config_path = PROJECT_ROOT / "configs" / "experiment.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()

    # Input: Phase 3 formulation results
    input_path = (
        PROJECT_ROOT
        / config["output"]["results_dir"]
        / "formulation_pilot.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Formulation results not found: {input_path}"
        )

    # Retrieval configuration
    retrieval_config = config["retrieval"]
    top_k = int(retrieval_config.get("top_k", 100))

    index_path = PROJECT_ROOT / retrieval_config.get(
        "index_path",
        "data/index/bm25_fts5_dedup_full.sqlite",
    )

    # Output directory
    output_dir = (
        PROJECT_ROOT
        / config["output"]["results_dir"]
        / "phase4"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "retrieval_results.jsonl"

    formulations = ["keep", "concat", "rewrite"]

    total_turns = 0
    successful_retrievals = 0
    failed_retrievals = 0

    print(f"Input: {input_path}")
    print(f"BM25 index: {index_path}")
    print(f"Top K: {top_k}")
    print("\nStarting retrieval evaluation...\n")

    # Open the BM25 index only once.
    with BM25Retriever(index_path) as retriever:

        print("Index metadata:")
        print(retriever.metadata())
        print()

        with input_path.open("r", encoding="utf-8") as input_file, \
             output_path.open("w", encoding="utf-8") as output_file:

            for line in input_file:
                if not line.strip():
                    continue

                record = json.loads(line)
                total_turns += 1

                result = {
                    "example_id": record["turn_id"],
                    "conversation_id": record["conversation_id"],
                    "turn_number": record["turn_number"],
                    "relevant_passage_ids": record[
                        "relevant_passage_ids"
                    ],
                    "retrieval": {},
                }

                for method in formulations:
                    query = record.get("formulations", {}).get(method)

                    if not query:
                        result["retrieval"][method] = {
                            "status": "missing_query",
                            "ranked_results": [],
                        }
                        continue

                    try:
                        ranked_results = retriever.retrieve(
                            query,
                            top_k=top_k,
                        )

                        result["retrieval"][method] = {
                            "status": "success",
                            "query": query,
                            "ranked_results": [
                                {
                                    "passage_id": passage_id,
                                    "bm25_score": score,
                                }
                                for passage_id, score in ranked_results
                            ],
                        }

                        successful_retrievals += 1

                    except Exception as error:
                        result["retrieval"][method] = {
                            "status": "failed",
                            "error": str(error),
                            "ranked_results": [],
                        }

                        failed_retrievals += 1

                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_file.flush()

                if total_turns % 10 == 0:
                    print(
                        f"Processed {total_turns} turns | "
                        f"Successful: {successful_retrievals} | "
                        f"Failed: {failed_retrievals}"
                    )

    print("\nRetrieval evaluation complete.")
    print(f"Turns processed: {total_turns}")
    print(f"Successful retrievals: {successful_retrievals}")
    print(f"Failed retrievals: {failed_retrievals}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
