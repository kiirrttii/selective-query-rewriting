"""
Generate publication-ready Phase 7 figures (PNG) and Excel results tables.

Run from the project root:
    python -m scripts.generate_phase7_publication_results

Inputs are the saved Phase 4/5/6/7 outputs. This script does not modify
any experiment outputs; it only creates figures, workbooks and a summary.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
PHASE4 = RESULTS / "phase4"
PHASE5 = RESULTS / "phase5"
PHASE6 = RESULTS / "phase6"
PHASE7 = RESULTS / "phase7"

FIGURES = PHASE7 / "figures"
TABLES = PHASE7 / "tables"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# Consistent, print-friendly defaults.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {FIGURES / filename}")


def add_sheet(workbook, title, headers, rows, note=None):
    ws = workbook.create_sheet(title=title)
    ws.append(headers)

    for row in rows:
        ws.append(list(row))

    header_fill = PatternFill("solid", fgColor="243B53")
    header_font = Font(color="FFFFFF", bold=True)
    thin_gray = Side(style="thin", color="D9E2EC")

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(bottom=thin_gray)

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top")
            cell.border = Border(bottom=Side(style="hair", color="D9E2EC"))

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 28

    for col in ws.columns:
        max_len = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in col
        )
        width = min(max(max_len + 3, 13), 42)
        ws.column_dimensions[get_column_letter(col[0].column)].width = width

    if note:
        note_row = ws.max_row + 2
        ws.cell(note_row, 1, "Note")
        ws.cell(note_row, 1).font = Font(bold=True, color="243B53")
        ws.cell(note_row, 2, note)
        ws.cell(note_row, 2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(
            start_row=note_row, start_column=2,
            end_row=note_row, end_column=max(2, len(headers))
        )
        ws.row_dimensions[note_row].height = 42

    return ws


def style_workbook(workbook):
    # Keep the first worksheet visible and set a useful zoom level.
    for ws in workbook.worksheets:
        ws.sheet_view.showGridLines = False
        ws.sheet_view.zoomScale = 90


def generate_outcome_chart(outcome_summary):
    counts = outcome_summary["counts"]
    helpful = counts.get("helpful", counts.get("HELPFUL", 0))
    tied = counts.get("tied", counts.get("TIED", 0))
    harmful = counts.get("harmful", counts.get("HARMFUL", 0))
    total = helpful + tied + harmful

    if total == 0:
        raise ValueError("Outcome summary contains no outcome records.")

    fig, ax = plt.subplots(figsize=(8.5, 2.7))
    colors = ["#27865C", "#A7ADB4", "#C65B5B"]
    labels = [
        f"Helpful\n{helpful} ({helpful / total:.1%})",
        f"Tied\n{tied} ({tied / total:.1%})",
        f"Harmful\n{harmful} ({harmful / total:.1%})",
    ]

    left = 0
    for value, color, label in zip([helpful, tied, harmful], colors, labels):
        ax.barh([0], [value], left=left, color=color, height=0.48, label=label)
        if value / total >= 0.07:
            ax.text(
                left + value / 2, 0, label,
                ha="center", va="center", color="white",
                fontsize=9, fontweight="bold"
            )
        left += value

    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel("Number of comparable turns")
    ax.set_title("Distribution of rewrite outcomes")
    ax.text(
        1.0, -0.42, f"N = {total} comparable turns",
        transform=ax.transAxes, ha="right", va="top", color="#52606D"
    )
    save_figure(fig, "fig1_outcome_distribution.png")


def generate_oracle_chart():
    # Phase 4 values are computed on the same 134 comparable turns.
    strategies = ["KEEP", "CONCAT", "REWRITE"]
    scores = [0.1350, 0.4405, 0.3779]
    oracle = 0.5491
    headroom = [oracle - score for score in scores]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(strategies, headroom, color=["#7B8794", "#27865C", "#4C78A8"])
    ax.set_ylabel("Oracle nDCG@10 minus strategy nDCG@10")
    ax.set_title("Retrieval headroom relative to the oracle")
    ax.set_ylim(0, max(headroom) * 1.22)

    for bar, value in zip(bars, headroom):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{value:.4f}",
            ha="center", va="bottom", fontweight="bold"
        )

    ax.text(
        0.99, 0.98,
        f"Oracle nDCG@10 = {oracle:.4f}\nN = 134",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0F4F8", edgecolor="none")
    )
    save_figure(fig, "fig2_oracle_headroom.png")


def generate_quality_curve():
    # Each point is a saved held-out test policy result.
    # CONCAT is a separate formulation baseline, not a KEEP-based gate.
    points = [
        ("KEEP", 0.0, 0.0942, "o", "#52606D"),
        ("CONCAT", 0.0, 0.3540, "s", "#27865C"),
        ("Learned 25%", 33.33, 0.2046, "o", "#4C78A8"),
        ("Random 25%", 33.33, 0.2138, "^", "#E09F3E"),
        ("Learned 50%", 37.50, 0.2046, "o", "#4C78A8"),
        ("Random 50%", 37.50, 0.1856, "^", "#E09F3E"),
        ("Learned 75%", 87.50, 0.3377, "o", "#4C78A8"),
        ("Random 75%", 87.50, 0.2825, "^", "#E09F3E"),
        ("Heuristic", 95.83, 0.3238, "D", "#8A6FB0"),
        ("Always rewrite", 100.0, 0.3377, "X", "#C65B5B"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    for label, rate, score, marker, color in points:
        ax.scatter(
            rate, score, marker=marker, s=75, color=color,
            edgecolor="white", linewidth=0.7, zorder=3, label=label
        )
        # Small offsets reduce overlap for duplicate x coordinates.
        offset_x = 1.0
        offset_y = 0.008
        if label == "KEEP":
            offset_y = -0.022
        elif label == "CONCAT":
            offset_y = 0.012
        elif label == "Random 25%":
            offset_y = 0.014
        elif label == "Learned 50%":
            offset_y = -0.022
        elif label == "Random 50%":
            offset_y = -0.022
        elif label == "Random 75%":
            offset_y = -0.024
        elif label == "Heuristic":
            offset_y = -0.025
        ax.annotate(
            label, (rate, score),
            xytext=(offset_x, offset_y * 100),
            textcoords="offset points", fontsize=8
        )

    ax.set_xlabel("Actual rewrite rate on held-out test set (%)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Retrieval quality versus actual rewrite rate")
    ax.set_xlim(-5, 108)
    ax.set_ylim(0.05, 0.41)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.02, 0.02,
        "CONCAT is a separate formulation baseline; its 0% refers to LLM rewrite calls.",
        transform=ax.transAxes, fontsize=8, color="#52606D"
    )
    save_figure(fig, "fig3_quality_vs_rewrite_rate.png")


def generate_error_chart(prediction_rows):
    budgets = [0.25, 0.50, 0.75, 1.00]
    labels = ["25%", "50%", "75%", "100%"]
    helpful_counts, tied_counts, harmful_counts, missed_counts = [], [], [], []

    for budget in budgets:
        rows = [r for r in prediction_rows if float(r["budget"]) == budget]
        selected = [r for r in rows if bool(r["learned_rewrite"])]

        helpful_counts.append(sum(str(r["outcome"]).lower() == "helpful" for r in selected))
        tied_counts.append(sum(str(r["outcome"]).lower() == "tied" for r in selected))
        harmful_counts.append(sum(str(r["outcome"]).lower() == "harmful" for r in selected))

        missed_counts.append(sum(
            str(r["outcome"]).lower() == "helpful"
            and not bool(r["learned_rewrite"])
            for r in rows
        ))

    x = np.arange(len(budgets))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    ax.bar(x, helpful_counts, color="#27865C", label="Helpful selected")
    ax.bar(
        x, tied_counts, bottom=helpful_counts,
        color="#A7ADB4", label="Tied selected"
    )
    bottoms = np.array(helpful_counts) + np.array(tied_counts)
    ax.bar(
        x, harmful_counts, bottom=bottoms,
        color="#C65B5B", label="Harmful selected"
    )

    for i, (h, t, bad, missed) in enumerate(
        zip(helpful_counts, tied_counts, harmful_counts, missed_counts)
    ):
        selected_total = h + t + bad
        ax.text(
            i, selected_total + 0.35,
            f"Missed helpful: {missed}",
            ha="center", va="bottom", fontsize=8
        )

    ax.set_xticks(x, labels)
    ax.set_xlabel("Configured rewrite budget")
    ax.set_ylabel("Selected rewrites (count)")
    ax.set_title("Outcome composition of learned-gate selections")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, 27)
    ax.text(
        0.99, 0.98, "Held-out test set: 24 turns",
        transform=ax.transAxes, ha="right", va="top",
        color="#52606D", fontsize=8
    )
    save_figure(fig, "fig4_gate_error_analysis.png")


def generate_feature_chart(category_rows):
    # Pronoun categories are the clearest interpretable signal in this pilot.
    pronoun_rows = [
        r for r in category_rows
        if r.get("feature") == "pronoun"
    ]

    if not pronoun_rows:
        print("[SKIP] No pronoun feature rows found.")
        return

    order = ["contains pronoun", "no pronoun"]
    by_category = {r["category"]: r for r in pronoun_rows}
    categories = [c for c in order if c in by_category]

    helpful = [float(by_category[c]["helpful_percent"]) for c in categories]
    tied = [float(by_category[c]["tied_percent"]) for c in categories]
    harmful = [float(by_category[c]["harmful_percent"]) for c in categories]

    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.bar(x, helpful, color="#27865C", label="Helpful")
    ax.bar(x, tied, bottom=helpful, color="#A7ADB4", label="Tied")
    ax.bar(
        x, harmful,
        bottom=np.array(helpful) + np.array(tied),
        color="#C65B5B", label="Harmful"
    )

    ax.set_xticks(x, ["Contains pronoun", "No pronoun"])
    ax.set_ylabel("Share of queries (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Rewrite outcomes by pronoun presence")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.text(
        0.99, 0.02,
        "Descriptive pilot analysis; N=134",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#52606D"
    )
    save_figure(fig, "fig5_feature_analysis.png")


def generate_main_workbook():
    wb = Workbook()
    default = wb.active
    wb.remove(default)

    formulation_rows = [
        ["KEEP", 0.1177, 0.1903, 0.3731, 0.1350],
        ["CONCAT", 0.3718, 0.6679, 0.8731, 0.4405],
        ["REWRITE", 0.3243, 0.5672, 0.8209, 0.3779],
        ["Oracle (upper bound)", None, None, None, 0.5491],
    ]

    add_sheet(
        wb, "Phase 4 formulations",
        ["Formulation", "MRR@10", "Recall@10", "Recall@100", "nDCG@10"],
        formulation_rows,
        note=(
            "134 comparable pilot turns. Oracle is a non-deployable "
            "upper bound selected using observed per-query retrieval outcomes."
        ),
    )

    gate_rows = [
        ["KEEP", 0.0942, 0.00, 0],
        ["CONCAT", 0.3540, 0.00, 0],
        ["REWRITE", 0.3377, 100.00, 24],
        ["HEURISTIC", 0.3238, 95.83, 23],
        ["LEARNED 25%", 0.2046, 33.33, 8],
        ["RANDOM 25%", 0.2138, 33.33, 8],
        ["LEARNED 50%", 0.2046, 37.50, 9],
        ["RANDOM 50%", 0.1856, 37.50, 9],
        ["LEARNED 75%", 0.3377, 87.50, 21],
        ["RANDOM 75%", 0.2825, 87.50, 21],
        ["LEARNED 100%", 0.3377, 100.00, 24],
        ["RANDOM 100%", 0.3377, 100.00, 24],
    ]

    add_sheet(
        wb, "Phase 5 held-out test",
        ["Policy", "nDCG@10", "Actual rewrite rate (%)", "Rewrite calls"],
        gate_rows,
        note=(
            "24 held-out test turns. Rewrite rates are actual test rates, "
            "not configured budget targets. CONCAT is a separate formulation."
        ),
    )

    style_workbook(wb)
    path = TABLES / "table1_main_results.xlsx"
    wb.save(path)
    print(f"[SAVED] {path}")


def generate_efficiency_workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Phase 6 efficiency"

    rows = [
        ["KEEP", 0.0942, 0, 0, 0.00],
        ["CONCAT", 0.3540, 0, 0, 0.00],
        ["REWRITE", 0.3377, 24, 9476, 89.69],
        ["HEURISTIC", 0.3238, 23, 9012, 77.28],
        ["LEARNED 25%", 0.2046, 8, 3430, 34.86],
        ["RANDOM 25%", 0.2138, 8, 2580, 20.67],
        ["LEARNED 50%", 0.2046, 9, 3822, 37.82],
        ["RANDOM 50%", 0.1856, 9, 3350, 28.70],
        ["LEARNED 75%", 0.3377, 21, 8608, 83.34],
        ["RANDOM 75%", 0.2825, 21, 8110, 75.94],
        ["LEARNED 100%", 0.3377, 24, 9476, 89.69],
        ["RANDOM 100%", 0.3377, 24, 9476, 89.69],
    ]

    add_sheet(
        wb, "Phase 6 efficiency",
        [
            "Policy", "nDCG@10", "Rewrite calls",
            "Total tokens", "Cumulative API latency (seconds)"
        ],
        rows,
        note=(
            "Held-out test policy costs are retrospective estimates based on "
            "observed per-query rewrite usage. Cumulative API latency is the "
            "sum of request latencies, not parallel deployment wall-clock time. "
            "CONCAT's zero refers to LLM rewrite calls; retrieval computation "
            "cost is not included. Monetary cost is not estimated."
        ),
    )

    # Separate sheet for pilot rewrite usage, excluding the unrelated manual entry.
    usage = wb.create_sheet("Pilot rewrite usage")
    usage.append(["Measure", "Value"])
    usage.append(["Successful QReCC rewrites", 134])
    usage.append(["Prompt tokens", 20830])
    usage.append(["Completion tokens", 29515])
    usage.append(["Total tokens", 50345])
    usage.append(["Cumulative API latency (ms)", 375150.16])
    usage.append(["Mean API latency per call (ms)", 2799.63])

    for cell in usage[1]:
        cell.fill = PatternFill("solid", fgColor="243B53")
        cell.font = Font(color="FFFFFF", bold=True)
    usage.column_dimensions["A"].width = 42
    usage.column_dimensions["B"].width = 22
    usage.freeze_panes = "A2"
    usage.sheet_view.showGridLines = False

    style_workbook(wb)
    path = TABLES / "table2_efficiency_results.xlsx"
    wb.save(path)
    print(f"[SAVED] {path}")


def main():
    required = [
        PHASE7 / "phase7_outcome_summary.json",
        PHASE7 / "feature_category_outcomes.csv",
        PHASE5 / "test_predictions.jsonl",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files. Run the Phase 7 failure analysis first:\n"
            + "\n".join(str(p) for p in missing)
        )

    outcome_summary = read_json(required[0])
    category_rows = read_csv(required[1])
    predictions = read_jsonl(required[2])

    print("Generating Phase 7 publication results...")

    generate_outcome_chart(outcome_summary)
    generate_oracle_chart()
    generate_quality_curve()
    generate_error_chart(predictions)
    generate_feature_chart(category_rows)

    generate_main_workbook()
    generate_efficiency_workbook()

    summary = {
        "figures": [
            "fig1_outcome_distribution.png",
            "fig2_oracle_headroom.png",
            "fig3_quality_vs_rewrite_rate.png",
            "fig4_gate_error_analysis.png",
            "fig5_feature_analysis.png",
        ],
        "tables": [
            "table1_main_results.xlsx",
            "table2_efficiency_results.xlsx",
        ],
        "important_caveats": [
            "Phase 4 formulation comparison uses 134 comparable pilot turns.",
            "Phase 5/6 policy comparison uses 24 held-out test turns.",
            "Actual test rewrite rates differ from configured budget targets.",
            "Oracle is a non-deployable upper bound.",
            "Phase 6 costs are retrospective estimates based on observed "
            "per-query rewrite costs.",
            "The pilot and held-out test set are small; avoid strong "
            "generalization claims.",
        ],
    }

    with open(
        PHASE7 / "publication_results_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nAll publication outputs generated.")
    print(f"Figures: {FIGURES}")
    print(f"Tables:  {TABLES}")
    print(f"Summary: {PHASE7 / 'publication_results_summary.json'}")


if __name__ == "__main__":
    main()
