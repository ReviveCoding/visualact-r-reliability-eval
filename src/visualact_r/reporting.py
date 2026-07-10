from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from visualact_r.utils import atomic_write_text, ensure_dir, write_json


def _save(fig: Any, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=140)
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def generate_visualizations(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    figures = ensure_dir(run_dir / "figures")
    paths: list[str] = []
    comparison_path = run_dir / "candidate_comparison.csv"
    if comparison_path.exists():
        frame = pd.read_csv(comparison_path)
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(frame["candidate_id"], frame["grounded_action_success"])
            ax.set_title("Candidate Grounded Action Success")
            ax.set_ylabel("GAS")
            _save(fig, figures, "candidate_grounded_action_success")
            paths.append("figures/candidate_grounded_action_success.png")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(frame["evidence_contamination"], frame["critical_evidence_recall"])
            for _, row in frame.iterrows():
                ax.annotate(
                    row["candidate_id"],
                    (row["evidence_contamination"], row["critical_evidence_recall"]),
                )
            ax.set_title("Coverage-Contamination Frontier")
            ax.set_xlabel("Evidence contamination")
            ax.set_ylabel("Critical evidence recall")
            _save(fig, figures, "coverage_contamination_frontier")
            paths.append("figures/coverage_contamination_frontier.png")
            if "trace_coverage_rate" in frame:
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.bar(frame["candidate_id"], frame["trace_coverage_rate"])
                ax.set_ylim(0.0, 1.05)
                ax.set_title("Trace Coverage Integrity")
                ax.set_ylabel("Completed trace share")
                _save(fig, figures, "trace_coverage_integrity")
                paths.append("figures/trace_coverage_integrity.png")
    scenarios_path = run_dir / "decision_scenarios.csv"
    if scenarios_path.exists():
        frame = pd.read_csv(scenarios_path)
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            for candidate, group in frame.groupby("candidate_id"):
                ax.plot(
                    group["review_rate"],
                    group["false_auto_clear_rate_after_routing"],
                    marker="o",
                    label=candidate,
                )
            ax.set_title("Review-Capacity Frontier")
            ax.set_xlabel("Review rate")
            ax.set_ylabel("False auto-clear after routing")
            ax.legend(fontsize=8)
            _save(fig, figures, "review_capacity_frontier")
            paths.append("figures/review_capacity_frontier.png")
    slice_path = run_dir / "slice_metrics.csv"
    if slice_path.exists():
        frame = pd.read_csv(slice_path)
        if not frame.empty:
            pivot = frame.pivot_table(
                index="slice_dimension",
                columns="slice_value",
                values="grounded_action_success",
                aggfunc="mean",
            )
            fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), 3.5))
            image = ax.imshow(pivot.fillna(0).values, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            ax.set_title("Hard-Slice Grounded Action Success")
            fig.colorbar(image, ax=ax)
            _save(fig, figures, "slice_gas_heatmap")
            paths.append("figures/slice_gas_heatmap.png")
    paired_path = run_dir / "paired_comparisons.csv"
    if paired_path.exists():
        frame = pd.read_csv(paired_path)
        required = {"comparison_id", "delta_mean", "ci95_low", "ci95_high", "status"}
        if not frame.empty and required.issubset(frame):
            plot = frame[frame["status"] == "PASS"].dropna(
                subset=["delta_mean", "ci95_low", "ci95_high"]
            )
            if not plot.empty:
                fig, ax = plt.subplots(figsize=(9, 4))
                y = list(range(len(plot)))
                lower = plot["delta_mean"] - plot["ci95_low"]
                upper = plot["ci95_high"] - plot["delta_mean"]
                ax.errorbar(plot["delta_mean"], y, xerr=[lower, upper], fmt="o")
                ax.axvline(0.0, linewidth=1)
                ax.set_yticks(y, plot["comparison_id"])
                ax.set_title("Paired Bootstrap Candidate Deltas")
                ax.set_xlabel("Right minus left delta")
                _save(fig, figures, "paired_bootstrap_deltas")
                paths.append("figures/paired_bootstrap_deltas.png")
    return paths


def _markdown_table(path: Path) -> str:
    if not path.exists():
        return "No artifact available."
    frame = pd.read_csv(path)
    return frame.to_markdown(index=False) if not frame.empty else "No rows available."


def _read_optional_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def generate_reports(
    run_dir: str | Path, release: dict[str, Any], manifest: dict[str, Any]
) -> None:
    run_dir = Path(run_dir)
    figures = generate_visualizations(run_dir)
    comparison = _markdown_table(run_dir / "candidate_comparison.csv")
    paired = _markdown_table(run_dir / "paired_comparisons.csv")
    scope = str(manifest.get("scope", "unknown"))
    fixture_scope = scope.startswith("fixture")
    claim_boundary = (
        "This fixture report validates interfaces, artifact lineage, and fail-closed contracts only. "
        "It cannot support model-performance claims."
        if fixture_scope
        else "This offline frozen-final report supports only the release wording permitted by the saved gates, "
        "paired statistics, and scope metadata. It does not support production or safety claims."
    )
    cef_gate = _read_optional_json(run_dir / "cef_development_gate.json")
    rctr_stress = _read_optional_json(run_dir / "rctr_controlled_stress_test.json")
    routing = _read_optional_json(run_dir / "rctr_routing_report.json")
    counterfactual = _markdown_table(run_dir / "counterfactual_evidence.csv")
    markdown = "\n".join(
        [
            "# VisualAct-R Evaluation Report",
            "",
            f"**Scope:** {scope}",
            f"**Release status:** {release.get('status')}",
            "",
            "## Claim boundary",
            claim_boundary,
            "",
            "## CEF development gate",
            f"`{cef_gate}`" if cef_gate else "No CEF development gate artifact available.",
            "",
            "## RCTR pre-decision routing contract",
            f"`{routing}`" if routing else "No frozen-final routing artifact available.",
            "",
            "## RCTR controlled routing contract",
            f"`{rctr_stress}`"
            if rctr_stress
            else "No controlled routing contract artifact available.",
            "",
            "## Candidate comparison",
            comparison,
            "",
            "## Paired comparisons",
            paired,
            "",
            "## Counterfactual evidence status",
            counterfactual,
            "",
            "## Generated figures",
            *[f"- `{path}`" for path in figures],
            "",
            "## Release rationale",
            str(release.get("reason")),
        ]
    )
    atomic_write_text(run_dir / "report.md", markdown)
    html = (
        "<html><body><pre>"
        + markdown.replace("&", "&amp;").replace("<", "&lt;")
        + "</pre></body></html>"
    )
    atomic_write_text(run_dir / "report.html", html)
    write_json(run_dir / "release_decision.json", release)
