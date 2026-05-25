#!/usr/bin/env python
"""Build summary tables and plots for own-FastText prefix-matrix results."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EVAL = Path("models") / "artifacts" / "prefix_eval_own_fasttext_full.csv"
DEFAULT_ARTIFACT = Path("models") / "artifacts" / "prefix_matrices_own_fasttext.pkl"
DEFAULT_COVERAGE = Path("models") / "embeddings" / "own_serbian_fasttext_coverage.csv"
DEFAULT_OUTPUT_DIR = Path("results") / "own_fasttext_prefix_report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report for own-FastText prefix matrices.")
    parser.add_argument("--eval-csv", default=str(DEFAULT_EVAL), help="Full evaluation CSV.")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT), help="Training artifact pickle.")
    parser.add_argument("--coverage-csv", default=str(DEFAULT_COVERAGE), help="Embedding coverage CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for report outputs.")
    return parser.parse_args()


def read_eval(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Evaluation CSV not found: {path}")

    df = pd.read_csv(path)
    needed = ["base_verb", "prefixed_verb", "prefix", "cos_global", "cos_prefix"]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise SystemExit(f"Evaluation CSV is missing columns: {missing}")

    df["cos_global"] = pd.to_numeric(df["cos_global"], errors="coerce")
    df["cos_prefix"] = pd.to_numeric(df["cos_prefix"], errors="coerce")
    df["delta_prefix_minus_global"] = df["cos_prefix"] - df["cos_global"]
    return df


def read_artifact(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def read_coverage(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    needed = ["prefix", "both_in_vocab"]
    if any(column not in df.columns for column in needed):
        return None

    df["both_in_vocab"] = df["both_in_vocab"].astype(bool)
    coverage = (
        df.groupby("prefix", as_index=False)
        .agg(
            pairs_total=("prefix", "size"),
            pairs_with_both_vectors=("both_in_vocab", "sum"),
        )
        .sort_values(["pairs_total", "prefix"], ascending=[False, True])
    )
    coverage["coverage"] = coverage["pairs_with_both_vectors"] / coverage["pairs_total"]
    return coverage


def build_prefix_summary(eval_df: pd.DataFrame, coverage_df: pd.DataFrame | None) -> pd.DataFrame:
    summary = (
        eval_df.groupby("prefix", as_index=False)
        .agg(
            evaluated_pairs=("prefix", "size"),
            mean_cos_global=("cos_global", "mean"),
            mean_cos_prefix=("cos_prefix", "mean"),
            median_cos_global=("cos_global", "median"),
            median_cos_prefix=("cos_prefix", "median"),
            mean_delta_prefix_minus_global=("delta_prefix_minus_global", "mean"),
        )
        .sort_values(["evaluated_pairs", "prefix"], ascending=[False, True])
    )

    if coverage_df is not None:
        summary = summary.merge(coverage_df, on="prefix", how="left")
        first_cols = ["prefix", "pairs_total", "pairs_with_both_vectors", "coverage", "evaluated_pairs"]
        other_cols = [column for column in summary.columns if column not in first_cols]
        summary = summary[first_cols + other_cols]

    return summary


def save_plots(prefix_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return []

    paths: list[Path] = []

    plot_df = prefix_summary.sort_values("evaluated_pairs", ascending=False).copy()
    x = np.arange(len(plot_df))
    width = 0.4

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, plot_df["mean_cos_global"], width, label="Global matrix")
    ax.bar(x + width / 2, plot_df["mean_cos_prefix"], width, label="Prefix matrix")
    ax.set_title("Mean cosine by prefix")
    ax.set_ylabel("Mean cosine")
    ax.set_xlabel("Prefix")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["prefix"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "prefix_cosine_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    delta_df = prefix_summary.dropna(subset=["mean_delta_prefix_minus_global"]).copy()
    delta_df = delta_df.sort_values("mean_delta_prefix_minus_global", ascending=False)
    colors = ["#2f7d32" if value >= 0 else "#b23b3b" for value in delta_df["mean_delta_prefix_minus_global"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(delta_df["prefix"], delta_df["mean_delta_prefix_minus_global"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Prefix matrix advantage over global matrix")
    ax.set_ylabel("Mean cosine difference")
    ax.set_xlabel("Prefix")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "prefix_delta_over_global.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if "coverage" in prefix_summary.columns:
        coverage_df = prefix_summary.dropna(subset=["coverage"]).sort_values("coverage", ascending=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(coverage_df["prefix"], coverage_df["coverage"])
        ax.set_title("Embedding coverage by prefix")
        ax.set_ylabel("Share of pairs with both vectors")
        ax.set_xlabel("Prefix")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = output_dir / "prefix_embedding_coverage.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    view = df.loc[:, columns].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else str(value))

    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, separator] + rows)


def write_markdown_report(
    output_path: Path,
    eval_df: pd.DataFrame,
    prefix_summary: pd.DataFrame,
    artifact: dict,
    plot_paths: list[Path],
) -> None:
    rows_evaluated = len(eval_df)
    mean_global = eval_df["cos_global"].mean()
    mean_prefix = eval_df["cos_prefix"].dropna().mean()
    rows_with_prefix = int(eval_df["cos_prefix"].notna().sum())

    artifact_config = artifact.get("config", {}) if artifact else {}
    artifact_metrics = artifact.get("metrics", {}) if artifact else {}

    best_prefix = prefix_summary.dropna(subset=["mean_cos_prefix"]).sort_values(
        "mean_cos_prefix", ascending=False
    )
    best_delta = prefix_summary.dropna(subset=["mean_delta_prefix_minus_global"]).sort_values(
        "mean_delta_prefix_minus_global", ascending=False
    )

    lines = [
        "# Own FastText Prefix-Matrix Report",
        "",
        "## Overall",
        "",
        f"- Evaluated pairs: {rows_evaluated}",
        f"- Pairs with prefix-specific matrix score: {rows_with_prefix}",
        f"- Mean cosine, global matrix: {mean_global:.4f}",
        f"- Mean cosine, prefix matrices: {mean_prefix:.4f}",
    ]

    if artifact_metrics:
        lines.extend(
            [
                "",
                "## Holdout From Training Artifact",
                "",
                f"- Holdout mean cosine, global matrix: {artifact_metrics.get('holdout_mean_cos_global', float('nan')):.4f}",
                f"- Holdout mean cosine, prefix matrices: {artifact_metrics.get('holdout_mean_cos_prefix', float('nan')):.4f}",
            ]
        )

    if artifact_config:
        lines.extend(
            [
                "",
                "## Training Setup",
                "",
                f"- Dataset: `{artifact_config.get('dataset', '')}`",
                f"- Embeddings: `{artifact_config.get('embeddings', '')}`",
                f"- Dimension: {artifact_config.get('dim', '')}",
                f"- Alpha: {artifact_config.get('alpha', '')}",
                f"- Min pairs: {artifact_config.get('min_pairs', '')}",
                f"- Train rows: {artifact_config.get('train_rows', '')}",
                f"- Test rows: {artifact_config.get('test_rows', '')}",
                f"- Trained prefixes: {artifact_config.get('trained_prefixes', '')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Best Prefixes By Mean Prefix Cosine",
            "",
            markdown_table(
                best_prefix,
                ["prefix", "evaluated_pairs", "mean_cos_global", "mean_cos_prefix"],
            ),
            "",
            "## Largest Prefix-Matrix Advantage",
            "",
            markdown_table(
                best_delta,
                ["prefix", "evaluated_pairs", "mean_delta_prefix_minus_global", "mean_cos_global", "mean_cos_prefix"],
            ),
            "",
            "## Plots",
            "",
        ]
    )

    if plot_paths:
        lines.extend(f"- `{path.name}`" for path in plot_paths)
    else:
        lines.append("- Plots were not generated.")

    lines.extend(
        [
            "",
            "## Suggested Interpretation",
            "",
            (
                "The own FastText embeddings cover most verb pairs in the prefix dataset. "
                "On the full set of covered pairs, prefix-specific matrices achieve a higher "
                "mean cosine than the global matrix. The holdout metric from the training artifact "
                "should be treated as the stricter evaluation, while the full-set metric describes "
                "how well the learned matrices reproduce the covered dataset structure."
            ),
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_csv)
    artifact_path = Path(args.artifact)
    coverage_path = Path(args.coverage_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = read_eval(eval_path)
    artifact = read_artifact(artifact_path)
    coverage_df = read_coverage(coverage_path)
    prefix_summary = build_prefix_summary(eval_df, coverage_df)

    summary_path = output_dir / "prefix_summary_own_fasttext.csv"
    prefix_summary.to_csv(summary_path, index=False)

    plot_paths = save_plots(prefix_summary, output_dir)

    report_path = output_dir / "own_fasttext_prefix_report.md"
    write_markdown_report(report_path, eval_df, prefix_summary, artifact, plot_paths)

    print("Saved prefix summary:", summary_path)
    print("Saved markdown report:", report_path)
    for path in plot_paths:
        print("Saved plot:", path)


if __name__ == "__main__":
    main()
