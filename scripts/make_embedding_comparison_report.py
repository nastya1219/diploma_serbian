#!/usr/bin/env python
"""Create a comparison report for own FastText and lemma Word2Vec results."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    short_label: str
    eval_csv: Path
    artifact: Path
    coverage_csv: Path
    description: str


DEFAULT_DATASET = Path("notebooks") / "serbian_verb_final_hf_wiki_5_ruscorpora_combined_50.csv"
DEFAULT_OUTPUT_DIR = Path("results") / "own_embeddings_comparison_report"
DEFAULT_SPECS = [
    ModelSpec(
        key="own_fasttext",
        label="Собственный FastText",
        short_label="FastText",
        eval_csv=Path("models") / "artifacts" / "prefix_eval_own_fasttext_full.csv",
        artifact=Path("models") / "artifacts" / "prefix_matrices_own_fasttext.pkl",
        coverage_csv=Path("models") / "embeddings" / "own_serbian_fasttext_coverage.csv",
        description="модель с субсловными n-граммами, обученная на локальных сербских текстах/XML-корпусе",
    ),
    ModelSpec(
        key="own_word2vec_lemmas",
        label="Собственный Word2Vec по леммам",
        short_label="Word2Vec lemmas",
        eval_csv=Path("models") / "artifacts" / "prefix_eval_own_word2vec_lemmas_full.csv",
        artifact=Path("models") / "artifacts" / "prefix_matrices_own_word2vec_lemmas.pkl",
        coverage_csv=Path("models") / "embeddings" / "own_serbian_word2vec_lemmas_coverage.csv",
        description="модель Word2Vec, обученная на последовательностях сербских лемм",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create own-embedding comparison report.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Verb-pair dataset.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for report outputs.")
    return parser.parse_args()


def require_path(path: Path, kind: str) -> None:
    if not path.exists():
        raise SystemExit(f"{kind} not found: {path}")


def read_eval(path: Path) -> pd.DataFrame:
    require_path(path, "Evaluation CSV")
    df = pd.read_csv(path)
    needed = ["base_verb", "prefixed_verb", "prefix", "cos_global", "cos_prefix"]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise SystemExit(f"Evaluation CSV {path} is missing columns: {missing}")
    df = df.loc[:, needed].copy()
    df["cos_global"] = pd.to_numeric(df["cos_global"], errors="coerce")
    df["cos_prefix"] = pd.to_numeric(df["cos_prefix"], errors="coerce")
    df["delta_prefix_minus_global"] = df["cos_prefix"] - df["cos_global"]
    return df


def read_artifact(path: Path) -> dict[str, Any]:
    require_path(path, "Artifact")
    with path.open("rb") as f:
        artifact = pickle.load(f)
    if not isinstance(artifact, dict):
        raise SystemExit(f"Artifact has unexpected format: {path}")
    return artifact


def read_coverage(path: Path) -> pd.DataFrame:
    require_path(path, "Coverage CSV")
    df = pd.read_csv(path)
    needed = ["prefix", "both_in_vocab"]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise SystemExit(f"Coverage CSV {path} is missing columns: {missing}")
    df = df.loc[:, needed].copy()
    df["both_in_vocab"] = df["both_in_vocab"].astype(bool)
    out = (
        df.groupby("prefix", as_index=False)
        .agg(
            pairs_total=("prefix", "size"),
            pairs_with_both_vectors=("both_in_vocab", "sum"),
        )
        .sort_values(["pairs_total", "prefix"], ascending=[False, True])
    )
    out["coverage"] = out["pairs_with_both_vectors"] / out["pairs_total"]
    return out


def build_prefix_summary(eval_df: pd.DataFrame, coverage_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        eval_df.groupby("prefix", as_index=False)
        .agg(
            evaluated_pairs=("prefix", "size"),
            prefix_scored_pairs=("cos_prefix", lambda s: int(s.notna().sum())),
            mean_cos_global=("cos_global", "mean"),
            mean_cos_prefix=("cos_prefix", "mean"),
            median_cos_global=("cos_global", "median"),
            median_cos_prefix=("cos_prefix", "median"),
            mean_delta_prefix_minus_global=("delta_prefix_minus_global", "mean"),
        )
        .sort_values(["evaluated_pairs", "prefix"], ascending=[False, True])
    )
    summary = summary.merge(coverage_df, on="prefix", how="left")
    first_cols = [
        "prefix",
        "pairs_total",
        "pairs_with_both_vectors",
        "coverage",
        "evaluated_pairs",
        "prefix_scored_pairs",
    ]
    return summary[first_cols + [column for column in summary.columns if column not in first_cols]]


def summarize_model(spec: ModelSpec, dataset_rows: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    eval_df = read_eval(spec.eval_csv)
    artifact = read_artifact(spec.artifact)
    coverage_df = read_coverage(spec.coverage_csv)
    prefix_summary = build_prefix_summary(eval_df, coverage_df)

    config = artifact.get("config", {})
    metrics = artifact.get("metrics", {})
    coverage_pairs = int(coverage_df["pairs_with_both_vectors"].sum())
    coverage_total = int(coverage_df["pairs_total"].sum())

    row = {
        "model_key": spec.key,
        "model": spec.label,
        "description": spec.description,
        "dataset_pairs": int(dataset_rows),
        "coverage_pairs": coverage_pairs,
        "coverage_total": coverage_total,
        "coverage_rate": coverage_pairs / coverage_total if coverage_total else np.nan,
        "full_evaluated_pairs": int(len(eval_df)),
        "full_prefix_scored_pairs": int(eval_df["cos_prefix"].notna().sum()),
        "full_mean_cos_global": float(eval_df["cos_global"].mean()),
        "full_mean_cos_prefix": float(eval_df["cos_prefix"].dropna().mean()),
        "full_delta_prefix_minus_global": float(eval_df["delta_prefix_minus_global"].dropna().mean()),
        "holdout_mean_cos_global": metrics.get("holdout_mean_cos_global", np.nan),
        "holdout_mean_cos_prefix": metrics.get("holdout_mean_cos_prefix", np.nan),
        "holdout_delta_prefix_minus_global": (
            metrics.get("holdout_mean_cos_prefix", np.nan) - metrics.get("holdout_mean_cos_global", np.nan)
        ),
        "embedding_dim": config.get("dim", np.nan),
        "alpha": config.get("alpha", np.nan),
        "min_pairs": config.get("min_pairs", np.nan),
        "train_rows": config.get("train_rows", np.nan),
        "test_rows": config.get("test_rows", np.nan),
        "trained_prefixes": config.get("trained_prefixes", np.nan),
        "dropped_rows": config.get("dropped_rows", np.nan),
        "eval_csv": str(spec.eval_csv),
        "artifact": str(spec.artifact),
        "coverage_csv": str(spec.coverage_csv),
    }

    prefix_summary.insert(0, "model_key", spec.key)
    prefix_summary.insert(1, "model", spec.label)
    eval_for_common = eval_df.copy()
    eval_for_common.insert(0, "model_key", spec.key)
    eval_for_common.insert(1, "model", spec.label)
    return row, prefix_summary, eval_for_common


def build_common_pair_tables(eval_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_key = {str(df["model_key"].iloc[0]): df for df in eval_frames}
    ft = by_key["own_fasttext"].copy()
    w2v = by_key["own_word2vec_lemmas"].copy()

    keys = ["base_verb", "prefixed_verb", "prefix"]
    ft = ft.loc[:, keys + ["cos_global", "cos_prefix"]].rename(
        columns={
            "cos_global": "cos_global_own_fasttext",
            "cos_prefix": "cos_prefix_own_fasttext",
        }
    )
    w2v = w2v.loc[:, keys + ["cos_global", "cos_prefix"]].rename(
        columns={
            "cos_global": "cos_global_own_word2vec_lemmas",
            "cos_prefix": "cos_prefix_own_word2vec_lemmas",
        }
    )
    common = ft.merge(w2v, on=keys, how="inner")
    common["word2vec_minus_fasttext_global_cos"] = (
        common["cos_global_own_word2vec_lemmas"] - common["cos_global_own_fasttext"]
    )
    common["word2vec_minus_fasttext_prefix_cos"] = (
        common["cos_prefix_own_word2vec_lemmas"] - common["cos_prefix_own_fasttext"]
    )

    common_overall = pd.DataFrame(
        [
            {
                "model_key": "own_fasttext",
                "model": "Собственный FastText",
                "common_pairs": len(common),
                "common_mean_cos_global": common["cos_global_own_fasttext"].mean(),
                "common_mean_cos_prefix": common["cos_prefix_own_fasttext"].dropna().mean(),
                "common_prefix_scored_pairs": int(common["cos_prefix_own_fasttext"].notna().sum()),
            },
            {
                "model_key": "own_word2vec_lemmas",
                "model": "Собственный Word2Vec по леммам",
                "common_pairs": len(common),
                "common_mean_cos_global": common["cos_global_own_word2vec_lemmas"].mean(),
                "common_mean_cos_prefix": common["cos_prefix_own_word2vec_lemmas"].dropna().mean(),
                "common_prefix_scored_pairs": int(common["cos_prefix_own_word2vec_lemmas"].notna().sum()),
            },
        ]
    )
    return common, common_overall


def build_comparison(prefix_summaries: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "prefix",
        "model_key",
        "pairs_with_both_vectors",
        "coverage",
        "evaluated_pairs",
        "mean_cos_global",
        "mean_cos_prefix",
        "mean_delta_prefix_minus_global",
    ]
    compact = prefix_summaries.loc[:, keep].copy()
    wide = compact.pivot(index="prefix", columns="model_key")
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()

    if "mean_cos_prefix_own_word2vec_lemmas" in wide and "mean_cos_prefix_own_fasttext" in wide:
        wide["word2vec_minus_fasttext_prefix_cos"] = (
            wide["mean_cos_prefix_own_word2vec_lemmas"] - wide["mean_cos_prefix_own_fasttext"]
        )
    if "coverage_own_word2vec_lemmas" in wide and "coverage_own_fasttext" in wide:
        wide["word2vec_minus_fasttext_coverage"] = wide["coverage_own_word2vec_lemmas"] - wide["coverage_own_fasttext"]

    sort_col = "pairs_with_both_vectors_own_word2vec_lemmas"
    if sort_col in wide:
        wide = wide.sort_values([sort_col, "prefix"], ascending=[False, True])
    return wide


def fmt(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None, digits: int = 4) -> str:
    view = df.loc[:, columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)

    rendered = view.copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(lambda value: fmt(value, digits=digits))

    header = "| " + " | ".join(rendered.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in rendered.to_numpy()]
    return "\n".join([header, separator] + rows)


def save_plots(overall: pd.DataFrame, prefix_comparison: pd.DataFrame, output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return []

    paths: list[Path] = []

    overall_plot = overall.copy()
    x = np.arange(len(overall_plot))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, overall_plot["full_mean_cos_global"], width, label="Full global")
    ax.bar(x - 0.5 * width, overall_plot["full_mean_cos_prefix"], width, label="Full prefix")
    ax.bar(x + 0.5 * width, overall_plot["holdout_mean_cos_global"], width, label="Holdout global")
    ax.bar(x + 1.5 * width, overall_plot["holdout_mean_cos_prefix"], width, label="Holdout prefix")
    ax.set_xticks(x)
    ax.set_xticklabels(overall_plot["model"], rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean cosine")
    ax.set_title("Overall matrix quality")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "overall_cosine_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(overall_plot["model"], overall_plot["coverage_rate"], color=["#5778a4", "#59a14f"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of dataset pairs")
    ax.set_title("Embedding coverage")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "overall_coverage_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    plot_df = prefix_comparison.sort_values("pairs_with_both_vectors_own_word2vec_lemmas", ascending=False).copy()
    x = np.arange(len(plot_df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, plot_df["mean_cos_prefix_own_fasttext"], width, label="FastText")
    ax.bar(x + width / 2, plot_df["mean_cos_prefix_own_word2vec_lemmas"], width, label="Word2Vec lemmas")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["prefix"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean cosine, prefix matrix")
    ax.set_title("Prefix-matrix cosine by prefix")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "prefix_cosine_fasttext_vs_word2vec.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if "word2vec_minus_fasttext_prefix_cos" in plot_df:
        delta_df = plot_df.sort_values("word2vec_minus_fasttext_prefix_cos", ascending=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(delta_df["prefix"], delta_df["word2vec_minus_fasttext_prefix_cos"], color="#59a14f")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Mean cosine difference")
        ax.set_title("Word2Vec lemma advantage over FastText by prefix")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = output_dir / "prefix_word2vec_minus_fasttext.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def write_report(
    path: Path,
    overall: pd.DataFrame,
    prefix_comparison: pd.DataFrame,
    prefix_summaries: pd.DataFrame,
    common_pairs: pd.DataFrame,
    common_overall: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    overall_view = overall.rename(
        columns={
            "model": "модель",
            "coverage_pairs": "пар с векторами",
            "dataset_pairs": "всего пар",
            "coverage_rate": "покрытие",
            "full_mean_cos_global": "full global",
            "full_mean_cos_prefix": "full prefix",
            "holdout_mean_cos_global": "holdout global",
            "holdout_mean_cos_prefix": "holdout prefix",
            "trained_prefixes": "матриц приставок",
        }
    )

    prefix_view = prefix_comparison.rename(
        columns={
            "prefix": "приставка",
            "pairs_with_both_vectors_own_fasttext": "FastText пар",
            "coverage_own_fasttext": "FastText покрытие",
            "mean_cos_prefix_own_fasttext": "FastText prefix cos",
            "pairs_with_both_vectors_own_word2vec_lemmas": "W2V пар",
            "coverage_own_word2vec_lemmas": "W2V покрытие",
            "mean_cos_prefix_own_word2vec_lemmas": "W2V prefix cos",
            "word2vec_minus_fasttext_prefix_cos": "W2V - FastText",
        }
    )
    common_view = common_overall.rename(
        columns={
            "model": "модель",
            "common_pairs": "общих пар",
            "common_prefix_scored_pairs": "пар с prefix",
            "common_mean_cos_global": "common global",
            "common_mean_cos_prefix": "common prefix",
        }
    )

    best_word2vec = (
        prefix_summaries[prefix_summaries["model_key"] == "own_word2vec_lemmas"]
        .dropna(subset=["mean_cos_prefix"])
        .sort_values("mean_cos_prefix", ascending=False)
    )
    best_fasttext = (
        prefix_summaries[prefix_summaries["model_key"] == "own_fasttext"]
        .dropna(subset=["mean_cos_prefix"])
        .sort_values("mean_cos_prefix", ascending=False)
    )

    ft = overall[overall["model_key"] == "own_fasttext"].iloc[0]
    w2v = overall[overall["model_key"] == "own_word2vec_lemmas"].iloc[0]
    common_ft = common_overall[common_overall["model_key"] == "own_fasttext"].iloc[0]
    common_w2v = common_overall[common_overall["model_key"] == "own_word2vec_lemmas"].iloc[0]
    full_gain = w2v["full_mean_cos_prefix"] - ft["full_mean_cos_prefix"]
    holdout_gain = w2v["holdout_mean_cos_prefix"] - ft["holdout_mean_cos_prefix"]
    common_gain = common_w2v["common_mean_cos_prefix"] - common_ft["common_mean_cos_prefix"]
    coverage_gain = w2v["coverage_pairs"] - ft["coverage_pairs"]

    lines = [
        "# Сравнение собственных эмбеддингов",
        "",
        "## Краткий вывод",
        "",
        (
            "Лемматизированная модель Word2Vec дает более высокие значения косинусной близости "
            "для задачи линейного моделирования приставочных преобразований, чем собственная "
            "FastText-модель. На полной выборке среднее качество приставочных матриц выше на "
            f"{full_gain:.4f}, а на отложенной выборке разница составляет {holdout_gain:.4f}. "
            f"На общей части выборки из {len(common_pairs)} пар разница составляет {common_gain:.4f}. "
            f"Покрытие Word2Vec по леммам больше на {int(coverage_gain)} пар."
        ),
        "",
        "Важно: full-оценка включает пары, участвовавшие в обучении матриц, поэтому для вывода о "
        "переносимости результата надежнее смотреть на holdout. Full-оценка полезна как описание "
        "того, насколько хорошо модель восстанавливает структуру всего доступного набора пар.",
        "",
        "## Сводная таблица",
        "",
        markdown_table(
            overall_view,
            [
                "модель",
                "пар с векторами",
                "всего пар",
                "покрытие",
                "full global",
                "full prefix",
                "holdout global",
                "holdout prefix",
                "матриц приставок",
            ],
        ),
        "",
        "## Оценка на общей части выборки",
        "",
        (
            "Так как модели покрывают не полностью одинаковые пары, отдельно посчитана оценка "
            "на пересечении: ниже обе модели сравниваются на одном и том же наборе глагольных пар."
        ),
        "",
        markdown_table(
            common_view,
            ["модель", "общих пар", "пар с prefix", "common global", "common prefix"],
        ),
        "",
        "## Интерпретация",
        "",
        (
            "FastText использует символьные n-граммы, поэтому теоретически лучше переносится на "
            "редкие или незнакомые формы. Однако в этой задаче пары заданы как леммы глаголов, "
            "и Word2Vec, обученный именно на последовательностях лемм, оказывается лучше согласован "
            "с форматом датасета."
        ),
        "",
        (
            "Разница между глобальной и приставочными матрицами у Word2Vec невелика: на полной "
            "выборке приставочные матрицы немного лучше, а на holdout глобальная матрица чуть выше. "
            "Это можно интерпретировать осторожно: лемматизированное пространство хорошо кодирует "
            "общее направление приставочного преобразования, но отдельные приставки не всегда дают "
            "большой выигрыш на отложенных примерах."
        ),
        "",
        "## По приставкам",
        "",
        markdown_table(
            prefix_view,
            [
                "приставка",
                "FastText пар",
                "FastText покрытие",
                "FastText prefix cos",
                "W2V пар",
                "W2V покрытие",
                "W2V prefix cos",
                "W2V - FastText",
            ],
            max_rows=None,
        ),
        "",
        "## Лучшие приставки",
        "",
        "### Word2Vec по леммам",
        "",
        markdown_table(
            best_word2vec.rename(
                columns={
                    "prefix": "приставка",
                    "evaluated_pairs": "пар",
                    "mean_cos_global": "global cos",
                    "mean_cos_prefix": "prefix cos",
                    "mean_delta_prefix_minus_global": "prefix - global",
                }
            ),
            ["приставка", "пар", "global cos", "prefix cos", "prefix - global"],
            max_rows=10,
        ),
        "",
        "### FastText",
        "",
        markdown_table(
            best_fasttext.rename(
                columns={
                    "prefix": "приставка",
                    "evaluated_pairs": "пар",
                    "mean_cos_global": "global cos",
                    "mean_cos_prefix": "prefix cos",
                    "mean_delta_prefix_minus_global": "prefix - global",
                }
            ),
            ["приставка", "пар", "global cos", "prefix cos", "prefix - global"],
            max_rows=10,
        ),
        "",
        "## Графики",
        "",
    ]

    for plot_path in plot_paths:
        lines.append(f"- `{plot_path.name}`")

    lines.extend(
        [
            "",
            "## Файлы",
            "",
            "- `overall_comparison.csv` — сводные метрики по двум моделям.",
            "- `common_pair_overall.csv` — сводные метрики на общей части выборки.",
            "- `common_pair_comparison.csv` — построчное сравнение на общей части выборки.",
            "- `prefix_comparison.csv` — сравнение по приставкам.",
            "- `prefix_summary_long.csv` — подробная таблица по модели и приставке.",
            "",
            "## Формулировка для диплома",
            "",
            (
                "В работе были сопоставлены две модели собственных сербских эмбеддингов: FastText, "
                "обученная на локальном корпусе с учетом субсловных n-грамм, и Word2Vec, обученная "
                "на лемматизированном корпусе. Для каждой модели были обучены глобальная матрица "
                "перехода от базового глагола к приставочному и отдельные матрицы для приставок. "
                f"Модель Word2Vec по леммам покрыла {int(w2v['coverage_pairs'])} из "
                f"{int(w2v['coverage_total'])} пар, тогда как FastText покрыла "
                f"{int(ft['coverage_pairs'])} пар. На полной выборке средняя косинусная близость "
                f"для приставочных матриц составила {w2v['full_mean_cos_prefix']:.4f} для Word2Vec "
                f"и {ft['full_mean_cos_prefix']:.4f} для FastText. На отложенной выборке значения "
                f"составили {w2v['holdout_mean_cos_prefix']:.4f} и "
                f"{ft['holdout_mean_cos_prefix']:.4f} соответственно. На общей части выборки из "
                f"{len(common_pairs)} пар средние значения для приставочных матриц равны "
                f"{common_w2v['common_mean_cos_prefix']:.4f} и "
                f"{common_ft['common_mean_cos_prefix']:.4f}. Полученные результаты "
                "показывают, что лемматизация корпуса и обучение репрезентаций в формате, близком "
                "к исследуемому датасету, повышают линейную воспроизводимость приставочных "
                "преобразований сербских глаголов."
            ),
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    require_path(dataset_path, "Dataset")
    dataset_rows = len(pd.read_csv(dataset_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    prefix_frames = []
    eval_frames = []
    for spec in DEFAULT_SPECS:
        row, prefix_summary, eval_df = summarize_model(spec, dataset_rows)
        overall_rows.append(row)
        prefix_frames.append(prefix_summary)
        eval_frames.append(eval_df)

    overall = pd.DataFrame(overall_rows)
    prefix_summaries = pd.concat(prefix_frames, ignore_index=True)
    prefix_comparison = build_comparison(prefix_summaries)
    common_pairs, common_overall = build_common_pair_tables(eval_frames)

    overall_path = output_dir / "overall_comparison.csv"
    prefix_summary_path = output_dir / "prefix_summary_long.csv"
    prefix_comparison_path = output_dir / "prefix_comparison.csv"
    common_pairs_path = output_dir / "common_pair_comparison.csv"
    common_overall_path = output_dir / "common_pair_overall.csv"
    report_path = output_dir / "own_embeddings_comparison_report.md"

    overall.to_csv(overall_path, index=False)
    prefix_summaries.to_csv(prefix_summary_path, index=False)
    prefix_comparison.to_csv(prefix_comparison_path, index=False)
    common_pairs.to_csv(common_pairs_path, index=False)
    common_overall.to_csv(common_overall_path, index=False)

    plot_paths = save_plots(overall, prefix_comparison, output_dir)
    write_report(report_path, overall, prefix_comparison, prefix_summaries, common_pairs, common_overall, plot_paths)

    print("Saved report:", report_path)
    print("Saved overall table:", overall_path)
    print("Saved common-pair overall table:", common_overall_path)
    print("Saved common-pair comparison:", common_pairs_path)
    print("Saved prefix comparison:", prefix_comparison_path)
    print("Saved long prefix summary:", prefix_summary_path)
    if plot_paths:
        print("Saved plots:")
        for path in plot_paths:
            print(" -", path)


if __name__ == "__main__":
    main()
