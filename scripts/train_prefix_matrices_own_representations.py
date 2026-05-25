#!/usr/bin/env python
"""Train prefix matrices on representations learned from a verb-pair table.

This script deliberately does not use pretrained FastText/Word2Vec vectors.

It learns "own" verb representations from the user's CSV itself:
  1. Build a verb-feature matrix from prefix relations and frequencies.
  2. Compress that matrix with SVD into dense verb vectors.
  3. Learn prefix-specific matrices:

       v(prefixed_verb) ~= v(base_verb) @ W_prefix

The input CSV must contain:
  base_verb,prefixed_verb,prefix,freq_base,freq_prefixed
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path("notebooks") / "serbian_verb_final_hf_wiki_5_ruscorpora_combined_50.csv"
DEFAULT_ARTIFACT = Path("models") / "artifacts" / "prefix_matrices_own_svd.pkl"
DEFAULT_VECTORS = Path("models") / "artifacts" / "own_svd_verb_vectors.csv"
DEFAULT_EVAL = Path("models") / "artifacts" / "prefix_eval_own_svd.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train prefix matrices on own relation-based verb representations."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="User-owned verb-pair CSV.")
    parser.add_argument("--output-artifact", default=str(DEFAULT_ARTIFACT), help="Where to save pickle artifact.")
    parser.add_argument("--output-vectors", default=str(DEFAULT_VECTORS), help="Where to save learned verb vectors.")
    parser.add_argument("--output-eval", default=str(DEFAULT_EVAL), help="Where to save row-level evaluation CSV.")
    parser.add_argument("--dim", type=int, default=50, help="Dimensionality of learned verb representations.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization for prefix matrices.")
    parser.add_argument("--min-pairs", type=int, default=8, help="Minimum train pairs per prefix matrix.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Holdout ratio inside each prefix group.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for holdout split.")
    parser.add_argument(
        "--weight-mode",
        choices=["binary", "log_freq"],
        default="log_freq",
        help="How to weight relation features built from CSV rows.",
    )
    parser.add_argument(
        "--representation-source",
        choices=["full", "train"],
        default="full",
        help=(
            "full uses all rows to learn verb representations; train avoids holdout leakage, "
            "but may leave rare holdout-only verbs with weak vectors."
        ),
    )
    return parser.parse_args()


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = ["base_verb", "prefixed_verb", "prefix", "freq_base", "freq_prefixed"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")

    df = df.loc[:, required].dropna(subset=["base_verb", "prefixed_verb", "prefix"]).copy()
    for column in ["base_verb", "prefixed_verb", "prefix"]:
        df[column] = df[column].astype(str).str.strip()

    for column in ["freq_base", "freq_prefixed"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df = df[(df["base_verb"] != "") & (df["prefixed_verb"] != "") & (df["prefix"] != "")]
    df = df.reset_index(drop=True)
    if df.empty:
        raise SystemExit("No usable rows after cleaning.")
    return df


def per_prefix_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.Index, pd.Index]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []

    for _, group in df.groupby("prefix"):
        idx = group.index.to_numpy()
        n = len(idx)
        if n < 3 or test_ratio <= 0.0:
            train_idx.extend(int(i) for i in idx)
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_test = min(n_test, n - 1)
        chosen = set(int(i) for i in rng.choice(idx, size=n_test, replace=False))

        for i in idx:
            if int(i) in chosen:
                test_idx.append(int(i))
            else:
                train_idx.append(int(i))

    return pd.Index(train_idx), pd.Index(test_idx)


def all_verbs(df: pd.DataFrame) -> List[str]:
    verbs = set(df["base_verb"].astype(str)) | set(df["prefixed_verb"].astype(str))
    return sorted(verbs)


def relation_weight(freq_prefixed: float, weight_mode: str) -> float:
    if weight_mode == "binary":
        return 1.0
    return float(np.log1p(max(freq_prefixed, 0.0)))


def add_value(matrix: np.ndarray, row: int, col: int, value: float) -> None:
    matrix[row, col] += np.float32(value)


def make_feature_id(feature_to_id: Dict[str, int], feature_name: str) -> int:
    if feature_name not in feature_to_id:
        feature_to_id[feature_name] = len(feature_to_id)
    return feature_to_id[feature_name]


def collect_feature_names(df: pd.DataFrame) -> Dict[str, int]:
    feature_to_id: Dict[str, int] = {}

    for _, row in df.iterrows():
        base = str(row["base_verb"])
        prefixed = str(row["prefixed_verb"])
        prefix = str(row["prefix"])

        for feature in (
            f"out_prefix::{prefix}",
            f"in_prefix::{prefix}",
            f"out_to::{prefixed}",
            f"in_from::{base}",
            "freq_as_base",
            "freq_as_prefixed",
        ):
            make_feature_id(feature_to_id, feature)

    return feature_to_id


def build_relation_feature_matrix(
    df: pd.DataFrame,
    verbs: Iterable[str],
    weight_mode: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    verb_list = list(verbs)
    verb_to_id = {verb: i for i, verb in enumerate(verb_list)}
    feature_to_id = collect_feature_names(df)
    feature_names = [None] * len(feature_to_id)
    for feature, i in feature_to_id.items():
        feature_names[i] = feature

    matrix = np.zeros((len(verb_list), len(feature_to_id)), dtype=np.float32)

    for _, row in df.iterrows():
        base = str(row["base_verb"])
        prefixed = str(row["prefixed_verb"])
        prefix = str(row["prefix"])

        if base not in verb_to_id or prefixed not in verb_to_id:
            continue

        base_id = verb_to_id[base]
        prefixed_id = verb_to_id[prefixed]
        weight = relation_weight(float(row["freq_prefixed"]), weight_mode)
        base_freq_weight = float(np.log1p(max(float(row["freq_base"]), 0.0)))
        prefixed_freq_weight = float(np.log1p(max(float(row["freq_prefixed"]), 0.0)))

        add_value(matrix, base_id, feature_to_id[f"out_prefix::{prefix}"], weight)
        add_value(matrix, base_id, feature_to_id[f"out_to::{prefixed}"], weight)
        add_value(matrix, base_id, feature_to_id["freq_as_base"], base_freq_weight)

        add_value(matrix, prefixed_id, feature_to_id[f"in_prefix::{prefix}"], weight)
        add_value(matrix, prefixed_id, feature_to_id[f"in_from::{base}"], weight)
        add_value(matrix, prefixed_id, feature_to_id["freq_as_prefixed"], prefixed_freq_weight)

    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, row_norms, out=np.zeros_like(matrix), where=row_norms > 0)
    return matrix, verb_list, feature_names


def svd_verb_vectors(matrix: np.ndarray, dim: int) -> Tuple[np.ndarray, int, float]:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise SystemExit("Need at least two verbs and two features to learn SVD representations.")

    actual_dim = min(dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if actual_dim < 1:
        raise SystemExit("SVD dimensionality became < 1. Check the dataset.")

    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    vectors = u[:, :actual_dim] * singular_values[:actual_dim]

    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, vector_norms, out=np.zeros_like(vectors), where=vector_norms > 0)

    total_energy = float(np.sum(singular_values**2))
    kept_energy = float(np.sum(singular_values[:actual_dim] ** 2))
    explained = kept_energy / total_energy if total_energy > 0.0 else 0.0
    return vectors.astype(np.float32), actual_dim, explained


def vector_dict(verbs: List[str], vectors: np.ndarray) -> Dict[str, np.ndarray]:
    return {verb: vectors[i] for i, verb in enumerate(verbs)}


def ridge_closed_form(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    d = x.shape[1]
    xtx = x.T @ x
    xty = x.T @ y
    reg = alpha * np.eye(d, dtype=np.float64)
    return np.linalg.solve(xtx + reg, xty).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def row_vectors(df: pd.DataFrame, vectors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    keep: List[int] = []

    for i, row in df.iterrows():
        base = str(row["base_verb"])
        prefixed = str(row["prefixed_verb"])
        x = vectors.get(base)
        y = vectors.get(prefixed)
        if x is None or y is None:
            continue
        x_rows.append(x)
        y_rows.append(y)
        keep.append(int(i))

    if not keep:
        raise SystemExit("No rows have both base and prefixed vectors.")
    return np.vstack(x_rows), np.vstack(y_rows), keep


def save_vectors_csv(path: Path, verbs: List[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"dim_{i + 1:03d}" for i in range(vectors.shape[1])]
    out = pd.DataFrame(vectors, columns=columns)
    out.insert(0, "verb", verbs)
    out.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    artifact_path = Path(args.output_artifact)
    vectors_path = Path(args.output_vectors)
    eval_path = Path(args.output_eval)

    df = read_pairs(dataset_path)
    train_idx, test_idx = per_prefix_split(df, args.test_ratio, args.seed)
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    representation_df = train_df if args.representation_source == "train" else df
    verbs = all_verbs(df)
    feature_matrix, verb_list, feature_names = build_relation_feature_matrix(
        representation_df,
        verbs,
        args.weight_mode,
    )
    vectors_array, actual_dim, explained = svd_verb_vectors(feature_matrix, args.dim)
    vectors = vector_dict(verb_list, vectors_array)

    save_vectors_csv(vectors_path, verb_list, vectors_array)

    x_train, y_train, train_keep = row_vectors(train_df, vectors)
    kept_train_df = train_df.loc[train_keep].copy()

    w_global = ridge_closed_form(x_train, y_train, args.alpha)

    prefix_matrices: Dict[str, np.ndarray] = {}
    prefix_stats: List[dict] = []

    for prefix, group in kept_train_df.groupby("prefix"):
        x_prefix, y_prefix, keep = row_vectors(group, vectors)
        if len(keep) < args.min_pairs:
            continue
        prefix_matrices[str(prefix)] = ridge_closed_form(x_prefix, y_prefix, args.alpha)
        prefix_stats.append({"prefix": str(prefix), "train_pairs": int(len(keep))})

    eval_rows: List[dict] = []
    for i, row in test_df.iterrows():
        base = str(row["base_verb"])
        prefixed = str(row["prefixed_verb"])
        prefix = str(row["prefix"])
        x = vectors.get(base)
        y = vectors.get(prefixed)
        if x is None or y is None:
            continue

        y_hat_global = x @ w_global
        cos_identity = cosine(x, y)
        cos_global = cosine(y_hat_global, y)

        if prefix in prefix_matrices:
            y_hat_prefix = x @ prefix_matrices[prefix]
            cos_prefix = cosine(y_hat_prefix, y)
        else:
            cos_prefix = np.nan

        eval_rows.append(
            {
                "base_verb": base,
                "prefixed_verb": prefixed,
                "prefix": prefix,
                "cos_identity": cos_identity,
                "cos_global": cos_global,
                "cos_prefix": cos_prefix,
            }
        )

    eval_df = pd.DataFrame(eval_rows)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(eval_path, index=False)

    metrics = {
        "holdout_rows": int(len(eval_df)),
        "holdout_mean_cos_identity": float(eval_df["cos_identity"].mean()) if not eval_df.empty else float("nan"),
        "holdout_mean_cos_global": float(eval_df["cos_global"].mean()) if not eval_df.empty else float("nan"),
        "holdout_mean_cos_prefix": (
            float(eval_df["cos_prefix"].dropna().mean())
            if not eval_df.empty and eval_df["cos_prefix"].notna().any()
            else float("nan")
        ),
    }

    artifact = {
        "config": {
            "dataset": str(dataset_path),
            "representation_type": "relation_svd",
            "representation_source": args.representation_source,
            "weight_mode": args.weight_mode,
            "requested_dim": int(args.dim),
            "actual_dim": int(actual_dim),
            "svd_explained_energy": float(explained),
            "alpha": float(args.alpha),
            "min_pairs": int(args.min_pairs),
            "test_ratio": float(args.test_ratio),
            "seed": int(args.seed),
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "verbs": int(len(verb_list)),
            "features": int(len(feature_names)),
            "trained_prefixes": int(len(prefix_matrices)),
        },
        "verb_vectors": vectors,
        "feature_names": feature_names,
        "W_global": w_global,
        "W_prefix": prefix_matrices,
        "prefix_stats": prefix_stats,
        "holdout_examples": eval_rows,
        "metrics": metrics,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as f:
        pickle.dump(artifact, f)

    print("Saved artifact:", artifact_path)
    print("Saved vectors:", vectors_path)
    print("Saved evaluation:", eval_path)
    print("Dataset rows:", len(df))
    print("Train rows:", len(train_df), "| Test rows:", len(test_df))
    print("Verbs:", len(verb_list), "| Features:", len(feature_names))
    print("Representation dim:", actual_dim, "| SVD explained energy:", round(explained, 4))
    print("Trained prefix matrices:", len(prefix_matrices))
    print("Mean cosine identity:", round(metrics["holdout_mean_cos_identity"], 4))
    print("Mean cosine global:", round(metrics["holdout_mean_cos_global"], 4))
    print("Mean cosine prefix:", round(metrics["holdout_mean_cos_prefix"], 4))

    if not eval_df.empty:
        by_prefix = (
            eval_df.groupby("prefix", as_index=False)
            .agg(
                n=("prefix", "size"),
                mean_cos_identity=("cos_identity", "mean"),
                mean_cos_global=("cos_global", "mean"),
                mean_cos_prefix=("cos_prefix", "mean"),
            )
            .sort_values(["n", "prefix"], ascending=[False, True])
        )
        print()
        print("Per-prefix holdout summary:")
        print(by_prefix.to_string(index=False))


if __name__ == "__main__":
    main()
