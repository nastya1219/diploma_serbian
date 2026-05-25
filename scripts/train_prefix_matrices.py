#!/usr/bin/env python
"""Train prefix-specific lexical-function matrices for Serbian verbs.

Model idea:
  v(prefixed_verb) ~= v(base_verb) @ W_prefix

Where W_prefix is a d x d matrix learned separately for each prefix.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prefix matrices from verb pairs.")
    parser.add_argument(
        "--dataset",
        default="serbian_verb_final_02_02_with_ruscorpora_500.csv",
        help="CSV with columns: base_verb,prefixed_verb,prefix",
    )
    parser.add_argument(
        "--embeddings",
        default="cc.sr.300.bin",
        help="FastText embeddings file (.bin preferred, .vec supported).",
    )
    parser.add_argument(
        "--output",
        default="prefix_matrices.pkl",
        help="Output pickle path for trained matrices/artifacts.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=8,
        help="Minimum samples per prefix to train a dedicated matrix.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="L2 regularization strength for ridge closed-form.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio per prefix for evaluation in this script.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for holdout split.",
    )
    return parser.parse_args()


def load_selected_vec(path: Path, needed_words: Set[str]) -> Tuple[Dict[str, np.ndarray], int]:
    vectors: Dict[str, np.ndarray] = {}
    dim = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        first_parts = first.split()
        has_header = len(first_parts) == 2 and first_parts[0].isdigit() and first_parts[1].isdigit()
        if has_header:
            dim = int(first_parts[1])
        else:
            line = first
            if line:
                word, sep, rest = line.partition(" ")
                if sep and word in needed_words:
                    vec = np.fromstring(rest, sep=" ", dtype=np.float32)
                    if vec.size > 0:
                        vectors[word] = vec
                        dim = int(vec.size)

        for line in f:
            word, sep, rest = line.partition(" ")
            if not sep or word not in needed_words:
                continue
            vec = np.fromstring(rest, sep=" ", dtype=np.float32)
            if vec.size == 0:
                continue
            if dim == 0:
                dim = int(vec.size)
            if vec.size != dim:
                continue
            vectors[word] = vec

    return vectors, dim


def load_vectors(path: Path, needed_words: Set[str]):
    try:
        from gensim.models.fasttext import load_facebook_model
    except ImportError as exc:
        raise SystemExit("gensim is required. Install with: pip install gensim") from exc

    if not path.exists():
        raise SystemExit(f"Embeddings not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".bin":
        model = load_facebook_model(str(path))
        return model.wv, "fasttext_bin", model.wv.vector_size

    if suffix == ".vec":
        vecs, dim = load_selected_vec(path, needed_words)
        if dim == 0:
            raise SystemExit("Could not infer vector dimensionality from .vec file.")
        return vecs, "selected_vec", dim

    raise SystemExit("Unsupported embeddings format. Use .bin or .vec")


def safe_get_vector(kv, word: str, mode: str) -> np.ndarray | None:
    try:
        if mode == "fasttext_bin":
            return np.asarray(kv.get_vector(word), dtype=np.float32)
        if mode == "selected_vec":
            vec = kv.get(word)
            return None if vec is None else np.asarray(vec, dtype=np.float32)
        return None
    except Exception:
        return None


def ridge_closed_form(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    d = X.shape[1]
    xtx = X.T @ X
    reg = alpha * np.eye(d, dtype=np.float64)
    xty = X.T @ Y
    W = np.linalg.solve(xtx + reg, xty)
    return W.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def per_prefix_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.Index, pd.Index]:
    rng = np.random.default_rng(seed)
    test_idx: List[int] = []
    train_idx: List[int] = []

    for _, group in df.groupby("prefix"):
        idx = group.index.to_numpy()
        n = len(idx)
        if n < 3:
            train_idx.extend(idx.tolist())
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_test = min(n_test, n - 1)
        chosen = rng.choice(idx, size=n_test, replace=False)
        test_set = set(chosen.tolist())

        for i in idx:
            if i in test_set:
                test_idx.append(int(i))
            else:
                train_idx.append(int(i))

    return pd.Index(train_idx), pd.Index(test_idx)


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    emb_path = Path(args.embeddings)
    out_path = Path(args.output)

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    needed = ["base_verb", "prefixed_verb", "prefix"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")

    df = df.loc[:, needed].dropna().copy()
    df["base_verb"] = df["base_verb"].astype(str)
    df["prefixed_verb"] = df["prefixed_verb"].astype(str)
    df["prefix"] = df["prefix"].astype(str)

    needed_words = set(df["base_verb"].tolist()) | set(df["prefixed_verb"].tolist())
    kv, mode, vec_dim = load_vectors(emb_path, needed_words)

    base_vecs = []
    pref_vecs = []
    keep_rows = []
    dropped = 0

    for idx, row in df.iterrows():
        v_base = safe_get_vector(kv, row["base_verb"], mode)
        v_pref = safe_get_vector(kv, row["prefixed_verb"], mode)
        if v_base is None or v_pref is None:
            dropped += 1
            continue
        keep_rows.append(idx)
        base_vecs.append(v_base)
        pref_vecs.append(v_pref)

    if not keep_rows:
        raise SystemExit("No valid rows left after embedding lookup.")

    df = df.loc[keep_rows].copy().reset_index(drop=True)
    X_all = np.vstack(base_vecs)
    Y_all = np.vstack(pref_vecs)

    d = X_all.shape[1]
    if d != vec_dim:
        print(f"Warning: inferred dim mismatch ({d} vs {vec_dim}), using {d}.")

    train_idx, test_idx = per_prefix_split(df, args.test_ratio, args.seed)

    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    X_train = X_all[train_idx.to_numpy()]
    Y_train = Y_all[train_idx.to_numpy()]

    W_global = ridge_closed_form(X_train, Y_train, args.alpha)

    matrices: Dict[str, np.ndarray] = {}
    prefix_stats = []

    for prefix, group in train_df.groupby("prefix"):
        g_idx = group.index.to_numpy()
        n = len(g_idx)
        if n < args.min_pairs:
            continue
        Xp = X_all[g_idx]
        Yp = Y_all[g_idx]
        matrices[prefix] = ridge_closed_form(Xp, Yp, args.alpha)
        prefix_stats.append({"prefix": prefix, "train_pairs": int(n)})

    if test_df.empty:
        print("Warning: empty holdout set. Evaluation metrics may be missing.")

    eval_rows = []
    for i in test_idx.to_numpy():
        x = X_all[i]
        y = Y_all[i]
        prefix = df.at[i, "prefix"]

        y_hat_global = x @ W_global
        cos_global = cosine(y_hat_global, y)

        if prefix in matrices:
            y_hat_pref = x @ matrices[prefix]
            cos_pref = cosine(y_hat_pref, y)
        else:
            cos_pref = None

        eval_rows.append(
            {
                "base_verb": df.at[i, "base_verb"],
                "prefixed_verb": df.at[i, "prefixed_verb"],
                "prefix": prefix,
                "cos_global": cos_global,
                "cos_prefix": cos_pref,
            }
        )

    eval_df = pd.DataFrame(eval_rows)
    if not eval_df.empty:
        overall_global = float(eval_df["cos_global"].mean())
        overall_prefix = float(eval_df["cos_prefix"].dropna().mean()) if eval_df["cos_prefix"].notna().any() else float("nan")
    else:
        overall_global = float("nan")
        overall_prefix = float("nan")

    artifact = {
        "config": {
            "dataset": str(dataset_path),
            "embeddings": str(emb_path),
            "embedding_mode": mode,
            "dim": int(d),
            "alpha": float(args.alpha),
            "min_pairs": int(args.min_pairs),
            "test_ratio": float(args.test_ratio),
            "seed": int(args.seed),
            "dropped_rows": int(dropped),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "trained_prefixes": int(len(matrices)),
            "loaded_word_vectors": int(len(kv)) if mode == "selected_vec" else None,
        },
        "W_global": W_global,
        "W_prefix": matrices,
        "prefix_stats": prefix_stats,
        "holdout_examples": eval_rows,
        "metrics": {
            "holdout_mean_cos_global": overall_global,
            "holdout_mean_cos_prefix": overall_prefix,
        },
    }

    with out_path.open("wb") as f:
        pickle.dump(artifact, f)

    print("Saved:", out_path)
    print("Embedding mode:", mode)
    if mode == "selected_vec":
        print("Loaded vectors for words:", len(kv), "of", len(needed_words))
    print("Rows kept:", len(df), "| dropped:", dropped)
    print("Train:", len(train_idx), "| Test:", len(test_idx))
    print("Trained prefix matrices:", len(matrices))
    print("Holdout mean cosine (global):", round(overall_global, 4) if not np.isnan(overall_global) else "nan")
    print("Holdout mean cosine (prefix):", round(overall_prefix, 4) if not np.isnan(overall_prefix) else "nan")


if __name__ == "__main__":
    main()
