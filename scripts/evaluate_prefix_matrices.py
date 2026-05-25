#!/usr/bin/env python
"""Evaluate trained prefix matrices on Serbian verb-pair data."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prefix matrices.")
    parser.add_argument("--artifact", default="prefix_matrices.pkl", help="Path to artifact from training.")
    parser.add_argument(
        "--dataset",
        default="serbian_verb_final_02_02_with_ruscorpora_500.csv",
        help="CSV with columns: base_verb,prefixed_verb,prefix",
    )
    parser.add_argument("--embeddings", default="cc.sr.300.bin", help="Embeddings (.bin preferred, .vec supported).")
    parser.add_argument(
        "--split",
        choices=["holdout", "full"],
        default="holdout",
        help="holdout = use training holdout saved in artifact; full = evaluate all available pairs.",
    )
    parser.add_argument(
        "--output-csv",
        default="prefix_eval_results.csv",
        help="Where to save row-level evaluation table.",
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
        return model.wv, "fasttext_bin"

    if suffix == ".vec":
        vecs, dim = load_selected_vec(path, needed_words)
        if dim == 0:
            raise SystemExit("Could not infer vector dimensionality from .vec file.")
        return vecs, "selected_vec"

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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    args = parse_args()

    artifact_path = Path(args.artifact)
    dataset_path = Path(args.dataset)
    emb_path = Path(args.embeddings)
    output_path = Path(args.output_csv)

    if not artifact_path.exists():
        raise SystemExit(f"Artifact not found: {artifact_path}")
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    with artifact_path.open("rb") as f:
        artifact = pickle.load(f)

    W_global = artifact["W_global"]
    W_prefix = artifact["W_prefix"]

    df = pd.read_csv(dataset_path)
    needed = ["base_verb", "prefixed_verb", "prefix"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")
    df = df.loc[:, needed].dropna().copy()

    if args.split == "holdout":
        holdout = artifact.get("holdout_examples", [])
        if not holdout:
            raise SystemExit("Artifact has no holdout examples. Run with --split full.")
        eval_candidates = holdout
    else:
        eval_candidates = df.to_dict(orient="records")

    needed_words = set(str(r["base_verb"]) for r in eval_candidates) | set(str(r["prefixed_verb"]) for r in eval_candidates)
    kv, mode = load_vectors(emb_path, needed_words)

    rows: List[dict] = []
    dropped = 0

    for row in eval_candidates:
        base = str(row["base_verb"])
        pref = str(row["prefixed_verb"])
        prefix = str(row["prefix"])

        x = safe_get_vector(kv, base, mode)
        y = safe_get_vector(kv, pref, mode)
        if x is None or y is None:
            dropped += 1
            continue

        y_hat_global = x @ W_global
        cos_global = cosine(y_hat_global, y)

        if prefix in W_prefix:
            y_hat_pref = x @ W_prefix[prefix]
            cos_prefix = cosine(y_hat_pref, y)
        else:
            cos_prefix = np.nan

        rows.append(
            {
                "base_verb": base,
                "prefixed_verb": pref,
                "prefix": prefix,
                "cos_global": cos_global,
                "cos_prefix": cos_prefix,
            }
        )

    if not rows:
        raise SystemExit("No rows to evaluate after embedding lookup.")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)

    mean_global = float(out_df["cos_global"].mean())
    mean_prefix = float(out_df["cos_prefix"].dropna().mean()) if out_df["cos_prefix"].notna().any() else float("nan")

    print("Saved:", output_path)
    print("Embedding mode:", mode)
    if mode == "selected_vec":
        print("Loaded vectors for words:", len(kv), "of", len(needed_words))
    print("Rows evaluated:", len(out_df), "| dropped:", dropped)
    print("Mean cosine (global):", round(mean_global, 4) if not np.isnan(mean_global) else "nan")
    print("Mean cosine (prefix):", round(mean_prefix, 4) if not np.isnan(mean_prefix) else "nan")

    by_prefix = out_df.groupby("prefix", as_index=False).agg(
        n=("prefix", "size"),
        mean_cos_global=("cos_global", "mean"),
        mean_cos_prefix=("cos_prefix", "mean"),
    )
    print("\nPer-prefix summary:")
    print(by_prefix.sort_values("n", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
