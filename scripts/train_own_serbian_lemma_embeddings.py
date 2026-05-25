#!/usr/bin/env python
"""Train Serbian embeddings on lemma sequences only.

This script is a lemma-only companion to train_own_serbian_embeddings.py.

Default lemma sources:
  - archives/serbian (1).zip: XML corpus, only <se lang="srp">, lemmas from
    <ana lex=... lex_translit=...>
  - data/text/set.sr.conll: CoNLL-style corpus, lemma column

Optional:
  - data/text/sr_Latn.txt can be lemmatized with Stanza and cached as
    data/text/sr_Latn_lemmas.txt. This is intentionally opt-in because it can
    take a long time.

Typical Word2Vec run:
  python scripts/train_own_serbian_lemma_embeddings.py --model-type word2vec

To include sr_Latn after creating/using a lemma cache:
  python scripts/train_own_serbian_lemma_embeddings.py --lemmatize-raw-texts
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterator, List, Sequence

from train_own_serbian_embeddings import (
    DEFAULT_PAIR_DATASET,
    configure_logging,
    coverage_report,
    iter_zip_xml,
    normalize_token,
    save_outputs,
    tokenize_text,
    train_model,
)


DEFAULT_ZIPS = [Path("archives") / "serbian (1).zip"]
DEFAULT_CONLLS = [Path("data") / "text" / "set.sr.conll"]
DEFAULT_RAW_TEXTS = [Path("data") / "text" / "sr_Latn.txt"]
DEFAULT_OUTPUT_PREFIX = Path("models") / "embeddings" / "own_serbian_word2vec_lemmas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Serbian embeddings on lemmas only.")
    parser.add_argument(
        "--zips",
        nargs="*",
        default=[str(path) for path in DEFAULT_ZIPS],
        help="Zip XML corpora with Serbian lemma markup.",
    )
    parser.add_argument(
        "--conlls",
        nargs="*",
        default=[str(path) for path in DEFAULT_CONLLS],
        help="CoNLL/CoNLL-U files. Lemma is read from column 3.",
    )
    parser.add_argument(
        "--lemma-texts",
        nargs="*",
        default=[],
        help="Already lemmatized plain-text files, one sentence per line.",
    )
    parser.add_argument(
        "--raw-texts",
        nargs="*",
        default=[str(path) for path in DEFAULT_RAW_TEXTS],
        help="Raw text files to lemmatize with Stanza when --lemmatize-raw-texts is set.",
    )
    parser.add_argument(
        "--lemmatize-raw-texts",
        action="store_true",
        help="Create/use lemma caches for --raw-texts via Stanza. This can be slow.",
    )
    parser.add_argument(
        "--lemma-cache-dir",
        default=str(Path("data") / "text"),
        help="Where raw-text lemma caches are stored.",
    )
    parser.add_argument(
        "--pair-dataset",
        default=str(DEFAULT_PAIR_DATASET),
        help="CSV with base_verb,prefixed_verb,prefix for coverage diagnostics.",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(DEFAULT_OUTPUT_PREFIX),
        help="Output path prefix, without extension.",
    )
    parser.add_argument("--model-type", choices=["word2vec", "fasttext"], default="word2vec")
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sg", type=int, choices=[0, 1], default=1, help="1 = skip-gram, 0 = CBOW.")
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--sample", type=float, default=1e-4)
    parser.add_argument("--bucket", type=int, default=1000000, help="FastText buckets; ignored for Word2Vec.")
    parser.add_argument("--min-n", type=int, default=3, help="FastText minimum char n-gram length.")
    parser.add_argument("--max-n", type=int, default=6, help="FastText maximum char n-gram length.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=0,
        help="Debug limit for training corpus. 0 means use all available lemma sentences.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Log corpus-reading progress every N yielded sentences. Use 0 to disable.",
    )
    parser.add_argument(
        "--raw-max-lines",
        type=int,
        default=0,
        help="Debug limit for raw text lemmatization. 0 means all lines.",
    )
    parser.add_argument("--stanza-lang", default="sr", help="Stanza language code for Serbian.")
    parser.add_argument(
        "--stanza-processors",
        default="tokenize,pos,lemma",
        help="Stanza processors used for raw text lemmatization.",
    )
    return parser.parse_args()


def iter_lemma_text(path: Path) -> Iterator[List[str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = [normalize_token(token) for token in line.split()]
            tokens = [token for token in tokens if token]
            if tokens:
                yield tokens


def iter_conll_lemmas(path: Path) -> Iterator[List[str]]:
    sentence: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if sentence:
                    yield sentence
                    sentence = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            token_id = parts[0]
            if "-" in token_id or "." in token_id:
                continue

            lemma = normalize_token(parts[2])
            if lemma and lemma != "_":
                sentence.append(lemma)

    if sentence:
        yield sentence


def cache_path_for_raw_text(raw_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{raw_path.stem}_lemmas.txt"


def lemmatize_raw_text_with_stanza(
    raw_path: Path,
    cache_path: Path,
    stanza_lang: str,
    stanza_processors: str,
    raw_max_lines: int,
    progress_every: int,
) -> None:
    try:
        import stanza
    except ImportError as exc:
        raise SystemExit(
            "Stanza is required for raw-text lemmatization. "
            "Use existing lemma sources, or run in an environment with stanza installed."
        ) from exc

    if not raw_path.exists():
        raise SystemExit(f"Raw text not found: {raw_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading Stanza pipeline: lang=%s processors=%s", stanza_lang, stanza_processors)
    try:
        nlp = stanza.Pipeline(
            stanza_lang,
            processors=stanza_processors,
            use_gpu=False,
            verbose=False,
        )
    except Exception as exc:
        raise SystemExit(
            "Could not initialize Stanza Serbian pipeline. "
            "If models are missing, download them once with stanza.download('sr')."
        ) from exc

    started_at = time.perf_counter()
    lines_seen = 0
    sentences_written = 0
    tokens_written = 0

    logging.info("Lemmatizing raw text: %s", raw_path)
    with raw_path.open("r", encoding="utf-8", errors="ignore") as src, cache_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line in src:
            lines_seen += 1
            text = line.strip()
            if not text:
                continue

            doc = nlp(text)
            for sentence in doc.sentences:
                lemmas = []
                for word in sentence.words:
                    lemma = normalize_token(word.lemma or word.text)
                    if lemma:
                        lemmas.append(lemma)
                if lemmas:
                    out.write(" ".join(lemmas) + "\n")
                    sentences_written += 1
                    tokens_written += len(lemmas)

            if progress_every and lines_seen % progress_every == 0:
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                logging.info(
                    "Raw lemmatization progress: %s lines, %s sentences, %s lemmas, %.1f lines/sec.",
                    f"{lines_seen:,}",
                    f"{sentences_written:,}",
                    f"{tokens_written:,}",
                    lines_seen / elapsed,
                )

            if raw_max_lines and lines_seen >= raw_max_lines:
                break

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    logging.info(
        "Raw lemmatization finished: %s lines, %s sentences, %s lemmas in %.1f sec. Cache: %s",
        f"{lines_seen:,}",
        f"{sentences_written:,}",
        f"{tokens_written:,}",
        elapsed,
        cache_path,
    )


def ensure_raw_lemma_caches(args: argparse.Namespace) -> list[Path]:
    cache_dir = Path(args.lemma_cache_dir)
    caches: list[Path] = []

    for raw_name in args.raw_texts:
        raw_path = Path(raw_name)
        cache_path = cache_path_for_raw_text(raw_path, cache_dir)

        if cache_path.exists():
            logging.info("Using existing raw-text lemma cache: %s", cache_path)
            caches.append(cache_path)
            continue

        if args.lemmatize_raw_texts:
            lemmatize_raw_text_with_stanza(
                raw_path=raw_path,
                cache_path=cache_path,
                stanza_lang=args.stanza_lang,
                stanza_processors=args.stanza_processors,
                raw_max_lines=args.raw_max_lines,
                progress_every=args.progress_every,
            )
            caches.append(cache_path)
        else:
            logging.info(
                "No lemma cache for %s. Skipping raw text. "
                "Run with --lemmatize-raw-texts to create %s.",
                raw_path,
                cache_path,
            )

    return caches


class SerbianLemmaSentenceCorpus:
    """Re-iterable lemma corpus for gensim."""

    def __init__(
        self,
        zip_paths: Sequence[Path],
        conll_paths: Sequence[Path],
        lemma_text_paths: Sequence[Path],
        max_sentences: int = 0,
        progress_every: int = 50000,
    ) -> None:
        self.zip_paths = list(zip_paths)
        self.conll_paths = list(conll_paths)
        self.lemma_text_paths = list(lemma_text_paths)
        self.max_sentences = max_sentences
        self.progress_every = progress_every

    def __iter__(self) -> Iterator[List[str]]:
        yielded = 0
        tokens_seen = 0
        started_at = time.perf_counter()
        logging.info("Lemma corpus pass started.")

        for path in self.lemma_text_paths:
            if not path.exists():
                logging.warning("Lemma text not found: %s", path)
                continue
            logging.info("Reading lemma text: %s", path)
            for tokens in iter_lemma_text(path):
                yielded, tokens_seen = self._yield_progress(yielded, tokens_seen, tokens, started_at)
                yield tokens
                if self.max_sentences and yielded >= self.max_sentences:
                    self._log_finished(yielded, tokens_seen, started_at, limited=True)
                    return

        for path in self.conll_paths:
            if not path.exists():
                logging.warning("CoNLL file not found: %s", path)
                continue
            logging.info("Reading CoNLL lemmas: %s", path)
            for tokens in iter_conll_lemmas(path):
                yielded, tokens_seen = self._yield_progress(yielded, tokens_seen, tokens, started_at)
                yield tokens
                if self.max_sentences and yielded >= self.max_sentences:
                    self._log_finished(yielded, tokens_seen, started_at, limited=True)
                    return

        for path in self.zip_paths:
            if not path.exists():
                logging.warning("Zip file not found: %s", path)
                continue
            logging.info("Reading XML lemmas from zip: %s", path)
            for tokens in iter_zip_xml(path, xml_mode="lemma"):
                yielded, tokens_seen = self._yield_progress(yielded, tokens_seen, tokens, started_at)
                yield tokens
                if self.max_sentences and yielded >= self.max_sentences:
                    self._log_finished(yielded, tokens_seen, started_at, limited=True)
                    return

        self._log_finished(yielded, tokens_seen, started_at, limited=False)

    def _yield_progress(
        self,
        yielded: int,
        tokens_seen: int,
        tokens: List[str],
        started_at: float,
    ) -> tuple[int, int]:
        yielded += 1
        tokens_seen += len(tokens)
        if self.progress_every and yielded % self.progress_every == 0:
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            logging.info(
                "Lemma corpus progress: %s sentences, %s lemmas, %.1f sentences/sec.",
                f"{yielded:,}",
                f"{tokens_seen:,}",
                yielded / elapsed,
            )
        return yielded, tokens_seen

    def _log_finished(self, yielded: int, tokens_seen: int, started_at: float, limited: bool) -> None:
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        suffix = " (stopped by --max-sentences)" if limited else ""
        logging.info(
            "Lemma corpus pass finished%s: %s sentences, %s lemmas in %.1f sec.",
            suffix,
            f"{yielded:,}",
            f"{tokens_seen:,}",
            elapsed,
        )


def main() -> None:
    configure_logging()
    args = parse_args()

    raw_caches = ensure_raw_lemma_caches(args)
    lemma_text_paths = [Path(path) for path in args.lemma_texts] + raw_caches

    corpus = SerbianLemmaSentenceCorpus(
        zip_paths=[Path(path) for path in args.zips],
        conll_paths=[Path(path) for path in args.conlls],
        lemma_text_paths=lemma_text_paths,
        max_sentences=args.max_sentences,
        progress_every=args.progress_every,
    )

    output_prefix = Path(args.output_prefix)

    print("Model type:", args.model_type)
    print("Lemma texts:", lemma_text_paths)
    print("CoNLL files:", [Path(path) for path in args.conlls])
    print("Zip files:", [Path(path) for path in args.zips])
    print("Output prefix:", output_prefix)
    print("Progress every sentences:", args.progress_every)
    if args.max_sentences:
        print("Debug max sentences:", args.max_sentences)

    model = train_model(args, corpus)
    save_outputs(model, output_prefix)
    coverage_report(model, Path(args.pair_dataset), output_prefix)


if __name__ == "__main__":
    main()
