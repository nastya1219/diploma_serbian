#!/usr/bin/env python
"""Train Serbian word embeddings on local text/XML corpora.

The default setup uses:
  - data/text/sr_Latn.txt
  - archives/serbian (1).zip

The zip is treated as an XML corpus. Only <se lang="srp"> segments are used.
For XML, the default mode extracts lemmas from <ana lex=... lex_translit=...>
because the prefix-pair dataset uses lemma-like forms such as biti, dobiti,
pisati, napisati.

Outputs:
  - .model: full gensim model, including FastText subword parameters
  - .kv: keyed vectors
  - .vec: word2vec text vectors for words seen in training vocabulary
  - coverage CSV for the prefix-pair dataset
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence
from xml.etree import ElementTree as ET

import pandas as pd


DEFAULT_TEXTS = [Path("data") / "text" / "sr_Latn.txt"]
DEFAULT_ZIPS = [Path("archives") / "serbian (1).zip"]
DEFAULT_PAIR_DATASET = Path("notebooks") / "serbian_verb_final_hf_wiki_5_ruscorpora_combined_50.csv"
DEFAULT_OUTPUT_PREFIX = Path("models") / "embeddings" / "own_serbian_fasttext"

TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)?", re.UNICODE)

CYR_TO_LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ђ": "đ",
    "е": "e",
    "ж": "ž",
    "з": "z",
    "и": "i",
    "ј": "j",
    "к": "k",
    "л": "l",
    "љ": "lj",
    "м": "m",
    "н": "n",
    "њ": "nj",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ћ": "ć",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "č",
    "џ": "dž",
    "ш": "š",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Serbian FastText/Word2Vec on local corpora.")
    parser.add_argument(
        "--texts",
        nargs="*",
        default=[str(path) for path in DEFAULT_TEXTS],
        help="Plain-text files. Each line is treated as a sentence/document fragment.",
    )
    parser.add_argument(
        "--zips",
        nargs="*",
        default=[str(path) for path in DEFAULT_ZIPS],
        help="Zip archives with XML files containing <se lang='srp'> segments.",
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
    parser.add_argument("--model-type", choices=["fasttext", "word2vec"], default="fasttext")
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--sg", type=int, choices=[0, 1], default=1, help="1 = skip-gram, 0 = CBOW.")
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--sample", type=float, default=1e-4)
    parser.add_argument("--bucket", type=int, default=1000000, help="FastText hash buckets for char n-grams.")
    parser.add_argument("--min-n", type=int, default=3, help="FastText minimum char n-gram length.")
    parser.add_argument("--max-n", type=int, default=6, help="FastText maximum char n-gram length.")
    parser.add_argument(
        "--xml-mode",
        choices=["lemma", "raw"],
        default="lemma",
        help="lemma extracts lex/lex_translit from XML analyses; raw extracts surface tokens.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=0,
        help="Debug limit. 0 means use the whole corpus.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Log corpus-reading progress every N yielded sentences. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )


def normalize_token(token: str) -> str:
    token = token.strip().lower()
    if not token:
        return ""
    token = transliterate_serbian_cyrillic(token)
    token = token.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    return token


def transliterate_serbian_cyrillic(text: str) -> str:
    out: List[str] = []
    for ch in text:
        lower = ch.lower()
        if lower in CYR_TO_LAT:
            latin = CYR_TO_LAT[lower]
            out.append(latin.capitalize() if ch.isupper() else latin)
        else:
            out.append(ch)
    return "".join(out)


def tokenize_text(text: str) -> List[str]:
    return [tok for tok in (normalize_token(match.group(0)) for match in TOKEN_RE.finditer(text)) if tok]


def latin_score(text: str) -> int:
    latin_letters = set("abcdefghijklmnopqrstuvwxyzčćđšž")
    return sum(1 for ch in text.lower() if ch in latin_letters)


def choose_latin_candidate(candidates: Sequence[str]) -> str:
    cleaned = [candidate.strip() for candidate in candidates if candidate and candidate.strip()]
    if not cleaned:
        return ""
    return max(cleaned, key=latin_score)


def xml_word_to_lemma_token(word_elem: ET.Element) -> str:
    candidates: List[str] = []
    for ana in word_elem.findall("ana"):
        candidates.append(ana.attrib.get("lex_translit", ""))
        candidates.append(ana.attrib.get("lex", ""))
    candidates.append(word_elem.attrib.get("translit", ""))
    candidates.append(word_elem.text or "")
    return normalize_token(choose_latin_candidate(candidates))


def xml_sentence_tokens(se_elem: ET.Element, xml_mode: str) -> List[str]:
    if xml_mode == "lemma":
        tokens = [xml_word_to_lemma_token(w) for w in se_elem.findall(".//w")]
        tokens = [token for token in tokens if token]
        if tokens:
            return tokens

    pieces: List[str] = []
    for w in se_elem.findall(".//w"):
        pieces.append(w.attrib.get("translit", ""))
        pieces.append(w.text or "")
    if pieces:
        return tokenize_text(" ".join(pieces))

    return tokenize_text(" ".join(se_elem.itertext()))


def iter_plain_text(path: Path) -> Iterator[List[str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = tokenize_text(line)
            if tokens:
                yield tokens


def iter_xml_file_bytes(data: bytes, xml_mode: str) -> Iterator[List[str]]:
    try:
        context = ET.iterparse(io.BytesIO(data), events=("end",))
        for _, elem in context:
            if elem.tag == "se" and elem.attrib.get("lang") == "srp":
                tokens = xml_sentence_tokens(elem, xml_mode)
                if tokens:
                    yield tokens
                elem.clear()
    except ET.ParseError:
        return


def iter_zip_xml(path: Path, xml_mode: str) -> Iterator[List[str]]:
    with zipfile.ZipFile(path) as zf:
        names = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".xml") and not name.lower().endswith(".bak")
        ]
        for name in names:
            with zf.open(name) as f:
                data = f.read()
            yield from iter_xml_file_bytes(data, xml_mode)


class SerbianSentenceCorpus:
    """Re-iterable corpus object for gensim's build_vocab and train passes."""

    def __init__(
        self,
        text_paths: Sequence[Path],
        zip_paths: Sequence[Path],
        xml_mode: str,
        max_sentences: int = 0,
        progress_every: int = 50000,
    ) -> None:
        self.text_paths = list(text_paths)
        self.zip_paths = list(zip_paths)
        self.xml_mode = xml_mode
        self.max_sentences = max_sentences
        self.progress_every = progress_every

    def __iter__(self) -> Iterator[List[str]]:
        yielded = 0
        tokens_seen = 0
        started_at = time.perf_counter()
        logging.info("Corpus pass started.")

        for path in self.text_paths:
            if not path.exists():
                logging.warning("Text file not found: %s", path)
                continue
            logging.info("Reading text file: %s", path)
            for tokens in iter_plain_text(path):
                yielded += 1
                tokens_seen += len(tokens)
                self._log_progress(yielded, tokens_seen, started_at)
                yield tokens
                if self.max_sentences and yielded >= self.max_sentences:
                    self._log_finished(yielded, tokens_seen, started_at, limited=True)
                    return

        for path in self.zip_paths:
            if not path.exists():
                logging.warning("Zip file not found: %s", path)
                continue
            logging.info("Reading zip XML corpus: %s", path)
            for tokens in iter_zip_xml(path, self.xml_mode):
                yielded += 1
                tokens_seen += len(tokens)
                self._log_progress(yielded, tokens_seen, started_at)
                yield tokens
                if self.max_sentences and yielded >= self.max_sentences:
                    self._log_finished(yielded, tokens_seen, started_at, limited=True)
                    return

        self._log_finished(yielded, tokens_seen, started_at, limited=False)

    def _log_progress(self, yielded: int, tokens_seen: int, started_at: float) -> None:
        if not self.progress_every or yielded % self.progress_every != 0:
            return
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        logging.info(
            "Corpus progress: %s sentences, %s tokens, %.1f sentences/sec.",
            f"{yielded:,}",
            f"{tokens_seen:,}",
            yielded / elapsed,
        )

    def _log_finished(self, yielded: int, tokens_seen: int, started_at: float, limited: bool) -> None:
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        suffix = " (stopped by --max-sentences)" if limited else ""
        logging.info(
            "Corpus pass finished%s: %s sentences, %s tokens in %.1f sec.",
            suffix,
            f"{yielded:,}",
            f"{tokens_seen:,}",
            elapsed,
        )


class EpochProgressLogger:
    """Gensim callback that logs epoch boundaries."""

    def __init__(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs
        self.epoch = 0
        self.started_at = 0.0

    def on_train_begin(self, model) -> None:
        logging.info("Training started.")

    def on_train_end(self, model) -> None:
        logging.info("Training finished.")

    def on_epoch_begin(self, model) -> None:
        self.started_at = time.perf_counter()
        logging.info("Epoch %s/%s started.", self.epoch + 1, self.total_epochs)

    def on_epoch_end(self, model) -> None:
        elapsed = time.perf_counter() - self.started_at
        self.epoch += 1
        logging.info("Epoch %s/%s finished in %.1f sec.", self.epoch, self.total_epochs, elapsed)


def train_model(args: argparse.Namespace, corpus: SerbianSentenceCorpus):
    if args.model_type == "fasttext":
        from gensim.models import FastText

        model = FastText(
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            workers=args.workers,
            sg=args.sg,
            negative=args.negative,
            sample=args.sample,
            bucket=args.bucket,
            min_n=args.min_n,
            max_n=args.max_n,
            seed=args.seed,
        )
    else:
        from gensim.models import Word2Vec

        model = Word2Vec(
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            workers=args.workers,
            sg=args.sg,
            negative=args.negative,
            sample=args.sample,
            seed=args.seed,
        )

    logging.info("Building vocabulary...")
    model.build_vocab(corpus)
    logging.info("Vocabulary size after min_count=%s: %s", args.min_count, len(model.wv))

    logging.info("Training %s for %s epochs...", args.model_type, args.epochs)
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=args.epochs,
        callbacks=[EpochProgressLogger(args.epochs)],
    )
    return model


def save_outputs(model, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    model_path = output_prefix.with_suffix(".model")
    kv_path = output_prefix.with_suffix(".kv")
    vec_path = output_prefix.with_suffix(".vec")

    model.save(str(model_path))
    model.wv.save(str(kv_path))
    model.wv.save_word2vec_format(str(vec_path), binary=False)

    print("Saved model:", model_path)
    print("Saved keyed vectors:", kv_path)
    print("Saved text vectors:", vec_path)


def coverage_report(model, pair_dataset: Path, output_prefix: Path) -> None:
    if not pair_dataset.exists():
        logging.warning("Pair dataset not found, skipping coverage report: %s", pair_dataset)
        return

    df = pd.read_csv(pair_dataset)
    needed = ["base_verb", "prefixed_verb", "prefix"]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        logging.warning("Pair dataset missing columns %s, skipping coverage.", missing)
        return

    rows = []
    for _, row in df.loc[:, needed].dropna().iterrows():
        base = normalize_token(str(row["base_verb"]))
        prefixed = normalize_token(str(row["prefixed_verb"]))
        prefix = str(row["prefix"])
        base_in_vocab = base in model.wv.key_to_index
        prefixed_in_vocab = prefixed in model.wv.key_to_index

        rows.append(
            {
                "base_verb": base,
                "prefixed_verb": prefixed,
                "prefix": prefix,
                "base_in_vocab": base_in_vocab,
                "prefixed_in_vocab": prefixed_in_vocab,
                "both_in_vocab": base_in_vocab and prefixed_in_vocab,
            }
        )

    out = pd.DataFrame(rows)
    coverage_path = output_prefix.parent / f"{output_prefix.name}_coverage.csv"
    out.to_csv(coverage_path, index=False)

    print("Saved coverage report:", coverage_path)
    print("Pair rows:", len(out))
    print("Both verbs in exact vocabulary:", int(out["both_in_vocab"].sum()), "of", len(out))

    by_prefix = (
        out.groupby("prefix", as_index=False)
        .agg(pairs=("prefix", "size"), both_in_vocab=("both_in_vocab", "sum"))
        .sort_values(["both_in_vocab", "pairs"], ascending=False)
    )
    by_prefix["coverage"] = by_prefix["both_in_vocab"] / by_prefix["pairs"]
    print()
    print("Coverage by prefix:")
    print(by_prefix.to_string(index=False))


def main() -> None:
    configure_logging()
    args = parse_args()

    text_paths = [Path(path) for path in args.texts]
    zip_paths = [Path(path) for path in args.zips]
    output_prefix = Path(args.output_prefix)

    corpus = SerbianSentenceCorpus(
        text_paths=text_paths,
        zip_paths=zip_paths,
        xml_mode=args.xml_mode,
        max_sentences=args.max_sentences,
        progress_every=args.progress_every,
    )

    print("Model type:", args.model_type)
    print("Text files:", text_paths)
    print("Zip files:", zip_paths)
    print("Output prefix:", output_prefix)
    print("XML mode:", args.xml_mode)
    if args.max_sentences:
        print("Debug max sentences:", args.max_sentences)
    print("Progress every sentences:", args.progress_every)

    model = train_model(args, corpus)
    save_outputs(model, output_prefix)
    coverage_report(model, Path(args.pair_dataset), output_prefix)


if __name__ == "__main__":
    main()
