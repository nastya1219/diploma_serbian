# Auto-generated variant: process docs 1001-2000
import os
import csv
import argparse
from collections import Counter

import stanza
from datasets import load_dataset


SERBIAN_PREFIXES = [
    "do", "na", "o", "ob", "od", "po", "pre", "pri", "pro", "raz",
    "s", "sa", "su", "u", "uz", "v", "iz", "za", "nad", "pod",
    "pred", "bez", "ras", "pr", "ot", "prema", "protiv", "obez",
]


def build_stanza_pipeline():
    # Download models if missing.
    try:
        stanza.download("sr", verbose=False)
    except Exception:
        # If already downloaded or offline, stanza may raise.
        pass
    return stanza.Pipeline(
        "sr",
        processors="tokenize,pos,lemma",
        use_gpu=False,
        verbose=False,
    )


def iter_text_stream(dataset_name, split, token=None):
    ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
    for row in ds:
        text = row.get("text")
        if text:
            yield text


def extract_verb_lemmas(text_iter, nlp, max_chars=50000, max_docs=None, start_doc=0):
    lemma_counter = Counter()
    seen_docs = 0
    processed_docs = 0
    buf = []
    buf_len = 0

    def flush_buffer():
        nonlocal buf, buf_len, seen_docs, processed_docs
        if not buf:
            return
        doc_text = " ".join(buf)
        buf = []
        buf_len = 0
        seen_docs += 1
        if seen_docs <= start_doc:
            return
        if max_docs is not None and processed_docs >= max_docs:
            return
        processed_docs += 1
        doc = nlp(doc_text)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos in ("VERB", "AUX"):
                    lemma = (word.lemma or "").lower()
                    if lemma:
                        lemma_counter[lemma] += 1

    for text in text_iter:
        if max_docs is not None and processed_docs >= max_docs:
            break
        text = text.strip()
        if not text:
            continue
        buf.append(text)
        buf_len += len(text)
        if buf_len >= max_chars:
            flush_buffer()

    flush_buffer()
    return lemma_counter


def save_lemma_freq(lemma_counter, out_csv):
    rows = sorted(lemma_counter.items(), key=lambda x: x[1], reverse=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["глагол", "частота"])
        writer.writerows(rows)


def build_pairs_from_freq(lemma_counter):
    freq_dict = dict(lemma_counter)
    pairs = []
    for prefixed in freq_dict.keys():
        for prefix in SERBIAN_PREFIXES:
            if prefixed.startswith(prefix) and len(prefixed) > len(prefix) + 2:
                base = prefixed[len(prefix):]
                if base in freq_dict:
                    pairs.append(
                        (
                            base,
                            prefixed,
                            prefix,
                            freq_dict[base],
                            freq_dict[prefixed],
                        )
                    )
                break
    return pairs


def save_pairs(pairs, out_csv):
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["base_verb", "prefixed_verb", "prefix", "freq_base", "freq_prefixed"]
        )
        writer.writerows(pairs)


def filter_pairs_min_freq(pairs, min_freq):
    return [p for p in pairs if p[3] >= min_freq and p[4] >= min_freq]


def main():
    parser = argparse.ArgumentParser(
        description="Build Serbian verb prefix dataset from HF SrpWikiDataset."
    )
    parser.add_argument(
        "--dataset",
        default="datatab/SrpWikiDataset",
        help="HF dataset name",
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=50000,
        help="Max chars per Stanza document",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=1000,
        help="Max docs to process (for quick tests)",
    )
    parser.add_argument(
        "--start-doc",
        type=int,
        default=1000,
        help="Skip first N docs (process from N+1)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=500,
        help="Minimum frequency for base and prefixed verbs",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("data", "csv"),
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    token = os.getenv("HF_TOKEN")
    nlp = build_stanza_pipeline()
    text_iter = iter_text_stream(args.dataset, args.split, token=token)
    lemma_counter = extract_verb_lemmas(
        text_iter,
        nlp,
        max_chars=args.max_chars,
        max_docs=args.max_docs,
        start_doc=args.start_doc,
    )

    lemma_csv = os.path.join(args.out_dir, "serbian_verbs_frequency_hf_1001_2000.csv")
    save_lemma_freq(lemma_counter, lemma_csv)

    pairs = build_pairs_from_freq(lemma_counter)
    pairs_csv = os.path.join(args.out_dir, "serbian_verb_pairs_hf_1001_2000.csv")
    save_pairs(pairs, pairs_csv)

    pairs_500 = filter_pairs_min_freq(pairs, args.min_freq)
    final_csv = os.path.join(
        args.out_dir, "serbian_verb_final_hf_wiki_5_1001_2000.csv"
    )
    save_pairs(pairs_500, final_csv)

    print(f"Lemma freq CSV: {lemma_csv}")
    print(f"All pairs CSV:  {pairs_csv}")
    print(f"Final (>= {args.min_freq}) CSV: {final_csv}")
    print(f"Total pairs: {len(pairs)}; Final pairs: {len(pairs_500)}")


if __name__ == "__main__":
    main()
