import stanza
from tqdm import tqdm

stanza.download("sr")
nlp = stanza.Pipeline(
    "sr",
    processors="tokenize,pos",
    tokenize_no_ssplit=True,
    verbose=False
)

verbs = []

with open("hbs.freq", encoding="utf-8") as f:
    for line in tqdm(f, desc="Обработка слов"):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        freq = int(parts[0])
        word = parts[1]

        # отсекаем пунктуацию и странные символы
        if not word.isalpha():
            continue

        doc = nlp(word)

        for sent in doc.sentences:
            for token in sent.words:
                if token.upos in {"VERB", "AUX"}:
                    verbs.append((word, freq))
with open("hbs_freq.txt", "w", encoding="utf-8") as out:
    for word, freq in verbs:
        out.write(f"{freq}\t{word}\n")
