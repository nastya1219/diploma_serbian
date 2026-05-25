import csv
from collections import defaultdict

input_file = "output_latin_ruscorpora.csv"
output_file = "aggregated_latin.csv"

freqs = defaultdict(int)

with open(input_file, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lemma = row["infinitive_latin_norm"]
        freq = int(row["frequency"])
        freqs[lemma] += freq

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lemma_latin", "total_frequency"])

    for lemma, freq in sorted(freqs.items(), key=lambda x: -x[1]):
        writer.writerow([lemma, freq])

print("Готово: aggregated_latin.csv")