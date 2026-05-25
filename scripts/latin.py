import csv

# Читаем данные
with open('суммарные_частотности_глаголов.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data = [(row['глагол'], int(row['общая_частота'])) for row in reader]

# Создаём словарь
freq_dict = {verb: freq for verb, freq in data}

# Уникальные приставки
prefixes = [
    'po', 'pri', 'od', 'iz', 'na', 'u', 'pre', 'za', 'do', 'pod', 'raz', 'pro',
    's', 'o', 'nad', 'ob', 'pr', 'ot', 'ras', 'bez', 'uz', 'is', 'pred'
]

# Находим пары
pairs = []

for prefixed_verb in freq_dict.keys():
    for prefix in prefixes:
        if prefixed_verb.startswith(prefix):
            base = prefixed_verb[len(prefix):]
            if base in freq_dict:
                pairs.append((
                    base,                    # base_verb
                    prefixed_verb,           # prefixed_verb
                    prefix,                  # prefix
                    freq_dict[base],         # freq_base
                    freq_dict[prefixed_verb] # freq_prefixed
                ))
                break  # чтобы не искать другие приставки для этого глагола

# Сохраняем результат
with open('verb_prefix_pairs.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['base_verb', 'prefixed_verb', 'prefix', 'freq_base', 'freq_prefixed'])
    writer.writerows(pairs)

print(f"Найдено {len(pairs)} пар. Результат сохранён в verb_prefix_pairs.csv")