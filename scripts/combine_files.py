import pandas as pd

# Загружаем данные
df_freq = pd.read_csv("aggregated_latin.csv")  # столбцы: глагол, частота
df_pairs = pd.read_csv("serbian_verb_final_02_02.csv")  # столбцы: base_verb, prefixed_verb, prefix, freq_base, freq_prefixed

print(f"Загружено {len(df_freq)} глаголов из serbian_verbs_frequency.csv")
print(f"Загружено {len(df_pairs)} пар из serbian_prefixed_verb_pairs.csv")

# Создаем словарь суммарных частот
total_freq_dict = {}

# 1. Добавляем частоты из serbian_verbs_frequency.csv
for _, row in df_freq.iterrows():
    verb = row['глагол']
    freq = row['общая_частота']
    total_freq_dict[verb] = total_freq_dict.get(verb, 0) + freq

# 2. Добавляем частоты из serbian_prefixed_verb_pairs.csv
for _, row in df_pairs.iterrows():
    base_verb = row['base_verb']
    base_freq = row['freq_base']
    total_freq_dict[base_verb] = total_freq_dict.get(base_verb, 0) + base_freq
    
    prefixed_verb = row['prefixed_verb']
    prefixed_freq = row['freq_prefixed']
    total_freq_dict[prefixed_verb] = total_freq_dict.get(prefixed_verb, 0) + prefixed_freq

# 3. Создаем новый DataFrame с суммарными частотами
result_data = []

for _, row in df_pairs.iterrows():
    base_verb = row['base_verb']
    prefixed_verb = row['prefixed_verb']
    prefix = row['prefix']
    
    # Суммарные частоты
    freq_base_total = total_freq_dict.get(base_verb, 0)
    freq_prefixed_total = total_freq_dict.get(prefixed_verb, 0)
    
    result_data.append({
        'base_verb': base_verb,
        'prefixed_verb': prefixed_verb,
        'prefix': prefix,
        'freq_base': freq_base_total,
        'freq_prefixed': freq_prefixed_total
    })

# 4. Создаем и сохраняем результат
df_result = pd.DataFrame(result_data)
output_file = 'serbian_verb_final_02_02_with_ruscorpora.csv'
df_result.to_csv(output_file, index=False, encoding='utf-8')

print(f"Результат сохранен в: {output_file}")
print(f"Создано {len(df_result)} пар с суммарными частотами")

# 5. Показываем примеры
print("Первые 5 строк результата:")
print(df_result.head().to_string(index=False))

# 6. Статистика
print(f"Статистика:")
print(f"Всего уникальных глаголов: {len(total_freq_dict)}")

# Пример сложения для первых 3 пар
print("Примеры сложения частот:")
for i in range(min(3, len(df_pairs))):
    base = df_pairs.iloc[i]['base_verb']
    prefixed = df_pairs.iloc[i]['prefixed_verb']
    
    # Частоты из freq файла
    base_freq1 = df_freq[df_freq['глагол'] == base]['общая_частота']
    base_freq1_val = base_freq1.iloc[0] if len(base_freq1) > 0 else 0
    
    prefixed_freq1 = df_freq[df_freq['глагол'] == prefixed]['общая_частота']
    prefixed_freq1_val = prefixed_freq1.iloc[0] if len(prefixed_freq1) > 0 else 0
    
    # Частоты из pairs файла (оригинальные)
    base_freq2 = df_pairs.iloc[i]['freq_base']
    prefixed_freq2 = df_pairs.iloc[i]['freq_prefixed']
    
    # Суммарные
    base_total = total_freq_dict.get(base, 0)
    prefixed_total = total_freq_dict.get(prefixed, 0)
    
    print(f"{i+1}. {base} → {prefixed}")
    print(f"   {base}: {base_freq1_val} + {base_freq2} = {base_total}")
    print(f"   {prefixed}: {prefixed_freq1_val} + {prefixed_freq2} = {prefixed_total}")