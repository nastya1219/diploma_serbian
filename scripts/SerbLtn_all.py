import pandas as pd
import glob

# Найти все CSV файлы
files = glob.glob("serbian_verbs_frequency_lines*.csv")

# Создать пустой словарь для агрегации
total_frequencies = {}

# Обработать каждый файл
for file in files:
    print(f"Обрабатываю файл: {file}")
    
    # Читаем CSV
    df = pd.read_csv(file)
    
    # Проверяем структуру
    if 'глагол' in df.columns and 'частота' in df.columns:
        # Добавляем частотности в общий словарь
        for _, row in df.iterrows():
            verb = row['глагол']
            freq = row['частота']
            
            if verb in total_frequencies:
                total_frequencies[verb] += freq
            else:
                total_frequencies[verb] = freq

# Создаем итоговый DataFrame
result_df = pd.DataFrame(list(total_frequencies.items()), 
                         columns=['глагол', 'общая_частота'])

# Сортируем по убыванию частотности
result_df = result_df.sort_values('общая_частота', ascending=False)

print(f"\nОбработано файлов: {len(files)}")
print(f"Уникальных глаголов: {len(result_df)}")
print("\nТоп-10 самых частотных глаголов:")
print(result_df.head(10))

# Сохраняем результат
result_df.to_csv("суммарные_частотности_глаголов.csv", index=False, encoding='utf-8')
print("Результат сохранен в: суммарные_частотности_глаголов.csv")