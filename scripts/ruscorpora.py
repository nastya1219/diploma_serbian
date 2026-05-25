import os
import re
from collections import Counter
import csv

print("=" * 60)
print("АНАЛИЗ ВСЕХ XML ФАЙЛОВ В ТЕКУЩЕЙ ПАПКЕ")
print("=" * 60)

# 1. Находим все XML файлы в текущей папке
current_dir = os.getcwd()
xml_files = [f for f in os.listdir(current_dir) 
             if f.lower().endswith('.xml') and os.path.isfile(f)]

print(f"Текущая папка: {current_dir}")
print(f"Найдено XML файлов: {len(xml_files)}")

if not xml_files:
    print("XML файлы не найдены в текущей папке!")
    print("Убедитесь, что файлы имеют расширение .xml")
    exit()

# Покажем список файлов
print("\nСписок XML файлов:")
for i, file in enumerate(xml_files, 1):
    size = os.path.getsize(file)
    print(f"  {i}. {file} ({size:,} байт)")

# 2. Функция для извлечения глаголов из одного файла
def extract_verbs_from_file(file_path):
    """Извлекает инфинитивы глаголов из XML файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Паттерн для поиска глаголов (lex_translit из тегов с gr="V")
        pattern = r'lex_translit="([^"]+)"[^>]*gr="[^"]*V[^"]*"'
        matches = re.findall(pattern, content)
        
        return Counter(matches), len(matches), len(set(matches))
    
    except Exception as e:
        print(f"  Ошибка чтения {file_path}: {e}")
        return Counter(), 0, 0

# 3. Обрабатываем все файлы
total_counter = Counter()  # Общий счётчик для всех файлов
file_results = []          # Результаты по каждому файлу

print("\n" + "=" * 60)
print("ОБРАБОТКА ФАЙЛОВ")
print("=" * 60)

for i, xml_file in enumerate(xml_files, 1):
    print(f"\n[{i}/{len(xml_files)}] Обрабатываю: {xml_file}")
    
    verb_counter, total_verbs, unique_verbs = extract_verbs_from_file(xml_file)
    
    # Добавляем в общий счётчик
    total_counter.update(verb_counter)
    
    # Сохраняем результаты для этого файла
    file_results.append({
        'filename': xml_file,
        'total_verbs': total_verbs,
        'unique_verbs': unique_verbs,
        'verb_counter': verb_counter
    })
    
    print(f"  ✓ Найдено глаголов: {total_verbs}")
    print(f"  ✓ Уникальных инфинитивов: {unique_verbs}")
    
    if verb_counter:
        top_verb, top_count = verb_counter.most_common(1)[0]
        print(f"  ✓ Самый частый: '{top_verb}' ({top_count} раз)")

# 4. Сохраняем ОБЩИЙ результат (все файлы вместе)
print("\n" + "=" * 60)
print("СОХРАНЕНИЕ ОБЩЕГО РЕЗУЛЬТАТА")
print("=" * 60)

output_file = 'all_xml_verbs_combined_0202.csv'
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['infinitive_latin', 'frequency'])
    for verb, count in total_counter.most_common():
        writer.writerow([verb, count])

print(f"✓ Общая частотность сохранена в: {output_file}")
print(f"  Всего уникальных глаголов (во всех файлах): {len(total_counter)}")

# 5. Сохраняем результаты по КАЖДОМУ файлу отдельно
print("\n" + "=" * 60)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ПО ФАЙЛАМ")
print("=" * 60)

for result in file_results:
    if result['verb_counter']:
        # Создаём имя файла для результатов
        base_name = result['filename'].replace('.xml', '_verbs.csv')
        with open(base_name, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['infinitive_latin', 'frequency'])
            for verb, count in result['verb_counter'].most_common():
                writer.writerow([verb, count])
        print(f"✓ {base_name} - {result['total_verbs']} глаголов")

# 6. Создаём сводную таблицу по файлам
summary_file = 'xml_files_summary.csv'
with open(summary_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['xml_file', 'total_verbs', 'unique_verbs', 'top_verb', 'top_count'])
    
    for result in file_results:
        if result['verb_counter']:
            top_verb, top_count = result['verb_counter'].most_common(1)[0]
        else:
            top_verb, top_count = 'нет', 0
        
        writer.writerow([
            result['filename'],
            result['total_verbs'],
            result['unique_verbs'],
            top_verb,
            top_count
        ])

print(f"\n✓ Сводная таблица сохранена в: {summary_file}")

# 7. Показываем общую статистику
print("\n" + "=" * 60)
print("ОБЩАЯ СТАТИСТИКА")
print("=" * 60)

# Подсчитываем общие числа
total_all_verbs = sum(r['total_verbs'] for r in file_results)
total_unique_all = len(total_counter)

print(f"Обработано XML файлов: {len(xml_files)}")
print(f"Всего вхождений глаголов: {total_all_verbs}")
print(f"Уникальных инфинитивов (во всех файлах): {total_unique_all}")

# Топ-20 самых частотных глаголов
print(f"\nТОП-20 самых частотных глаголов (все файлы):")
print("-" * 60)
print(f"{'Глагол':<25} {'Частота':<10} {'%':<8}")
print("-" * 60)

for verb, count in total_counter.most_common(20):
    percentage = (count / total_all_verbs * 100) if total_all_verbs > 0 else 0
    print(f"{verb:<25} {count:<10} {percentage:.2f}%")

# 8. Создаём текстовый отчёт
report_file = 'verbs_analysis_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ ПО АНАЛИЗУ СЕРБСКИХ ГЛАГОЛОВ ИЗ XML ФАЙЛОВ\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Дата анализа: {os.popen('date /t').read().strip() if os.name == 'nt' else ''}\n")
    f.write(f"Папка: {current_dir}\n")
    f.write(f"Количество XML файлов: {len(xml_files)}\n")
    f.write(f"Всего вхождений глаголов: {total_all_verbs}\n")
    f.write(f"Уникальных инфинитивов: {total_unique_all}\n\n")
    
    f.write("Топ-15 самых частотных глаголов:\n")
    f.write("-" * 50 + "\n")
    for verb, count in total_counter.most_common(15):
        percentage = (count / total_all_verbs * 100) if total_all_verbs > 0 else 0
        f.write(f"{verb:<20} {count:<8} ({percentage:.2f}%)\n")
    
    f.write("\nСтатистика по файлам:\n")
    f.write("-" * 80 + "\n")
    for result in file_results:
        if result['verb_counter']:
            top_verb, top_count = result['verb_counter'].most_common(1)[0]
        else:
            top_verb, top_count = 'нет', 0
        
        f.write(f"{result['filename']:<30} {result['total_verbs']:>6} глаголов, "
                f"{result['unique_verbs']:>4} уникальных, "
                f"топ: {top_verb} ({top_count})\n")

print(f"\n✓ Текстовый отчёт сохранён в: {report_file}")

# 9. Проверка на возможные ошибки разметки
print("\n" + "=" * 60)
print("ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
print("=" * 60)

# Ищем слова, которые не похожи на глаголы
suspicious_words = []
for verb in total_counter:
    # Сербские глаголы обычно оканчиваются на -ti, -ći, -sti
    if not any(verb.endswith(ending) for ending in ['ti', 'ći', 'sti', 'ći se', 'ti se']):
        suspicious_words.append(verb)

if suspicious_words:
    print(f"Найдено {len(suspicious_words)} слов, не похожих на глаголы:")
    print("(возможная ошибка разметки в XML)")
    
    suspicious_counts = [(verb, total_counter[verb]) for verb in suspicious_words]
    suspicious_counts.sort(key=lambda x: x[1], reverse=True)
    
    for verb, count in suspicious_counts[:10]:  # показываем топ-10
        print(f"  '{verb}': {count} раз")
    
    # Сохраняем список подозрительных слов
    with open('suspicious_non_verbs.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'frequency'])
        for verb, count in suspicious_counts:
            writer.writerow([verb, count])
    
    print(f"\n✓ Список подозрительных слов сохранён в: suspicious_non_verbs.csv")
else:
    print("✓ Все слова выглядят как глаголы (оканчиваются на -ti, -ći, -sti)")

print("\n" + "=" * 60)
print("ГОТОВО! СОЗДАНЫ ФАЙЛЫ:")
print("=" * 60)
print(f"1. {output_file} - все глаголы из всех XML (основной файл)")
print(f"2. {summary_file} - статистика по каждому XML файлу")
print(f"3. {report_file} - текстовый отчёт")
print(f"4. *_verbs.csv - результаты по каждому файлу отдельно")
if suspicious_words:
    print(f"5. suspicious_non_verbs.csv - подозрительные слова")
print("\nМожете открывать CSV файлы в Excel или любой табличной программе.")