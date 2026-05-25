import stanza
import re
import os
import csv
from collections import Counter

INPUT_FILE = "sr_Latn.txt"
OUTPUT_CSV = "serbian_verbs_frequency_lines_0_400000.csv"
OUTPUT_DETAILED = "serbian_verbs_detailed_lines_0_400000.txt"
MAX_CHARS_PER_PART = 50000
START_LINE = 130000
END_LINE = 400000


try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    nlp = stanza.Pipeline(
        "sr",
        processors="tokenize,pos,lemma",
        use_gpu=False,
        verbose=False
    )

except Exception as e:
    exit()

def read_specific_lines_range(file_path, start_line, end_line):
    encodings = ['utf-8', 'utf-8-sig', 'windows-1250', 'cp1251', 'iso-8859-2']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content_lines = []
                lines_read = 0
                
                for line_num, line in enumerate(f, 1):
                    if line_num < start_line:
                        continue
                    if line_num > end_line:
                        break
                    
                    content_lines.append(line.strip())
                    lines_read += 1
                
                content = ' '.join(content_lines)
                print(f"Прочитано {lines_read} строк ({start_line}-{min(end_line, line_num)})")
                return content, enc
                
        except UnicodeDecodeError:
            continue
    
    raise Exception(f"Не удалось прочитать файл {file_path}")

def extract_verbs_with_frequency(text):
    """Извлекает глаголы и считает их частотность."""
    print(f"Анализ текста ({len(text):,} символов)...")
    
    if len(text) > MAX_CHARS_PER_PART:
        print(f"Текст большой, обрабатываю частями по {MAX_CHARS_PER_PART:,} символов")
        parts = [text[i:i+MAX_CHARS_PER_PART] for i in range(0, len(text), MAX_CHARS_PER_PART)]
    else:
        parts = [text]
    
    lemma_counter = Counter()
    form_counter = Counter()
    all_verbs_detailed = []
    
    for i, part in enumerate(parts):
        if len(parts) > 1:
            print(f"  Обработка части {i+1}/{len(parts)}...")
        
        try:
            doc = nlp(part)
            
            for sentence in doc.sentences:
                for word in sentence.words:
                    if word.upos in ["VERB", "AUX"]:
                        lemma = word.lemma.lower()
                        form = word.text
                        
                        lemma_counter[lemma] += 1
                        form_counter[form] += 1
                        
                        all_verbs_detailed.append({
                            'lemma': lemma,
                            'form': form,
                            'pos': word.upos,
                            'feats': word.feats or "",
                            'sentence': sentence.text[:150]
                        })
                        
        except Exception as e:
            print(f"Ошибка в части {i+1}: {e}")
            continue
    
    return all_verbs_detailed, lemma_counter, form_counter

def save_to_csv(lemma_counter, output_csv):
    """Сохраняет леммы и их частотность в CSV файл."""
    print(f"Сохранение результатов в CSV: {output_csv}")
    
    sorted_lemmas = sorted(lemma_counter.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['глагол', 'частота'])
        
        for lemma, frequency in sorted_lemmas:
            writer.writerow([lemma, frequency])
    
    print(f"CSV файл создан. Сохранено {len(sorted_lemmas)} уникальных глаголов.")

def save_detailed_results(verbs_detailed, output_file):
    """Сохраняет подробные результаты в текстовый файл."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ПОДРОБНЫЕ РЕЗУЛЬТАТЫ ИЗВЛЕЧЕНИЯ ГЛАГОЛОВ\n")
        f.write("=" * 80 + "\n\n")
        
        for i, verb in enumerate(verbs_detailed, 1):
            f.write(f"{i:5d}. Лемма: {verb['lemma']:20s} | Форма: {verb['form']:15s} | ")
            f.write(f"POS: {verb['pos']:5s} | Морфология: {verb['feats']:20s}\n")
            f.write(f"     Контекст: {verb['sentence']}\n")
            f.write("-" * 80 + "\n")

def display_summary(lemma_counter, verbs_detailed):
    """Выводит краткую статистику."""
    total_verbs = len(verbs_detailed)
    unique_lemmas = len(lemma_counter)
    
    print("\n" + "=" * 60)
    print("КРАТКАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"Всего глаголов в тексте: {total_verbs:,}")
    print(f"Уникальных глаголов (лемм): {unique_lemmas:,}")
    
    if unique_lemmas > 0:
        print(f"Средняя частота на глагол: {total_verbs/unique_lemmas:.2f}")
    
    print(f"ТОП-10 САМЫХ ЧАСТЫХ ГЛАГОЛОВ:")
    print("-" * 40)
    for lemma, count in lemma_counter.most_common(10):
        percentage = (count / total_verbs) * 100 if total_verbs > 0 else 0
        print(f"  {lemma:20s} : {count:6d} ({percentage:5.2f}%)")

# =====================
# ОСНОВНАЯ ПРОГРАММА
# =====================
if __name__ == "__main__":
    # Проверка файла
    if not os.path.exists(INPUT_FILE):
        print(f"Файл '{INPUT_FILE}' не найден!")
        print("Файлы в текущей папке:")
        for file in sorted(os.listdir('.')):
            print(f"  - {file}")
        exit()
    
    try:
        # 1. Чтение только нужного диапазона строк
        print(f"Чтение файла '{INPUT_FILE}' (строки {START_LINE}-{END_LINE})...")
        text, encoding = read_specific_lines_range(INPUT_FILE, START_LINE, END_LINE)
        
        if len(text.strip()) == 0:
            print(f"Диапазон строк {START_LINE}-{END_LINE} пуст!")
            exit()
        
        # 2. Извлечение глаголов
        verbs_detailed, lemma_counter, form_counter = extract_verbs_with_frequency(text)
        
        if not verbs_detailed:
            print("В выбранном диапазоне не найдено глаголов!")
            exit()
        
        # 3. Сохранение результатов
        save_to_csv(lemma_counter, OUTPUT_CSV)
        save_detailed_results(verbs_detailed, OUTPUT_DETAILED)
        
        # 4. Вывод статистики
        display_summary(lemma_counter, verbs_detailed)
        
        print(f"Результаты для строк {START_LINE}-{END_LINE} сохранены в:")
        print(f"  1. {OUTPUT_CSV}")
        print(f"  2. {OUTPUT_DETAILED}")
        
        print("\n" + "=" * 60)
        print("ПРОГРАММА ЗАВЕРШЕНА УСПЕШНО!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("Программа прервана пользователем")
    except Exception as e:
        print(f"ОШИБКА: {e}")
        import traceback
        traceback.print_exc()