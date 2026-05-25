import pandas as pd
import re
import json
from datetime import datetime

# КОРРЕКТНЫЙ список сербских глагольных приставок
SERBIAN_PREFIXES = [
    'do', 'na', 'o', 'ob', 'od', 'po', 'pre', 'pri', 'pro', 'raz',
    's', 'sa', 'su', 'u', 'uz', 'v', 'iz', 'za', 'nad', 'pod', 
    'pred', 'bez', 'ras', 'pr', 'ot', 'prema', 'protiv', 'obez'
]

def find_verb_pairs(verbs_df):
    """Находит пары базовых и приставочных глаголов"""
    
    all_verbs = set(verbs_df['глагол'].tolist())
    verb_freq_dict = dict(zip(verbs_df['глагол'], verbs_df['частота']))
    
    potential_pairs = {}
    
    # Проходим по всем глаголам
    for verb in all_verbs:
        # Пропускаем очень короткие
        if len(verb) < 4:
            continue
            
        # Пытаемся найти приставку
        found_prefix = None
        found_root = None
        
        for prefix in SERBIAN_PREFIXES:
            if verb.startswith(prefix) and len(verb) > len(prefix) + 2:
                root = verb[len(prefix):]
                # Проверяем, существует ли корень как отдельный глагол
                if root in all_verbs:
                    found_prefix = prefix
                    found_root = root
                    break
        
        if found_root:
            if found_root not in potential_pairs:
                potential_pairs[found_root] = {
                    'base_verb': found_root,
                    'base_freq': verb_freq_dict.get(found_root, 0),
                    'derivatives': []
                }
            
            # Добавляем производный глагол (если это не сам базовый)
            if verb != found_root:
                potential_pairs[found_root]['derivatives'].append({
                    'verb': verb,
                    'prefix': found_prefix,
                    'frequency': verb_freq_dict.get(verb, 0)
                })
    
    # Формируем финальный список пар
    pairs = []
    for root, data in potential_pairs.items():
        if data['base_freq'] > 0 and data['derivatives']:
            # Сортируем производные по частоте
            data['derivatives'].sort(key=lambda x: -x['frequency'])
            pairs.append({
                'base_verb': data['base_verb'],
                'base_frequency': data['base_freq'],
                'derivatives_count': len(data['derivatives']),
                'derivatives': data['derivatives']
            })
    
    # Сортируем пары
    pairs.sort(key=lambda x: (-x['derivatives_count'], -x['base_frequency']))
    return pairs

def save_to_csv(pairs, filename_prefix='serbian_verb_pairs'):
    """Сохраняет результаты в CSV файлы"""
    
    # 1. Основной файл со всеми парами в развернутом виде
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_filename = f"{filename_prefix}_{timestamp}.csv"
    
    rows = []
    for pair in pairs:
        base_row = {
            'type': 'BASE',
            'verb': pair['base_verb'],
            'frequency': pair['base_frequency'],
            'prefix': '',
            'base_verb': pair['base_verb'],
            'derivatives_count': pair['derivatives_count']
        }
        rows.append(base_row)
        
        for deriv in pair['derivatives']:
            deriv_row = {
                'type': 'DERIVATIVE',
                'verb': deriv['verb'],
                'frequency': deriv['frequency'],
                'prefix': deriv['prefix'],
                'base_verb': pair['base_verb'],
                'derivatives_count': ''
            }
            rows.append(deriv_row)
    
    df_main = pd.DataFrame(rows)
    df_main.to_csv(main_filename, index=False, encoding='utf-8')
    print(f"✓ Основной файл сохранен: {main_filename}")
    print(f"  Содержит {len(rows)} строк ({len(pairs)} пар глаголов)")
    
    # 2. Сводная таблица (компактный вид)
    summary_filename = f"{filename_prefix}_summary_{timestamp}.csv"
    
    summary_rows = []
    for pair in pairs:
        # Собираем все производные в одну строку
        derivatives_str = "; ".join([
            f"{d['prefix']}+{pair['base_verb']} ({d['frequency']})"
            for d in pair['derivatives'][:10]  # Первые 10
        ])
        if len(pair['derivatives']) > 10:
            derivatives_str += f" ... и еще {len(pair['derivatives']) - 10}"
        
        summary_rows.append({
            'base_verb': pair['base_verb'],
            'base_frequency': pair['base_frequency'],
            'derivatives_count': pair['derivatives_count'],
            'derivatives_examples': derivatives_str
        })
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_filename, index=False, encoding='utf-8')
    print(f"✓ Сводная таблица сохранена: {summary_filename}")
    
    # 3. Статистика по приставкам
    stats_filename = f"{filename_prefix}_stats_{timestamp}.csv"
    
    prefix_stats = {}
    for pair in pairs:
        for deriv in pair['derivatives']:
            prefix = deriv['prefix']
            if prefix not in prefix_stats:
                prefix_stats[prefix] = {
                    'count': 0,
                    'total_frequency': 0,
                    'examples': []
                }
            prefix_stats[prefix]['count'] += 1
            prefix_stats[prefix]['total_frequency'] += deriv['frequency']
            prefix_stats[prefix]['examples'].append(deriv['verb'])
    
    stats_rows = []
    for prefix, data in prefix_stats.items():
        # Берем 5 примеров
        examples = ", ".join(data['examples'][:5])
        if len(data['examples']) > 5:
            examples += f" (+{len(data['examples']) - 5})"
        
        stats_rows.append({
            'prefix': prefix,
            'verb_count': data['count'],
            'total_frequency': data['total_frequency'],
            'avg_frequency': data['total_frequency'] / data['count'],
            'examples': examples
        })
    
    df_stats = pd.DataFrame(stats_rows)
    df_stats = df_stats.sort_values('verb_count', ascending=False)
    df_stats.to_csv(stats_filename, index=False, encoding='utf-8')
    print(f"✓ Статистика по приставкам сохранена: {stats_filename}")
    
    return df_main, df_summary, df_stats

def save_to_json(pairs, filename_prefix='serbian_verb_pairs'):
    """Сохраняет результаты в JSON файл"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{filename_prefix}_{timestamp}.json"
    
    # Создаем структурированный JSON
    result_data = {
        'analysis_date': datetime.now().isoformat(),
        'total_pairs': len(pairs),
        'total_base_verbs': len(pairs),
        'total_derivatives': sum(p['derivatives_count'] for p in pairs),
        'pairs': pairs
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ JSON файл сохранен: {json_filename}")
    return result_data

def save_to_excel_alternative(pairs, filename_prefix='serbian_verb_pairs'):
    """Альтернатива сохранения в Excel - создает несколько CSV файлов вместо одного Excel"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Создаем папку для результатов
    import os
    folder_name = f"{filename_prefix}_excel_alternative_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    
    # 1. Все глаголы
    rows = []
    for pair in pairs:
        base_row = {
            'Тип': 'Базовый',
            'Глагол': pair['base_verb'],
            'Частота': pair['base_frequency'],
            'Приставка': '',
            'Базовый глагол': pair['base_verb'],
            'Кол-во производных': pair['derivatives_count']
        }
        rows.append(base_row)
        
        for deriv in pair['derivatives']:
            deriv_row = {
                'Тип': 'Производный',
                'Глагол': deriv['verb'],
                'Частота': deriv['frequency'],
                'Приставка': deriv['prefix'],
                'Базовый глагол': pair['base_verb'],
                'Кол-во производных': ''
            }
            rows.append(deriv_row)
    
    df_main = pd.DataFrame(rows)
    df_main.to_csv(os.path.join(folder_name, '1_все_глаголы.csv'), index=False, encoding='utf-8')
    
    # 2. Сводная таблица
    summary_rows = []
    for pair in pairs:
        derivatives_list = "\n".join([
            f"{d['prefix']}+{pair['base_verb']} = {d['verb']} ({d['frequency']})"
            for d in pair['derivatives']
        ])
        
        summary_rows.append({
            'Базовый глагол': pair['base_verb'],
            'Частота': pair['base_frequency'],
            'Кол-во производных': pair['derivatives_count'],
            'Производные глаголы': derivatives_list
        })
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(folder_name, '2_сводная_таблица.csv'), index=False, encoding='utf-8')
    
    # 3. Топ-50 пар
    df_top = pd.DataFrame(summary_rows[:50])
    df_top.to_csv(os.path.join(folder_name, '3_топ_50_пар.csv'), index=False, encoding='utf-8')
    
    # 4. Статистика
    stats_data = {
        'Метрика': [
            'Всего пар',
            'Всего базовых глаголов', 
            'Всего производных глаголов',
            'Среднее производных на базовый',
            'Максимум производных у одного',
            'Минимум производных у одного'
        ],
        'Значение': [
            len(pairs),
            len(pairs),
            sum(p['derivatives_count'] for p in pairs),
            sum(p['derivatives_count'] for p in pairs) / len(pairs),
            max(p['derivatives_count'] for p in pairs),
            min(p['derivatives_count'] for p in pairs)
        ]
    }
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_csv(os.path.join(folder_name, '4_статистика.csv'), index=False, encoding='utf-8')
    
    # 5. Статистика по приставкам
    prefix_stats = {}
    for pair in pairs:
        for deriv in pair['derivatives']:
            prefix = deriv['prefix']
            if prefix not in prefix_stats:
                prefix_stats[prefix] = {
                    'count': 0,
                    'total_frequency': 0,
                    'examples': []
                }
            prefix_stats[prefix]['count'] += 1
            prefix_stats[prefix]['total_frequency'] += deriv['frequency']
            prefix_stats[prefix]['examples'].append(deriv['verb'])
    
    stats_prefix_rows = []
    for prefix, data in prefix_stats.items():
        examples = ", ".join(data['examples'][:5])
        if len(data['examples']) > 5:
            examples += f" (+{len(data['examples']) - 5})"
        
        stats_prefix_rows.append({
            'Приставка': prefix,
            'Количество_глаголов': data['count'],
            'Суммарная_частота': data['total_frequency'],
            'Средняя_частота': data['total_frequency'] / data['count'],
            'Примеры': examples
        })
    
    df_prefix_stats = pd.DataFrame(stats_prefix_rows)
    df_prefix_stats = df_prefix_stats.sort_values('Количество_глаголов', ascending=False)
    df_prefix_stats.to_csv(os.path.join(folder_name, '5_статистика_приставок.csv'), index=False, encoding='utf-8')
    
    print(f"✓ Создана папка с CSV файлами: {folder_name}")
    print(f"  Файлы в папке:")
    print(f"  1. 1_все_глаголы.csv")
    print(f"  2. 2_сводная_таблица.csv")
    print(f"  3. 3_топ_50_пар.csv")
    print(f"  4. 4_статистика.csv")
    print(f"  5. 5_статистика_приставок.csv")
    
    return folder_name

def print_statistics(pairs, total_verbs):
    """Выводит статистику анализа"""
    print("\n" + "="*70)
    print("СТАТИСТИКА АНАЛИЗА")
    print("="*70)
    
    verbs_in_pairs = sum(len(pair['derivatives']) + 1 for pair in pairs)
    
    print(f"Всего глаголов в исходном списке: {total_verbs}")
    print(f"Глаголов, включенных в пары: {verbs_in_pairs} ({verbs_in_pairs/total_verbs*100:.1f}%)")
    print(f"Найдено пар базовый-производный: {len(pairs)}")
    print(f"Всего производных глаголов: {sum(p['derivatives_count'] for p in pairs)}")
    print(f"Среднее количество производных: {sum(p['derivatives_count'] for p in pairs)/len(pairs):.1f}")
    
    # Топ-10 по количеству производных
    print("\nТОП-10 базовых глаголов по количеству производных:")
    print("-" * 70)
    top_pairs = sorted(pairs, key=lambda x: -x['derivatives_count'])[:10]
    
    for i, pair in enumerate(top_pairs, 1):
        print(f"{i:2}. {pair['base_verb']:15} (частота: {pair['base_frequency']:6}) → {pair['derivatives_count']:2} производных")
    
    # Наиболее продуктивные приставки
    print("\nНАИБОЛЕЕ ЧАСТЫЕ ПРИСТАВКИ:")
    print("-" * 70)
    
    prefix_counts = {}
    for pair in pairs:
        for deriv in pair['derivatives']:
            prefix = deriv['prefix']
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    
    for i, (prefix, count) in enumerate(sorted(prefix_counts.items(), key=lambda x: -x[1])[:10], 1):
        print(f"{i:2}. Приставка '{prefix:5}': {count:3} глаголов")

def main():
    # Загрузка данных
    print("Загрузка данных из CSV...")
    try:
        df = pd.read_csv('serbian_verbs_frequency.csv')
        print(f"✓ Загружено {len(df)} глаголов")
    except FileNotFoundError:
        print("✗ Ошибка: Файл 'serbian_verbs_frequency.csv' не найден!")
        print("  Убедитесь, что файл находится в той же папке, что и скрипт.")
        return
    
    # Анализ глаголов
    print("\nАнализ глаголов и поиск пар...")
    pairs = find_verb_pairs(df)
    
    # Вывод статистики
    print_statistics(pairs, len(df))
    
    # Сохранение результатов
    print("\n" + "="*70)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*70)
    
    # Сохраняем в разные форматы
    csv_main, csv_summary, csv_stats = save_to_csv(pairs)
    json_data = save_to_json(pairs)
    
    # Используем альтернативу Excel (несколько CSV)
    save_to_excel_alternative(pairs)
    
    # Дополнительно: показываем примеры
    print("\n" + "="*70)
    print("ПРИМЕРЫ НАЙДЕННЫХ ПАР:")
    print("="*70)
    
    example_pairs = pairs[:5]  # Первые 5 пар
    for i, pair in enumerate(example_pairs, 1):
        print(f"\n{i}. БАЗОВЫЙ: {pair['base_verb']} (частота: {pair['base_frequency']})")
        print("   ПРОИЗВОДНЫЕ:")
        for j, deriv in enumerate(pair['derivatives'][:5], 1):
            print(f"   {j:2}. {deriv['prefix']:5} → {deriv['verb']:20} (частота: {deriv['frequency']})")
        if len(pair['derivatives']) > 5:
            print(f"   ... и еще {len(pair['derivatives']) - 5}")
    
    print("\n" + "="*70)
    print("✓ Анализ завершен! Созданы файлы:")
    print("  1. CSV с подробными данными")
    print("  2. CSV со сводной таблицей") 
    print("  3. CSV со статистикой по приставкам")
    print("  4. JSON с полными данными")
    print("  5. Папка с несколькими CSV файлами (альтернатива Excel)")
    print("="*70)

if __name__ == "__main__":
    main()