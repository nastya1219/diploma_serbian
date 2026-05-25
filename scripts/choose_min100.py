import pandas as pd

# Загружаем файл с суммарными частотами
input_file = 'serbian_verb_pairs_summed.csv'  # или ваш файл с результатом
df = pd.read_csv(input_file)

print(f"📊 Загружено {len(df)} пар глаголов из файла: {input_file}")

# Фильтруем: оставляем только пары, где ОБА глагола встречаются > 100 раз
df_filtered = df[(df['freq_base'] > 100) & (df['freq_prefixed'] > 100)]

print(f"\n🔍 Фильтрация: freq_base > 100 и freq_prefixed > 100")
print(f"   Осталось пар: {len(df_filtered)} из {len(df)}")
print(f"   Удалено пар: {len(df) - len(df_filtered)}")

# Сортируем по убыванию частоты префиксального глагола
df_filtered = df_filtered.sort_values(by='freq_prefixed', ascending=False)

# Сохраняем результат
output_file = 'serbian_verb_pairs_filtered_over_100.csv'
df_filtered.to_csv(output_file, index=False, encoding='utf-8')

print(f"\n✅ Отфильтрованные данные сохранены в: {output_file}")

# Статистика по удаленным парам
removed_df = df[~((df['freq_base'] > 100) & (df['freq_prefixed'] > 100))]
print(f"\n📈 Статистика удаленных пар:")
print(f"   С freq_base <= 100: {len(df[df['freq_base'] <= 100])}")
print(f"   С freq_prefixed <= 100: {len(df[df['freq_prefixed'] <= 100])}")

# Показываем примеры удаленных пар
if len(removed_df) > 0:
    print(f"\n🗑️  Примеры удаленных пар (первые 5):")
    for i, row in removed_df.head().iterrows():
        reason = []
        if row['freq_base'] <= 100:
            reason.append(f"freq_base={row['freq_base']}")
        if row['freq_prefixed'] <= 100:
            reason.append(f"freq_prefixed={row['freq_prefixed']}")
        print(f"   {row['base_verb']} → {row['prefixed_verb']}: {', '.join(reason)}")

# Показываем топ-10 самых частых пар после фильтрации
if len(df_filtered) > 0:
    print(f"\n🏆 ТОП-10 самых частых пар после фильтрации:")
    top_pairs = df_filtered.head(10)
    for i, (_, row) in enumerate(top_pairs.iterrows(), 1):
        print(f"  {i:2}. {row['base_verb']:15} → {row['prefixed_verb']:20} "
              f"({row['prefix']}): {row['freq_prefixed']:,}")
else:
    print("\n⚠️  После фильтрации не осталось ни одной пары!")

# Дополнительные варианты фильтрации (опционально)
print("\n⚙️  Дополнительные варианты фильтрации:")
print(f"1. Только базовые глаголы > 100: {len(df[df['freq_base'] > 100])} пар")
print(f"2. Только префиксальные глаголы > 100: {len(df[df['freq_prefixed'] > 100])} пар")
print(f"3. Любой из глаголов > 100: {len(df[(df['freq_base'] > 100) | (df['freq_prefixed'] > 100)])} пар")