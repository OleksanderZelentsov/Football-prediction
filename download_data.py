import pandas as pd

print("⏳ З'єднання з серверами football-data.co.uk...")

# Прямі посилання на останні 5 завершених сезонів АПЛ
urls = [
    'https://www.football-data.co.uk/mmz4281/2324/E0.csv', # Сезон 23/24
    'https://www.football-data.co.uk/mmz4281/2223/E0.csv', # Сезон 22/23
    'https://www.football-data.co.uk/mmz4281/2122/E0.csv', # Сезон 21/22
    'https://www.football-data.co.uk/mmz4281/2021/E0.csv', # Сезон 20/21
    'https://www.football-data.co.uk/mmz4281/1920/E0.csv'  # Сезон 19/20
]

dfs = []

for url in urls:
    season_name = url.split('/')[4]
    print(f"📥 Завантаження сезону {season_name[:2]}/{season_name[2:]}...")
    # Читаємо файл прямо з інтернету
    df = pd.read_csv(url)
    
    # Залишаємо тільки ті колонки, які потрібні для нашого ШІ (щоб не смітити в базі)
    cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
    # Якщо якихось колонок немає (буває в старих сезонах), беремо ті, що є
    cols_available = [c for c in cols_to_keep if c in df.columns]
    
    dfs.append(df[cols_available])

print("🔄 Об'єднання даних (Агрегація)...")
# Зліплюємо всі 5 сезонів в одну велику таблицю
final_df = pd.concat(dfs, ignore_index=True)

# Зберігаємо результат у файл epl.csv
final_df.to_csv('epl.csv', index=False)

print("=======================================")
print(f"✅ УСПІХ! Файл epl.csv створено.")
print(f"📊 Загальна кількість матчів у базі: {len(final_df)}")
print("=======================================")