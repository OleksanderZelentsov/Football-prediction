import streamlit as st
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from statsbombpy import sb

st.set_page_config(page_title="Тактичний Аналіз", layout="wide")

st.title(" Модуль 2: Динамічний тактичний аналіз (DSS)")
st.markdown("---")

@st.cache_data
def load_match_data():
    return sb.events(match_id=3869151) # ЧС-2022 Фінал

st.sidebar.header("🎛️ Панель керування")
uploaded_file = st.sidebar.file_uploader("📂 Завантажити лог матчу (JSON)", type="json")

with st.spinner("📡 Обробка датасету..."):
    if uploaded_file is not None:
        events = pd.read_json(uploaded_file)
    else:
        events = load_match_data()

teams = events['team'].dropna().unique()
selected_team = st.sidebar.selectbox("1. Аналізована команда (Наші):", teams)
opponent_team = [t for t in teams if t != selected_team][0] 

team_players = sorted(events[events['team'] == selected_team]['player'].dropna().unique())
selected_player = st.sidebar.selectbox("2. Гравець для Теплової карти:", team_players)

passes = events[(events['type'] == 'Pass') & (events['team'] == selected_team) & (events['pass_outcome'].isnull())].dropna(subset=['location', 'pass_end_location'])
passes[['x', 'y']] = passes['location'].apply(pd.Series)
passes[['end_x', 'end_y']] = passes['pass_end_location'].apply(pd.Series)
passes['player_name'] = passes['player'].apply(lambda x: str(x).split()[-1] if pd.notnull(x) else 'Unknown')
passes['pass_recipient_name'] = passes['pass_recipient'].apply(lambda x: str(x).split()[-1] if pd.notnull(x) else 'Unknown')

avg_locs = passes.groupby('player_name').agg({'x': ['mean'], 'y': ['mean', 'count']})
avg_locs.columns = ['x', 'y', 'count']
avg_locs = avg_locs[avg_locs['count'] > 5] 

# Основа
tab1, tab2, tab3, tab4 = st.tabs(["↗️ Мережа передач", "🗺️ Зональна битва (Вороний)", f"🔥 Теплова карта", "🎯 Битва ударів (xG)"])

# Мережа передач
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        pass_between = passes.groupby(['player_name', 'pass_recipient_name']).size().reset_index(name='pass_count')
        pass_between = pass_between[pass_between['player_name'].isin(avg_locs.index) & pass_between['pass_recipient_name'].isin(avg_locs.index)]
        pass_between = pass_between[pass_between['pass_count'] > 3]

        pitch1 = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='#7a7a7a')
        fig1, ax1 = pitch1.draw(figsize=(10, 7))
        fig1.patch.set_facecolor('#1a1a1a')

        for index, row in pass_between.iterrows():
            p1, p2 = row['player_name'], row['pass_recipient_name']
            ax1.plot([avg_locs.loc[p1, 'x'], avg_locs.loc[p2, 'x']], 
                     [avg_locs.loc[p1, 'y'], avg_locs.loc[p2, 'y']], 
                     color='#e3c25b', alpha=0.7, lw=row['pass_count']*0.3)

        pitch1.scatter(avg_locs.x, avg_locs.y, s=avg_locs['count']*15, color='#ffffff', edgecolors='#e3c25b', linewidth=2, ax=ax1)
        for name, row in avg_locs.iterrows():
            pitch1.annotate(name, xy=(row.x, row.y - 3), c='white', va='center', ha='center', size=11, weight='bold', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.markdown("### 📖 Легенда:")
        st.caption("• **Точки:** Де гравець в середньому отримував м'яч.\n"
                   "• **Лінії:** Найчастіші маршрути пасів (чим товща, тим більше пасів).")
        st.markdown("---")
        
        st.subheader("📊 Аналіз Білд-апу")
        if not avg_locs.empty and not pass_between.empty:
            top_passer = avg_locs.sort_values(by='count', ascending=False).iloc[0].name
            top_link = pass_between.sort_values(by='pass_count', ascending=False).iloc[0]
            
            st.success(f"**🟢 Ключовий вузол:** {top_passer}\n\n**🟢 Головний вектор:** {top_link['player_name']} ➡️ {top_link['pass_recipient_name']}")
            st.warning(f"**🔴 Вразливість:** Перекриття гравця {top_passer} зламає ваш вихід з оборони.")
            
            st.info("**💡 Рекомендації штабу:**\n"
                    f"Додайте альтернативний вектор розіграшу через флангових захисників, щоб розтягнути пресинг суперника та зняти навантаження з {top_passer}.")

# Діаграма Вороного
with tab2:
    col3, col4 = st.columns([3, 1])
    with col3:
        passes_opp = events[(events['type'] == 'Pass') & (events['team'] == opponent_team) & (events['pass_outcome'].isnull())].dropna(subset=['location'])
        passes_opp[['x', 'y']] = passes_opp['location'].apply(pd.Series)
        passes_opp['player_name'] = passes_opp['player'].apply(lambda x: str(x).split()[-1] if pd.notnull(x) else 'Unknown')
        avg_locs_opp = passes_opp.groupby('player_name').agg({'x': ['mean'], 'y': ['mean', 'count']})
        avg_locs_opp.columns = ['x', 'y', 'count']
        avg_locs_opp = avg_locs_opp[avg_locs_opp['count'] > 5]
        
        avg_locs_opp['x'] = 120 - avg_locs_opp['x']
        avg_locs_opp['y'] = 80 - avg_locs_opp['y']

        df_all = pd.concat([avg_locs.assign(is_our=True), avg_locs_opp.assign(is_our=False)])

        pitch2 = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='#7a7a7a')
        fig2, ax2 = pitch2.draw(figsize=(12, 8))
        fig2.patch.set_facecolor('#1a1a1a')

        if len(df_all) > 10:
            t1, t2 = pitch2.voronoi(df_all.x, df_all.y, df_all.is_our)
            pitch2.polygon(t1, color='#43a1cf', alpha=0.4, edgecolor='white', linewidth=1, ax=ax2) # Наші (Сині)
            pitch2.polygon(t2, color='#e24a33', alpha=0.4, edgecolor='white', linewidth=1, ax=ax2) # Суперник (Червоні)

        pitch2.scatter(avg_locs.x, avg_locs.y, s=150, color='#43a1cf', edgecolors='white', linewidth=2, ax=ax2, label=selected_team)
        pitch2.scatter(avg_locs_opp.x, avg_locs_opp.y, s=150, color='#e24a33', edgecolors='white', linewidth=2, ax=ax2, label=opponent_team)

        for name, row in avg_locs.iterrows():
            pitch2.annotate(name, xy=(row.x, row.y - 3), c='white', va='center', ha='center', size=9, weight='bold', ax=ax2)
        for name, row in avg_locs_opp.iterrows():
            pitch2.annotate(name, xy=(row.x, row.y - 3), c='white', va='center', ha='center', size=9, weight='bold', ax=ax2)

        ax2.legend(loc='upper left', facecolor='#1a1a1a', labelcolor='white')
        st.pyplot(fig2)

    with col4:
        st.markdown("### 📖 Легенда:")
        st.caption("• **Сині зони:** Простір, який контролює наша команда.\n"
                   "• **Червоні зони:** Простір під контролем суперника.")
        st.markdown("---")
        
        st.subheader("🗺️ Аналіз Території")
        st.warning("**🔴 Ризик:** Подивіться на лінію зіткнення (де синій колір межує з червоним). Якщо великий червоний багатокутник суперника врізається в нашу половину поля — там у нас діра в опорній зоні.")
        
        st.info("**💡 Рекомендації штабу:**\n"
                "1. **Атака:** Шукайте гравця нашої команди, який знаходиться найближче до великого 'червоного' простору. Це ідеальна мішень для пасу в розріз.\n"
                "2. **Оборона:** Якщо суперник відтіснив нас до наших воріт (мало синього кольору), необхідно підняти лінію захисту на 10-15 метрів для створення штучного офсайду.")

# Теплова карта
with tab3:
    col5, col6 = st.columns([3, 1])
    with col5:
        player_events = events[events['player'] == selected_player].dropna(subset=['location'])
        if not player_events.empty:
            player_events[['x', 'y']] = player_events['location'].apply(pd.Series)
            pitch3 = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
            fig3, ax3 = pitch3.draw(figsize=(10, 7))
            fig3.patch.set_facecolor('#22312b')
            pitch3.kdeplot(player_events.x, player_events.y, ax=ax3, fill=True, levels=100, cmap='hot', alpha=0.6)
            st.pyplot(fig3)
        
    with col6:
        st.markdown("### 📖 Легенда:")
        st.caption("• **Яскраві плями:** Місця найчастіших дотиків гравця до м'яча.")
        st.markdown("---")
        
        st.subheader("🔥 Індивідуальний аналіз")
        st.warning("**🔴 Тактичний маркер:** Часто вінгери зміщуються в центр, залишаючи бровку порожньою. Чи відповідає пляма позиції гравця на папері?")
        
        st.info("**💡 Рекомендації штабу:**\n"
                "1. **Як захищатись:** Якщо ядро чітко в центрі — перекривайте опорну зону. Дозвольте йому йти у фланги, де він менш ефективний.\n"
                "2. **Як атакувати (для нашого гравця):** Просіть гравця періодично міняти фланг, щоб заплутати персонального опікуна.")

# Удари
with tab4:
    col7, col8 = st.columns([3, 1])
    with col7:
        shots_our = events[(events['type'] == 'Shot') & (events['team'] == selected_team)].dropna(subset=['location', 'shot_statsbomb_xg'])
        shots_opp = events[(events['type'] == 'Shot') & (events['team'] == opponent_team)].dropna(subset=['location', 'shot_statsbomb_xg'])

        pitch4 = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='#c7d5cc')
        fig4, ax4 = pitch4.draw(figsize=(12, 8))
        fig4.patch.set_facecolor('#1a1a1a')

        if not shots_our.empty:
            shots_our[['x', 'y']] = shots_our['location'].apply(pd.Series)
            shots_our['xg'] = shots_our['shot_statsbomb_xg']
            shots_our['is_goal'] = shots_our['shot_outcome'] == 'Goal'
            pitch4.scatter(shots_our[~shots_our['is_goal']].x, shots_our[~shots_our['is_goal']].y, s=shots_our[~shots_our['is_goal']]['xg']*2000, alpha=0.5, color='#43a1cf', edgecolors='white', ax=ax4)
            pitch4.scatter(shots_our[shots_our['is_goal']].x, shots_our[shots_our['is_goal']].y, s=shots_our[shots_our['is_goal']]['xg']*2000, alpha=0.9, color='#e3c25b', edgecolors='white', linewidth=2, ax=ax4)

        if not shots_opp.empty:
            shots_opp[['x', 'y']] = shots_opp['location'].apply(pd.Series)
            shots_opp['x'] = 120 - shots_opp['x'] 
            shots_opp['y'] = 80 - shots_opp['y']  
            shots_opp['xg'] = shots_opp['shot_statsbomb_xg']
            shots_opp['is_goal'] = shots_opp['shot_outcome'] == 'Goal'
            pitch4.scatter(shots_opp[~shots_opp['is_goal']].x, shots_opp[~shots_opp['is_goal']].y, s=shots_opp[~shots_opp['is_goal']]['xg']*2000, alpha=0.5, color='#e24a33', edgecolors='white', ax=ax4)
            pitch4.scatter(shots_opp[shots_opp['is_goal']].x, shots_opp[shots_opp['is_goal']].y, s=shots_opp[shots_opp['is_goal']]['xg']*2000, alpha=0.9, color='#e3c25b', edgecolors='white', linewidth=2, ax=ax4)

        ax4.text(30, -5, f"⬅️ Атакує {opponent_team} (Червоні)", color='#e24a33', ha='center', fontsize=14, weight='bold')
        ax4.text(90, -5, f"Атакує {selected_team} (Сині) ➡️", color='#43a1cf', ha='center', fontsize=14, weight='bold')

        st.pyplot(fig4)
            
    with col8:
        st.markdown("### 📖 Легенда:")
        st.caption("• **Розмір точки:** Ймовірність голу (xG).\n"
                   "• **Золоті кола:** Забиті голи.")
        st.markdown("---")
        
        st.subheader("🎯 Хто заслужив перемогу?")
        if not shots_our.empty and not shots_opp.empty:
            our_xg = shots_our['xg'].sum()
            opp_xg = shots_opp['xg'].sum()
            
            st.success(f"**{selected_team}:** {our_xg:.2f} xG (Голів: {len(shots_our[shots_our['is_goal']])})")
            st.error(f"**{opponent_team}:** {opp_xg:.2f} xG (Голів: {len(shots_opp[shots_opp['is_goal']])})")
            
            if our_xg > opp_xg + 0.5:
                st.write("📈 Наша команда повністю домінувала за гостротою моментів.")
            elif opp_xg > our_xg + 0.5:
                st.write("📉 Суперник створив більше небезпеки. Нам пощастило в обороні.")
            else:
                st.write("⚖️ Абсолютно рівна гра математично.")
                
            st.info("**💡 Рекомендації штабу:**\n"
                    "Якщо у суперника багато великих точок з меж нашого штрафного — проблема з опорними півзахисниками, які не блокують удари по центру.")