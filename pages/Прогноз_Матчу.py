import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="Прогноз Матчу", layout="wide")

st.title(" Модуль 1: ML-Прогнозування та Порівняння моделей")
st.markdown("---")

def calculate_elo(df, k=20):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    elo = {team: 1500 for team in teams}
    
    home_elo = []
    away_elo = []
    for _, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        
        home_elo.append(elo[h])
        away_elo.append(elo[a])
        
        expected_home = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        
        if row['FTR'] == 'H':
            score = 1
        elif row['FTR'] == 'D':
            score = 0.5
        else:
            score = 0
        
        elo[h] += k * (score - expected_home)
        elo[a] += k * ((1 - score) - (1 - expected_home))
    
    df['Home_ELO'] = home_elo
    df['Away_ELO'] = away_elo
    df['Home_ELO'] = (df['Home_ELO'] - 1500) / 400
    df['Away_ELO'] = (df['Away_ELO'] - 1500) / 400
    return df

@st.cache_data
def get_data():
    df = pd.read_csv('epl.csv')

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    def get_season(date):
        if date.month >= 7:
            return f"{date.year}/{date.year + 1}"
        else:
            return f"{date.year - 1}/{date.year}"

    df['Season'] = df['Date'].apply(get_season)
    df['Season_Match_Number'] = df.groupby('Season').cumcount()
    df['Round'] = (df['Season_Match_Number'] // 10) + 1

    df = calculate_elo(df)
    df['Odds_Diff'] = df['B365H'] - df['B365A']
    st.subheader("📊 Попередній перегляд даних")
    st.dataframe(df.head(10))

    def get_team_stats(team, current_index, data):
        past = data.loc[:current_index-1]
        team_matches = past[(past['HomeTeam'] == team) | (past['AwayTeam'] == team)]
        
        matches_3 = team_matches.tail(3)
        matches_5 = team_matches.tail(5)
        
        def calc_stats(matches):
            points, goals_scored, goals_conceded = 0, 0, 0
            for _, match in matches.iterrows():
                if match['HomeTeam'] == team:
                    goals_scored += match['FTHG']
                    goals_conceded += match['FTAG']
                    if match['FTR'] == 'H': points += 3
                    elif match['FTR'] == 'D': points += 1
                else:
                    goals_scored += match['FTAG']
                    goals_conceded += match['FTHG']
                    if match['FTR'] == 'A': points += 3
                    elif match['FTR'] == 'D': points += 1
            return points, goals_scored, goals_conceded
        
        p3, gs3, gc3 = calc_stats(matches_3)
        p5, gs5, gc5 = calc_stats(matches_5)
        
        return pd.Series([p3, gs3, gc3, p5, gs5, gc5])

    df[['Home_Form_3', 'Home_Attack_3', 'Home_Defense_3', 'Home_Form_5', 'Home_Attack_5', 'Home_Defense_5']] = df.apply(lambda row: get_team_stats(row['HomeTeam'], row.name, df), axis=1)
    df[['Away_Form_3', 'Away_Attack_3', 'Away_Defense_3', 'Away_Form_5', 'Away_Attack_5', 'Away_Defense_5']] = df.apply(lambda row: get_team_stats(row['AwayTeam'], row.name, df), axis=1)
    
    df['Form_Diff_3'] = df['Home_Form_3'] - df['Away_Form_3']
    df['Form_Diff_5'] = df['Home_Form_5'] - df['Away_Form_5']
    
    df['Attack_Diff_3'] = df['Home_Attack_3'] - df['Away_Attack_3']
    df['Attack_Diff_5'] = df['Home_Attack_5'] - df['Away_Attack_5']
    
    df['Defense_Diff_3'] = df['Home_Defense_3'] - df['Away_Defense_3']
    df['Defense_Diff_5'] = df['Home_Defense_5'] - df['Away_Defense_5']
    df['Goal_Diff_3'] = df['Attack_Diff_3'] - df['Defense_Diff_3']
    df['Goal_Diff_5'] = df['Attack_Diff_5'] - df['Defense_Diff_5']
    def count_team_matches_in_season_before(team, current_index, season, data):
        past = data.loc[:current_index-1]
        past = past[past['Season'] == season]
        team_matches = past[(past['HomeTeam'] == team) | (past['AwayTeam'] == team)]
        return len(team_matches)

    df['Home_Season_Matches_Before'] = df.apply(
        lambda row: count_team_matches_in_season_before(row['HomeTeam'], row.name, row['Season'], df),
        axis=1
    )

    df['Away_Season_Matches_Before'] = df.apply(
        lambda row: count_team_matches_in_season_before(row['AwayTeam'], row.name, row['Season'], df),
        axis=1
    )
    df['Has_Enough_History'] = (
        (df['Home_Season_Matches_Before'] >= 5) &
        (df['Away_Season_Matches_Before'] >= 5)
    )
    
    st.subheader("📊 Попередній перегляд оброблених даних")
    st.dataframe(df.tail(10))
    return df.dropna()

with st.spinner("🧠 Ініціалізація ядра та розрахунок спортивних метрик..."):
    df = get_data()

# Набори ознак
base_features = [
    'HomeTeam', 'AwayTeam',
    'Home_Form_3', 'Away_Form_3',
    'Home_Form_5', 'Home_Attack_5', 'Home_Defense_5',
    'Away_Form_5', 'Away_Attack_5', 'Away_Defense_5',
    'Attack_Diff_3', 'Attack_Diff_5',
    'Defense_Diff_3', 'Defense_Diff_5',
    'Form_Diff_3', 'Form_Diff_5',
    'Home_ELO', 'Away_ELO',
    'Goal_Diff_3', 'Goal_Diff_5'
]

market_features = ['B365H', 'B365D', 'B365A', 'Odds_Diff']


col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Тренування ШІ")
    st.info("Система навчає та порівнює 5 моделей машинного навчання на підготовлених спортивних ознаках.")
    
    

    st.markdown("---")
    st.markdown("### 🔁 Прогнозування на тестових даних")

    wf_model_name = st.selectbox(
    "Оберіть модель для walk-forward тестування:",
    [
        "Логістична Регресія",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "Neural Network"
    ],
    key="wf_model_select")
    wf_data_mode = st.radio(
    "Ознаки для walk-forward тестування:",
    [
        "Чиста аналітика без букмекерів",
        "Синтезована модель з букмекерами"
    ],
    index=0,
    key="wf_data_mode")
    wf_train_history_mode = st.radio(
    "Які матчі використовувати для навчання?",
    [
        "Тільки поточний сезон до прогнозованого туру",
        "Попередній сезон + поточний сезон до прогнозованого туру",
        "Усі попередні сезони + поточний сезон до прогнозованого туру"
    ],
    index=0,
    key="wf_train_history_mode")
    wf_task_type = st.radio(
    "Тип задачі для walk-forward тестування:",
    [
        "Мультикласова (П1, Х, П2)",
        "Бінарна (П1 проти Х/П2)"
    ],
    horizontal=True,
    key="wf_task_type")

    available_seasons = sorted(df['Season'].unique())

    if len(available_seasons) > 1:
        default_season_index = len(available_seasons) - 1
    else:
        default_season_index = 0

    wf_season = st.selectbox(
        "Оберіть сезон для покрокового прогнозування:",
        available_seasons,
        index=default_season_index,
        key="wf_season_select"
    )

    max_round = int(df[df['Season'] == wf_season]['Round'].max())

    wf_start_round = st.number_input(
        "Початковий тур для перевірки:",
        min_value=6,
        max_value=max_round,
        value=min(6, max_round),
        step=1,
        key="wf_start_round"
    )

    wf_end_round = st.number_input(
        "Кінцевий тур для перевірки:",
        min_value=int(wf_start_round),
        max_value=max_round,
        value=max_round,
        step=1,
        key="wf_end_round"
    )

    if st.button("🔁 Запустити прогнозування по турах", type="primary", key="wf_btn"):

        with st.spinner("Виконується послідовне тестування: навчання → прогноз туру → перевірка результату..."):

            y_all = df['FTR'].copy()

            if "Бінарна" in wf_task_type:
                y_all = y_all.replace({'H': 'H', 'D': 'Not_H', 'A': 'Not_H'})
                class_order = ['H', 'Not_H']
            else:
                class_order = ['H', 'D', 'A']

            if wf_data_mode == "Чиста аналітика без букмекерів":
                wf_features = base_features
            else:
                wf_features = base_features + market_features

            X_text_all = df[wf_features]
            X_all = pd.get_dummies(X_text_all, dtype=int)

            round_results = []
            match_results = []

            all_true = []
            all_pred = []
            all_prob = []
            test_rounds = range(int(wf_start_round), int(wf_end_round) + 1)

            season_order = (
                df.groupby('Season')['Date']
                .min()
                .sort_values()
                .index
                .tolist())
            test_rounds = range(int(wf_start_round), int(wf_end_round) + 1)
            def align_probabilities(prob_array, model_classes, target_classes):
                aligned = np.zeros((prob_array.shape[0], len(target_classes)))

                for i, cls in enumerate(model_classes):
                    if cls in target_classes:
                        target_idx = target_classes.index(cls)
                        aligned[:, target_idx] = prob_array[:, i]

                return aligned
        
            for round_num in test_rounds:

                current_round_df = df[
                    (df['Season'] == wf_season) &
                    (df['Round'] == round_num) &
                    (df['Has_Enough_History'])
                ]

                if current_round_df.empty:
                    continue

                round_start_date = current_round_df['Date'].min()

                current_season_train = df[
                    (df['Season'] == wf_season) &
                    (df['Round'] < round_num) &
                    (df['Has_Enough_History'])
                ]

                if wf_train_history_mode == "Тільки поточний сезон до прогнозованого туру":
                    train_df = current_season_train

                elif wf_train_history_mode == "Попередній сезон + поточний сезон до прогнозованого туру":
                    current_season_position = season_order.index(wf_season)

                    if current_season_position == 0:
                        train_df = current_season_train
                    else:
                        previous_season = season_order[current_season_position - 1]

                        previous_season_train = df[
                            (df['Season'] == previous_season) &
                            (df['Has_Enough_History'])
                        ]

                        train_df = pd.concat([previous_season_train, current_season_train])

                else:
                    train_df = df[
                        (df['Date'] < round_start_date) &
                        (df['Has_Enough_History'])
                    ]

                if len(train_df) < 30:
                    continue

                train_idx = train_df.index
                test_idx = current_round_df.index

                X_train_wf = X_all.loc[train_idx]
                X_test_wf = X_all.loc[test_idx]

                y_train_wf = y_all.loc[train_idx]
                y_test_wf = y_all.loc[test_idx]
                bookmaker_pred = current_round_df[['B365H', 'B365D', 'B365A']].idxmin(axis=1)

                bookmaker_pred = bookmaker_pred.replace({
                    'B365H': 'H',
                    'B365D': 'D',
                    'B365A': 'A'
                })

                # задача бінарна, об'єднуємо D та A у Not_H
                if "Бінарна" in wf_task_type:
                    bookmaker_pred = bookmaker_pred.replace({
                        'D': 'Not_H',
                        'A': 'Not_H'
                    })

                bookmaker_acc = accuracy_score(y_test_wf.values, bookmaker_pred.values)
                wf_scaler = StandardScaler()
                X_train_wf_scaled = wf_scaler.fit_transform(X_train_wf)
                X_test_wf_scaled = wf_scaler.transform(X_test_wf)

                best_params_wf = "-"

                if wf_model_name == "Логістична Регресія":
                    lr_params = {
                        'C': [0.001, 0.01, 0.1],
                        'solver': ['lbfgs', 'liblinear']
                    }

                    grid = GridSearchCV(
                        LogisticRegression(max_iter=1000, class_weight='balanced'),
                        lr_params,
                        cv=3,
                        n_jobs=-1
                    )

                    grid.fit(X_train_wf_scaled, y_train_wf)
                    model = grid.best_estimator_
                    y_pred_wf = model.predict(X_test_wf_scaled)
                    best_params_wf = grid.best_params_
                    y_prob_raw = model.predict_proba(X_test_wf_scaled)
                    prob_classes = list(model.classes_)

                elif wf_model_name == "Random Forest":
                    rf_params = {
                        'n_estimators': [100, 200],
                        'max_depth': [5, 10, 15],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }

                    grid = GridSearchCV(
                        RandomForestClassifier(random_state=42, class_weight='balanced'),
                        rf_params,
                        cv=3,
                        n_jobs=-1
                    )

                    grid.fit(X_train_wf_scaled, y_train_wf)
                    model = grid.best_estimator_
                    y_pred_wf = model.predict(X_test_wf_scaled)
                    best_params_wf = grid.best_params_
                    y_prob_raw = model.predict_proba(X_test_wf_scaled)
                    prob_classes = list(model.classes_)
                    

                elif wf_model_name == "Gradient Boosting":
                    gb_params = {
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 1.0]
                    }

                    grid = GridSearchCV(
                        GradientBoostingClassifier(n_estimators=100, random_state=42),
                        gb_params,
                        cv=3,
                        n_jobs=-1
                    )

                    grid.fit(X_train_wf_scaled, y_train_wf)
                    model = grid.best_estimator_
                    y_pred_wf = model.predict(X_test_wf_scaled)
                    best_params_wf = grid.best_params_
                    y_prob_raw = model.predict_proba(X_test_wf_scaled)
                    prob_classes = list(model.classes_)

                elif wf_model_name == "XGBoost":
                    le_wf = LabelEncoder()
                    y_train_encoded = le_wf.fit_transform(y_train_wf)

                    eval_metric = 'logloss' if "Бінарна" in wf_task_type else 'mlogloss'

                    xgb_params = {
                        'learning_rate': [0.01, 0.05],
                        'max_depth': [3, 4, 5],
                        'n_estimators': [200, 300],
                        'subsample': [0.8],
                        'colsample_bytree': [0.6, 0.8],
                        'gamma': [0, 0.1, 0.3],
                        'reg_lambda': [10, 50, 100]
                    }

                    grid = GridSearchCV(
                        XGBClassifier(
                            use_label_encoder=False,
                            eval_metric=eval_metric,
                            random_state=42
                        ),
                        xgb_params,
                        cv=3,
                        n_jobs=-1
                    )

                    grid.fit(X_train_wf_scaled, y_train_encoded)
                    model = grid.best_estimator_

                    y_pred_encoded = model.predict(X_test_wf_scaled)
                    y_pred_wf = le_wf.inverse_transform(y_pred_encoded)
                    best_params_wf = grid.best_params_
                    y_prob_raw = model.predict_proba(X_test_wf_scaled)
                    prob_classes = list(le_wf.classes_)

                elif wf_model_name == "Neural Network":
                    mlp_params = {
                        'hidden_layer_sizes': [(50,), (100, 50), (100, 100, 50)],
                        'alpha': [0.1, 1.0, 5.0],
                        'activation': ['relu', 'tanh']
                    }

                    grid = GridSearchCV(
                        MLPClassifier(max_iter=1000, random_state=42),
                        mlp_params,
                        cv=3,
                        n_jobs=-1
                    )

                    grid.fit(X_train_wf_scaled, y_train_wf)
                    model = grid.best_estimator_
                    y_pred_wf = model.predict(X_test_wf_scaled)
                    best_params_wf = grid.best_params_
                    y_prob_raw = model.predict_proba(X_test_wf_scaled)
                    prob_classes = list(model.classes_)

                # Метрики по туру 
                y_prob_aligned = align_probabilities(
                    y_prob_raw,
                    prob_classes,
                    class_order
                )

                correct_count = int((y_pred_wf == y_test_wf.values).sum())
                total_count = len(y_test_wf)
                round_acc = correct_count / total_count

                round_log_loss = log_loss(
                    y_test_wf,
                    y_prob_aligned,
                    labels=class_order
                )

                round_results.append({
                    'Сезон': wf_season,
                    'Тур': round_num,
                    'Модель': wf_model_name,
                    'Тип задачі': wf_task_type,
                    'Ознаки': wf_data_mode,
                    'Історія навчання': wf_train_history_mode,
                    'Матчів': total_count,
                    'Bookmaker baseline': round(bookmaker_acc * 100, 2),
                    'Різниця з baseline': round((round_acc - bookmaker_acc) * 100, 2),
                    'Правильно': correct_count,
                    'Accuracy': round(round_acc * 100, 2),
                    'Log-Loss': round(round_log_loss, 3),
                    'Кращі параметри': str(best_params_wf)
                })

                all_true.extend(y_test_wf.values)
                all_pred.extend(y_pred_wf)
                all_prob.extend(y_prob_aligned.tolist())

                # --- Детальні результати по матчах ---
                for idx, true_result, pred_result in zip(test_idx, y_test_wf.values, y_pred_wf):
                    row = df.loc[idx]

                    match_results.append({
                        'Сезон': row['Season'],
                        'Тур': round_num,
                        'Дата': row['Date'].date(),
                        'Матч': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                        'Факт': true_result,
                        'Прогноз': pred_result,
                        'Вірно': '✅' if true_result == pred_result else '❌'
                    })

            if len(round_results) == 0:
                st.error("Не вдалося сформувати результати. Можливо, для вибраного сезону або туру недостатньо історичних матчів.")
            else:
                wf_round_df = pd.DataFrame(round_results)
                wf_match_df = pd.DataFrame(match_results)
                mean_model_acc_by_round = wf_round_df['Accuracy'].mean()
                mean_bookmaker_acc_by_round = wf_round_df['Bookmaker baseline'].mean()
                mean_diff_with_baseline = wf_round_df['Різниця з baseline'].mean()

                overall_acc = accuracy_score(all_true, all_pred)
                overall_log_loss = log_loss(
                    all_true,
                    np.array(all_prob),
                    labels=class_order
                )
                round_mean_acc = wf_round_df['Accuracy'].mean()

                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_true,
                    all_pred,
                    average='macro',
                    zero_division=0
                )

                st.session_state['wf_round_df'] = wf_round_df
                st.session_state['wf_match_df'] = wf_match_df
                st.session_state['wf_summary'] = {
                    'Модель': wf_model_name,
                    'Ознаки': wf_data_mode,
                    'Тип задачі': wf_task_type,
                    'Історія навчання': wf_train_history_mode,
                    'Сезон': wf_season,
                    'Турів перевірено': len(wf_round_df),
                    'Матчів перевірено': len(wf_match_df),
                    'Accuracy': round(mean_model_acc_by_round, 2),
                    'Bookmaker baseline': round(mean_bookmaker_acc_by_round, 2),
                    'Різниця з baseline': round(mean_diff_with_baseline, 2),
                    'Accuracy по матчах': round(overall_acc * 100, 2),
                    'Log-Loss': round(overall_log_loss, 3),
                    'Precision': round(precision, 3),
                    'Recall': round(recall, 3),
                    'F1-score': round(f1, 3)
                }
      

                experiment_summary = st.session_state['wf_summary'].copy()

                experiment_summary['Початковий тур'] = int(wf_start_round)
                experiment_summary['Кінцевий тур'] = int(wf_end_round)

                if 'wf_all_experiments' not in st.session_state:
                    st.session_state['wf_all_experiments'] = []

                st.session_state['wf_all_experiments'].append(experiment_summary)


                st.session_state['wf_true'] = all_true
                st.session_state['wf_pred'] = all_pred
                st.session_state['wf_class_order'] = class_order
        
                st.session_state['wf_model'] = model
                st.session_state['wf_scaler'] = wf_scaler
                st.session_state['wf_model_columns'] = X_train_wf.columns.tolist()
                st.session_state['wf_prob_classes'] = prob_classes
                st.session_state['wf_feature_names'] = X_train_wf.columns.tolist()
                st.session_state['wf_features'] = wf_features
                st.session_state['wf_model_name_used'] = wf_model_name
                st.session_state['wf_data_mode_used'] = wf_data_mode
                st.session_state['wf_task_type_used'] = wf_task_type

                st.success("Walk-forward експеримент завершено!")

                

with col2:
    st.header("📊 Результати walk-forward тестування")

    if 'wf_summary' not in st.session_state:
        st.warning("👈 Спочатку запустіть walk-forward тестування по турах.")
    else:
        summary = st.session_state['wf_summary']

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Модель", summary['Модель'])
        c2.metric("Сезон", summary['Сезон'])
        c3.metric("Матчів", summary['Матчів перевірено'])
        c4.metric("Accuracy", f"{summary['Accuracy']}%")
        c5.metric("Log-Loss", summary['Log-Loss'])

        tab_metrics, tab_rounds, tab_cm, tab_feat, tab_whatif = st.tabs([
            "📊 Метрики",
            "📋 Тури та матчі",
            "🧮 Матриця помилок",
            "🧬 Важливість ознак",
            "🎛️ What-If"
        ])

        with tab_metrics:
            st.markdown("### Загальні метрики експерименту")
            metrics_df = pd.DataFrame([summary])
            st.dataframe(metrics_df, width="stretch")

            if 'wf_all_experiments' in st.session_state and len(st.session_state['wf_all_experiments']) > 0:
                st.markdown("### Зведена таблиця всіх walk-forward запусків")

                all_exp_df = pd.DataFrame(st.session_state['wf_all_experiments'])

                all_exp_df['Діапазон турів'] = (
                    all_exp_df['Початковий тур'].astype(str)
                    + "–"
                    + all_exp_df['Кінцевий тур'].astype(str)
                )

                show_cols = [
                    'Сезон',
                    'Діапазон турів',
                    'Модель',
                    'Тип задачі',
                    'Ознаки',
                    'Історія навчання',
                    'Матчів перевірено',
                    'Accuracy',
                    'Bookmaker baseline',
                    'Різниця з baseline',
                    'Log-Loss',
                    'Precision',
                    'Recall',
                    'F1-score'
                ]

                existing_cols = [col for col in show_cols if col in all_exp_df.columns]
                st.dataframe(all_exp_df[existing_cols], width="stretch")

        with tab_rounds:
            st.markdown("### Результати по турах")

            if 'wf_round_df' in st.session_state:
                st.dataframe(st.session_state['wf_round_df'], width="stretch")

            st.markdown("### Детальні результати по матчах")

            if 'wf_match_df' in st.session_state:
                st.dataframe(st.session_state['wf_match_df'], width="stretch")

        with tab_cm:
            st.markdown("### Матриця помилок walk-forward моделі")

            labels = st.session_state['wf_class_order']
            cm = confusion_matrix(
                st.session_state['wf_true'],
                st.session_state['wf_pred'],
                labels=labels
            )

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax
            )

            ax.set_xlabel("Прогноз моделі")
            ax.set_ylabel("Фактичний результат")
            ax.set_title(f"Матриця помилок: {summary['Модель']}")
            st.pyplot(fig)
            plt.close(fig)

        with tab_feat:
            st.markdown("### Важливість ознак walk-forward моделі")

            model = st.session_state.get('wf_model')
            feature_names = st.session_state.get('wf_feature_names')

            if model is None or feature_names is None:
                st.info("Модель або список ознак ще не збережені.")
            elif hasattr(model, 'feature_importances_'):
                vals = model.feature_importances_

                df_feat = pd.DataFrame({
                    'Ознака': feature_names,
                    'Сила': vals
                })

                df_feat = df_feat[~df_feat['Ознака'].str.contains('HomeTeam_|AwayTeam_', regex=True)]
                df_feat = df_feat.sort_values(by='Сила', ascending=False).head(15)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Сила', y='Ознака', data=df_feat, ax=ax)
                ax.set_xlabel("Відносна важливість ознаки")
                ax.set_ylabel("")
                ax.set_title(f"Топ-15 ознак: {summary['Модель']}")
                st.pyplot(fig)
                plt.close(fig)

            elif hasattr(model, 'coef_'):
                vals = np.abs(model.coef_).mean(axis=0)

                df_feat = pd.DataFrame({
                    'Ознака': feature_names,
                    'Сила': vals
                })

                df_feat = df_feat[~df_feat['Ознака'].str.contains('HomeTeam_|AwayTeam_', regex=True)]
                df_feat = df_feat.sort_values(by='Сила', ascending=False).head(15)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Сила', y='Ознака', data=df_feat, ax=ax)
                ax.set_xlabel("Середній модуль коефіцієнта")
                ax.set_ylabel("")
                ax.set_title(f"Топ-15 ознак: {summary['Модель']}")
                st.pyplot(fig)
                plt.close(fig)

            else:
                st.info("Для цієї моделі графік важливості ознак недоступний.")

        with tab_whatif:
            st.markdown("### 🎛️ Ручне моделювання What-If")
            st.write("Модуль використовує останню модель, навчену через walk-forward тестування.")

            final_model = st.session_state['wf_model']
            wf_scaler = st.session_state['wf_scaler']
            model_columns = st.session_state['wf_model_columns']
            prob_classes = st.session_state['wf_prob_classes']
            wf_data_mode_used = st.session_state['wf_data_mode_used']
            wf_task_type_used = st.session_state['wf_task_type_used']

            use_market = "букмекерами" in wf_data_mode_used.lower()

            if use_market:
                col_w1, col_w2, col_w3 = st.columns(3)
            else:
                col_w1, col_w2 = st.columns(2)

            with col_w1:
                st.markdown("#### Господарі")
                w_h_team = st.selectbox("Команда господарів:", sorted(df['HomeTeam'].unique()), key='wf_w_h')
                w_h_form = st.slider("Очки господарів за останні 3 матчі:", 0, 9, 4, key='wf_whf')
                w_h_atk = st.slider("Забиті голи господарів:", 0, 15, 4, key='wf_wha')
                w_h_def = st.slider("Пропущені голи господарів:", 0, 15, 3, key='wf_whd')

            with col_w2:
                st.markdown("#### Гості")
                w_a_team = st.selectbox("Команда гостей:", sorted(df['AwayTeam'].unique()), index=1, key='wf_w_a')
                w_a_form = st.slider("Очки гостей за останні 3 матчі:", 0, 9, 4, key='wf_waf')
                w_a_atk = st.slider("Забиті голи гостей:", 0, 15, 3, key='wf_waa')
                w_a_def = st.slider("Пропущені голи гостей:", 0, 15, 4, key='wf_wad')

            if use_market:
                with col_w3:
                    st.markdown("#### Букмекери")
                    w_b365h = st.number_input("П1:", min_value=1.01, value=2.50, key='wf_bh')
                    w_b365d = st.number_input("Х:", min_value=1.01, value=3.20, key='wf_bd')
                    w_b365a = st.number_input("П2:", min_value=1.01, value=2.80, key='wf_ba')
            else:
                w_b365h, w_b365d, w_b365a = 0, 0, 0

            def get_latest_elo(team_name):
                team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]
                if team_matches.empty:
                    return 0

                last_match = team_matches.iloc[-1]

                if last_match['HomeTeam'] == team_name:
                    return last_match['Home_ELO']
                return last_match['Away_ELO']

            if st.button("Симулювати матч", type="primary", key="wf_sim_btn"):
                if w_h_team == w_a_team:
                    st.error("Команда не може грати сама з собою.")
                else:
                    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

                    # One-Hot Encoding команд
                    home_col = f"HomeTeam_{w_h_team}"
                    away_col = f"AwayTeam_{w_a_team}"

                    if home_col in input_df.columns:
                        input_df[home_col] = 1

                    if away_col in input_df.columns:
                        input_df[away_col] = 1

                    # Коротка форма
                    if 'Home_Form_3' in input_df.columns:
                        input_df['Home_Form_3'] = w_h_form
                    if 'Away_Form_3' in input_df.columns:
                        input_df['Away_Form_3'] = w_a_form

                    if 'Home_Attack_3' in input_df.columns:
                        input_df['Home_Attack_3'] = w_h_atk
                    if 'Away_Attack_3' in input_df.columns:
                        input_df['Away_Attack_3'] = w_a_atk

                    if 'Home_Defense_3' in input_df.columns:
                        input_df['Home_Defense_3'] = w_h_def
                    if 'Away_Defense_3' in input_df.columns:
                        input_df['Away_Defense_3'] = w_a_def

                    # Умовне масштабування 3 матчів до 5 матчів
                    scale_factor = 5 / 3

                    if 'Home_Form_5' in input_df.columns:
                        input_df['Home_Form_5'] = int(w_h_form * scale_factor)
                    if 'Away_Form_5' in input_df.columns:
                        input_df['Away_Form_5'] = int(w_a_form * scale_factor)

                    if 'Home_Attack_5' in input_df.columns:
                        input_df['Home_Attack_5'] = int(w_h_atk * scale_factor)
                    if 'Away_Attack_5' in input_df.columns:
                        input_df['Away_Attack_5'] = int(w_a_atk * scale_factor)

                    if 'Home_Defense_5' in input_df.columns:
                        input_df['Home_Defense_5'] = int(w_h_def * scale_factor)
                    if 'Away_Defense_5' in input_df.columns:
                        input_df['Away_Defense_5'] = int(w_a_def * scale_factor)

                    # Різницеві ознаки
                    if 'Form_Diff_3' in input_df.columns:
                        input_df['Form_Diff_3'] = w_h_form - w_a_form
                    if 'Form_Diff_5' in input_df.columns:
                        input_df['Form_Diff_5'] = int(w_h_form * scale_factor) - int(w_a_form * scale_factor)

                    if 'Attack_Diff_3' in input_df.columns:
                        input_df['Attack_Diff_3'] = w_h_atk - w_a_atk
                    if 'Attack_Diff_5' in input_df.columns:
                        input_df['Attack_Diff_5'] = int(w_h_atk * scale_factor) - int(w_a_atk * scale_factor)

                    if 'Defense_Diff_3' in input_df.columns:
                        input_df['Defense_Diff_3'] = w_h_def - w_a_def
                    if 'Defense_Diff_5' in input_df.columns:
                        input_df['Defense_Diff_5'] = int(w_h_def * scale_factor) - int(w_a_def * scale_factor)

                    # Goal_Diff
                    home_gd_3 = w_h_atk - w_h_def
                    away_gd_3 = w_a_atk - w_a_def

                    if 'Goal_Diff_3' in input_df.columns:
                        input_df['Goal_Diff_3'] = home_gd_3 - away_gd_3

                    if 'Goal_Diff_5' in input_df.columns:
                        home_gd_5 = int(home_gd_3 * scale_factor)
                        away_gd_5 = int(away_gd_3 * scale_factor)
                        input_df['Goal_Diff_5'] = home_gd_5 - away_gd_5

                    # ELO
                    if 'Home_ELO' in input_df.columns:
                        input_df['Home_ELO'] = get_latest_elo(w_h_team)

                    if 'Away_ELO' in input_df.columns:
                        input_df['Away_ELO'] = get_latest_elo(w_a_team)

                    # Букмекерські ознаки
                    if 'B365H' in input_df.columns:
                        input_df['B365H'] = w_b365h
                    if 'B365D' in input_df.columns:
                        input_df['B365D'] = w_b365d
                    if 'B365A' in input_df.columns:
                        input_df['B365A'] = w_b365a
                    if 'Odds_Diff' in input_df.columns:
                        input_df['Odds_Diff'] = w_b365h - w_b365a

                    input_scaled = wf_scaler.transform(input_df)
                    proba = final_model.predict_proba(input_scaled)[0]
                    prob_dict = dict(zip(prob_classes, proba))

                    st.success("Симуляцію завершено.")

                    st.write(f"**Перемога {w_h_team} (П1):** {prob_dict.get('H', 0) * 100:.1f}%")

                    if "Бінарна" in wf_task_type_used:
                        st.write(f"**Втрата очок (Х або П2):** {prob_dict.get('Not_H', 0) * 100:.1f}%")
                    else:
                        st.write(f"**Нічия (Х):** {prob_dict.get('D', 0) * 100:.1f}%")
                        st.write(f"**Перемога {w_a_team} (П2):** {prob_dict.get('A', 0) * 100:.1f}%")
