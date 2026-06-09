"""
Football Analytics Pipeline
European Soccer Database (database.sqlite)
Tables: Match, Team, Team_Attributes, Player, Player_Attributes, League, Country
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/football/database.sqlite"


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_football_data():
    """Load tables from SQLite database."""
    conn = sqlite3.connect(DATA_PATH)

    matches = pd.read_sql("SELECT * FROM Match", conn)
    teams = pd.read_sql("SELECT * FROM Team", conn)
    leagues = pd.read_sql("SELECT * FROM League", conn)
    countries = pd.read_sql("SELECT * FROM Country", conn)

    # Team attributes (latest per team)
    team_attrs = pd.read_sql(
        "SELECT * FROM Team_Attributes ORDER BY date DESC", conn
    )

    conn.close()
    return matches, teams, leagues, countries, team_attrs


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────

def clean_football_data(matches, teams, leagues, countries, team_attrs):
    """Clean and merge football datasets."""
    # Keep relevant match columns
    core_cols = ['id', 'country_id', 'league_id', 'season', 'stage', 'date',
                 'home_team_api_id', 'away_team_api_id',
                 'home_team_goal', 'away_team_goal']
    available = [c for c in core_cols if c in matches.columns]
    df = matches[available].copy()
    df.dropna(subset=['home_team_goal', 'away_team_goal'], inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Outcome: H=home win, A=away win, D=draw
    df['result'] = np.where(df['home_team_goal'] > df['away_team_goal'], 'H',
                   np.where(df['home_team_goal'] < df['away_team_goal'], 'A', 'D'))
    df['result_enc'] = df['result'].map({'H': 2, 'D': 1, 'A': 0})

    # Merge team names
    team_map = teams.set_index('team_api_id')['team_long_name'].to_dict()
    team_short_map = teams.set_index('team_api_id')['team_short_name'].to_dict()
    df['home_team'] = df['home_team_api_id'].map(team_map)
    df['away_team'] = df['away_team_api_id'].map(team_map)
    df['home_team_short'] = df['home_team_api_id'].map(team_short_map)
    df['away_team_short'] = df['away_team_api_id'].map(team_short_map)

    # Merge league names
    league_map = leagues.set_index('id')['name'].to_dict()
    df['league'] = df['league_id'].map(league_map)

    # Latest team attributes
    latest_attrs = team_attrs.sort_values('date').groupby('team_api_id').last().reset_index()
    attr_cols = ['team_api_id', 'buildUpPlaySpeed', 'chanceCreationPassing',
                 'defencePressure', 'defenceAggression']
    attr_cols = [c for c in attr_cols if c in latest_attrs.columns]
    latest_attrs = latest_attrs[attr_cols]

    df = df.merge(latest_attrs.rename(columns={'team_api_id': 'home_team_api_id',
                                                'buildUpPlaySpeed': 'home_speed',
                                                'chanceCreationPassing': 'home_chance',
                                                'defencePressure': 'home_def_press',
                                                'defenceAggression': 'home_def_agg'}),
                  on='home_team_api_id', how='left')
    df = df.merge(latest_attrs.rename(columns={'team_api_id': 'away_team_api_id',
                                                'buildUpPlaySpeed': 'away_speed',
                                                'chanceCreationPassing': 'away_chance',
                                                'defencePressure': 'away_def_press',
                                                'defenceAggression': 'away_def_agg'}),
                  on='away_team_api_id', how='left')

    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_team_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team stats across all matches."""
    all_team_ids = pd.concat([df['home_team_api_id'], df['away_team_api_id']]).unique()
    stats = []
    team_map = df.drop_duplicates('home_team_api_id').set_index('home_team_api_id')['home_team'].to_dict()

    for tid in all_team_ids:
        home = df[df['home_team_api_id'] == tid]
        away = df[df['away_team_api_id'] == tid]
        total = len(home) + len(away)
        if total == 0:
            continue
        wins = len(home[home['result'] == 'H']) + len(away[away['result'] == 'A'])
        draws = len(home[home['result'] == 'D']) + len(away[away['result'] == 'D'])
        losses = total - wins - draws

        goals_scored = home['home_team_goal'].sum() + away['away_team_goal'].sum()
        goals_conceded = home['away_team_goal'].sum() + away['home_team_goal'].sum()

        home_wins = len(home[home['result'] == 'H'])
        home_total = len(home)

        stats.append({
            'team_id': tid,
            'team': team_map.get(tid, str(tid)),
            'total_matches': total,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': round(wins / total, 4),
            'goals_scored': int(goals_scored),
            'goals_conceded': int(goals_conceded),
            'goal_diff': int(goals_scored - goals_conceded),
            'home_win_rate': round(home_wins / home_total if home_total else 0, 4),
        })

    return pd.DataFrame(stats).sort_values('win_rate', ascending=False).reset_index(drop=True)


def league_standings(df: pd.DataFrame, league_name: str, season: str = None) -> pd.DataFrame:
    """Compute league standings (3pts win, 1pt draw)."""
    filt = df[df['league'] == league_name].copy()
    if season:
        filt = filt[filt['season'] == season]
    if filt.empty:
        return pd.DataFrame()

    teams = pd.concat([filt['home_team'], filt['away_team']]).dropna().unique()

    rows = []
    for team in teams:
        home = filt[filt['home_team'] == team]
        away = filt[filt['away_team'] == team]
        w = len(home[home['result'] == 'H']) + len(away[away['result'] == 'A'])
        d = len(home[home['result'] == 'D']) + len(away[away['result'] == 'D'])
        l = len(home[home['result'] == 'A']) + len(away[away['result'] == 'H'])
        gf = home['home_team_goal'].sum() + away['away_team_goal'].sum()
        ga = home['away_team_goal'].sum() + away['home_team_goal'].sum()
        rows.append({'Team': team, 'MP': w+d+l,
                     'W': w, 'D': d, 'L': l, 'GF': int(gf), 'GA': int(ga),
                     'GD': int(gf - ga), 'Pts': w*3+d})

    standings = pd.DataFrame(rows).sort_values(['Pts', 'GD'], ascending=False)
    standings['Rank'] = range(1, len(standings)+1)
    return standings.reset_index(drop=True)


def home_away_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Overall home vs away win rates by league."""
    agg = df.groupby('league').agg(
        home_wins=('result', lambda x: (x == 'H').sum()),
        away_wins=('result', lambda x: (x == 'A').sum()),
        draws=('result', lambda x: (x == 'D').sum()),
        total=('result', 'count'),
    ).reset_index()
    agg['home_win_pct'] = (agg['home_wins'] / agg['total'] * 100).round(1)
    agg['away_win_pct'] = (agg['away_wins'] / agg['total'] * 100).round(1)
    agg['draw_pct'] = (agg['draws'] / agg['total'] * 100).round(1)
    return agg.sort_values('home_win_pct', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. ML MODEL TRAINING
# ─────────────────────────────────────────────

def prepare_football_features(df: pd.DataFrame):
    """Build feature matrix for match outcome prediction (H/D/A)."""
    feature_cols = ['home_team_api_id', 'away_team_api_id']
    attr_cols = ['home_speed', 'home_chance', 'home_def_press', 'home_def_agg',
                 'away_speed', 'away_chance', 'away_def_press', 'away_def_agg']
    available_attr = [c for c in attr_cols if c in df.columns]
    feature_cols += available_attr
    if 'year' in df.columns:
        feature_cols.append('year')

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['result_enc']
    return X, y, feature_cols


def train_football_models(df: pd.DataFrame) -> dict:
    """Train classification models for football match prediction."""
    X, y, feature_names = prepare_football_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            'model': model,
            'accuracy': round(acc * 100, 2),
            'confusion_matrix': cm,
            'report': report,
            'y_test': y_test.values,
            'y_pred': y_pred,
        }
        print(f"  {name}: {acc*100:.2f}% accuracy")

    return results, feature_names


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_football_pipeline():
    """Run the full football analytics pipeline."""
    print("⚽ Loading Football Data...")
    matches, teams, leagues, countries, team_attrs = load_football_data()

    print("🧹 Cleaning & Merging Data...")
    df = clean_football_data(matches, teams, leagues, countries, team_attrs)

    print("⚙️  Feature Engineering...")
    team_perf = compute_team_performance(df)
    ha_analysis = home_away_analysis(df)

    print("🤖 Training ML Models...")
    model_results, feature_names = train_football_models(df)

    print("✅ Football Pipeline Complete!")
    return {
        'matches': df,
        'teams': teams,
        'leagues': leagues,
        'team_performance': team_perf,
        'home_away_analysis': ha_analysis,
        'models': model_results,
        'feature_names': feature_names,
    }
