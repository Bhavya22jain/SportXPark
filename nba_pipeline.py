"""
NBA Analytics Pipeline
Datasets: games.csv, games_details.csv, teams.csv, players.csv, ranking.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/nba"

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_nba_data():
    """Load all NBA CSV files."""
    games = pd.read_csv(f"{DATA_PATH}/games.csv")
    games_details = pd.read_csv(f"{DATA_PATH}/games_details.csv", low_memory=False)
    teams = pd.read_csv(f"{DATA_PATH}/teams.csv")
    players = pd.read_csv(f"{DATA_PATH}/players.csv")
    ranking = pd.read_csv(f"{DATA_PATH}/ranking.csv")
    return games, games_details, teams, players, ranking


# ─────────────────────────────────────────────
# 2. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────

def clean_nba_data(games, games_details, teams, players, ranking):
    """Clean and structure all NBA tables."""
    # 1. Clean games
    games = games.dropna(subset=['HOME_TEAM_WINS', 'PTS_home', 'PTS_away']).copy()
    games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'])
    games = games.sort_values('GAME_DATE_EST').reset_index(drop=True)

    # Map team names/abbreviations
    team_name_map = teams.set_index('TEAM_ID')['NICKNAME'].to_dict()
    team_full_map = dict(zip(teams['TEAM_ID'], teams['CITY'] + ' ' + teams['NICKNAME']))
    team_abbr_map = teams.set_index('TEAM_ID')['ABBREVIATION'].to_dict()

    games['home_team_name'] = games['HOME_TEAM_ID'].map(team_full_map)
    games['away_team_name'] = games['VISITOR_TEAM_ID'].map(team_full_map)
    games['home_team_abbr'] = games['HOME_TEAM_ID'].map(team_abbr_map)
    games['away_team_abbr'] = games['VISITOR_TEAM_ID'].map(team_abbr_map)

    # 2. Clean games_details
    games_details['MIN_clean'] = games_details['MIN'].astype(str).str.split(':').str[0]
    games_details['MIN_clean'] = pd.to_numeric(games_details['MIN_clean'], errors='coerce').fillna(0)
    games_details['PTS'] = games_details['PTS'].fillna(0)
    games_details['AST'] = games_details['AST'].fillna(0)
    games_details['REB'] = games_details['REB'].fillna(0)
    games_details['STL'] = games_details['STL'].fillna(0)
    games_details['BLK'] = games_details['BLK'].fillna(0)
    games_details['TO'] = games_details['TO'].fillna(0)
    games_details['FGM'] = games_details['FGM'].fillna(0)
    games_details['FGA'] = games_details['FGA'].fillna(0)
    games_details['FTM'] = games_details['FTM'].fillna(0)
    games_details['FTA'] = games_details['FTA'].fillna(0)

    # 3. Clean rankings
    ranking['STANDINGSDATE'] = pd.to_datetime(ranking['STANDINGSDATE'])
    ranking = ranking.sort_values('STANDINGSDATE').reset_index(drop=True)

    return games, games_details, teams, players, ranking, team_full_map, team_abbr_map


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_team_rolling_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features for each team to improve predictive modeling."""
    df = games.copy()

    # Create long-form df of team performances
    home_df = df[['GAME_DATE_EST', 'GAME_ID', 'HOME_TEAM_ID', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'HOME_TEAM_WINS']].copy()
    home_df.columns = ['date', 'game_id', 'team_id', 'pts', 'fg_pct', 'ft_pct', 'fg3_pct', 'ast', 'reb', 'won']
    home_df['is_home'] = 1

    away_df = df[['GAME_DATE_EST', 'GAME_ID', 'VISITOR_TEAM_ID', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'HOME_TEAM_WINS']].copy()
    away_df.columns = ['date', 'game_id', 'team_id', 'pts', 'fg_pct', 'ft_pct', 'fg3_pct', 'ast', 'reb', 'won']
    away_df['won'] = 1 - away_df['won']
    away_df['is_home'] = 0

    team_perf = pd.concat([home_df, away_df]).sort_values(['team_id', 'date']).reset_index(drop=True)

    # Calculate 10-game rolling averages (shift by 1 to prevent data leakage)
    g = team_perf.groupby('team_id')
    team_perf['roll_pts'] = g['pts'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    team_perf['roll_fg_pct'] = g['fg_pct'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    team_perf['roll_fg3_pct'] = g['fg3_pct'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    team_perf['roll_ast'] = g['ast'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    team_perf['roll_reb'] = g['reb'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    team_perf['roll_win_rate'] = g['won'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

    # Map back to games df
    home_roll = team_perf[team_perf['is_home'] == 1][['game_id', 'roll_pts', 'roll_fg_pct', 'roll_fg3_pct', 'roll_ast', 'roll_reb', 'roll_win_rate']].rename(
        columns=lambda x: x + '_home' if x != 'game_id' else 'GAME_ID'
    )
    away_roll = team_perf[team_perf['is_home'] == 0][['game_id', 'roll_pts', 'roll_fg_pct', 'roll_fg3_pct', 'roll_ast', 'roll_reb', 'roll_win_rate']].rename(
        columns=lambda x: x + '_away' if x != 'game_id' else 'GAME_ID'
    )

    df = df.merge(home_roll, on='GAME_ID', how='left')
    df = df.merge(away_roll, on='GAME_ID', how='left')

    # Fill NaNs from early games
    fill_cols = [c for c in df.columns if 'roll_' in c]
    df[fill_cols] = df[fill_cols].fillna(df[fill_cols].median())

    return df


def compute_player_efficiency(games_details: pd.DataFrame) -> pd.DataFrame:
    """Calculate NBA player efficiency rating (EFF)."""
    # EFF = (PTS + REB + AST + STL + BLK - Missed FG - Missed FT - TO)
    gd = games_details.copy()
    gd['missed_fg'] = gd['FGA'] - gd['FGM']
    gd['missed_ft'] = gd['FTA'] - gd['FTM']

    gd['EFF'] = (
        gd['PTS'] + gd['REB'] + gd['AST'] + gd['STL'] + gd['BLK']
        - gd['missed_fg'] - gd['missed_ft'] - gd['TO']
    )

    # Aggregate by player across seasons/all matches
    player_agg = gd.groupby(['PLAYER_ID', 'PLAYER_NAME']).agg(
        gp=('GAME_ID', 'count'),
        avg_pts=('PTS', 'mean'),
        avg_ast=('AST', 'mean'),
        avg_reb=('REB', 'mean'),
        avg_stl=('STL', 'mean'),
        avg_blk=('BLK', 'mean'),
        avg_eff=('EFF', 'mean'),
        total_pts=('PTS', 'sum'),
        total_ast=('AST', 'sum'),
        total_reb=('REB', 'sum'),
    ).reset_index()

    player_agg = player_agg[player_agg['gp'] >= 20]  # Min 20 games played
    player_agg = player_agg.round(2).sort_values('avg_eff', ascending=False).reset_index(drop=True)

    # MVP ranking metric: weighted blend of efficiency, scoring and volume
    max_eff = player_agg['avg_eff'].max()
    max_pts = player_agg['avg_pts'].max()
    player_agg['MVP_score'] = (
        (player_agg['avg_eff'] / max_eff * 50) +
        (player_agg['avg_pts'] / max_pts * 30) +
        (player_agg['gp'] / player_agg['gp'].max() * 20)
    ).round(2)

    return player_agg.sort_values('MVP_score', ascending=False).reset_index(drop=True)


def compute_team_ratings(games: pd.DataFrame, team_full_map: dict) -> pd.DataFrame:
    """Compute offensive and defensive ratings based on goals scored/conceded."""
    teams = list(team_full_map.keys())
    ratings = []

    for tid in teams:
        home_games = games[games['HOME_TEAM_ID'] == tid]
        away_games = games[games['VISITOR_TEAM_ID'] == tid]
        total_games = len(home_games) + len(away_games)
        if total_games == 0:
            continue

        pts_scored = home_games['PTS_home'].sum() + away_games['PTS_away'].sum()
        pts_conceded = home_games['PTS_away'].sum() + away_games['PTS_home'].sum()

        fg_pct = (home_games['FG_PCT_home'].mean() + away_games['FG_PCT_away'].mean()) / 2
        fg3_pct = (home_games['FG3_PCT_home'].mean() + away_games['FG3_PCT_away'].mean()) / 2
        ast = (home_games['AST_home'].mean() + away_games['AST_away'].mean()) / 2
        reb = (home_games['REB_home'].mean() + away_games['REB_away'].mean()) / 2

        wins = len(home_games[home_games['HOME_TEAM_WINS'] == 1]) + len(away_games[away_games['HOME_TEAM_WINS'] == 0])

        ratings.append({
            'team_id': tid,
            'team': team_full_map[tid],
            'gp': total_games,
            'win_rate': round(wins / total_games, 4),
            'off_rating': round(pts_scored / total_games, 2),  # Average points scored
            'def_rating': round(pts_conceded / total_games, 2),  # Average points conceded
            'net_rating': round((pts_scored - pts_conceded) / total_games, 2),
            'avg_fg_pct': round(fg_pct, 4),
            'avg_fg3_pct': round(fg3_pct, 4),
            'avg_ast': round(ast, 2),
            'avg_reb': round(reb, 2),
        })

    return pd.DataFrame(ratings).sort_values('win_rate', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. ML MODEL TRAINING
# ─────────────────────────────────────────────

def train_nba_models(games: pd.DataFrame) -> tuple:
    """Train classification models to predict HOME_TEAM_WINS."""
    df = games.copy()

    # Features: rolling averages + team IDs + home advantage
    features = [
        'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON',
        'roll_pts_home', 'roll_fg_pct_home', 'roll_fg3_pct_home', 'roll_ast_home', 'roll_reb_home', 'roll_win_rate_home',
        'roll_pts_away', 'roll_fg_pct_away', 'roll_fg3_pct_away', 'roll_ast_away', 'roll_reb_away', 'roll_win_rate_away'
    ]

    # Label Encoder for teams
    le_team = LabelEncoder()
    all_teams = pd.concat([df['HOME_TEAM_ID'], df['VISITOR_TEAM_ID']]).unique()
    le_team.fit(all_teams)

    df['HOME_TEAM_enc'] = le_team.transform(df['HOME_TEAM_ID'])
    df['VISITOR_TEAM_enc'] = le_team.transform(df['VISITOR_TEAM_ID'])

    encoded_features = [
        'HOME_TEAM_enc', 'VISITOR_TEAM_enc', 'SEASON',
        'roll_pts_home', 'roll_fg_pct_home', 'roll_fg3_pct_home', 'roll_ast_home', 'roll_reb_home', 'roll_win_rate_home',
        'roll_pts_away', 'roll_fg_pct_away', 'roll_fg3_pct_away', 'roll_ast_away', 'roll_reb_away', 'roll_win_rate_away'
    ]

    X = df[encoded_features]
    y = df['HOME_TEAM_WINS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=42),
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
            'y_test': y_test,
            'y_pred': y_pred,
        }
        print(f"  {name}: {acc*100:.2f}% accuracy")

    return results, le_team, encoded_features


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_nba_pipeline():
    """Run the entire NBA pipeline."""
    print("🏀 Loading NBA Data...")
    games, games_details, teams, players, ranking = load_nba_data()

    print("🧹 Cleaning Data...")
    games, games_details, teams, players, ranking, team_full_map, team_abbr_map = clean_nba_data(
        games, games_details, teams, players, ranking
    )

    print("⚙️  Feature Engineering...")
    games_roll = compute_team_rolling_features(games)
    player_eff = compute_player_efficiency(games_details)
    team_ratings = compute_team_ratings(games, team_full_map)

    print("🤖 Training ML Models...")
    model_results, le_team, feature_names = train_nba_models(games_roll)

    print("✅ NBA Pipeline Complete!")
    return {
        'games': games_roll,
        'games_details': games_details,
        'teams': teams,
        'players': players,
        'ranking': ranking,
        'player_efficiency': player_eff,
        'team_ratings': team_ratings,
        'models': model_results,
        'le_team': le_team,
        'feature_names': feature_names,
        'team_full_map': team_full_map,
        'team_abbr_map': team_abbr_map,
    }
