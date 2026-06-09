"""
Cricket Analytics Pipeline
IPL match data: deliveries.csv + matches.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/cricket"

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_cricket_data():
    """Load and merge matches + deliveries datasets."""
    matches = pd.read_csv(f"{DATA_PATH}/matches.csv")
    deliveries = pd.read_csv(f"{DATA_PATH}/deliveries.csv")
    return matches, deliveries


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────

def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Clean IPL matches dataset."""
    df = matches.copy()

    # Fill common missing columns
    for col in ['player_of_match', 'city', 'venue', 'umpire1', 'umpire2']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Normalise team names  (various abbreviation differences across seasons)
    team_rename = {
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Delhi Daredevils': 'Delhi Capitals',
    }
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        if col in df.columns:
            df[col] = df[col].replace(team_rename)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

    df.dropna(subset=['winner'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Clean IPL deliveries dataset."""
    df = deliveries.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)
    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_team_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute win-rate, home-win-rate and recent form for each team."""
    teams = pd.concat([matches['team1'], matches['team2']]).unique()
    stats = []
    for team in teams:
        team_matches = matches[(matches['team1'] == team) | (matches['team2'] == team)]
        total = len(team_matches)
        wins = len(team_matches[team_matches['winner'] == team])
        win_rate = wins / total if total > 0 else 0

        # Toss advantage
        toss_wins = len(team_matches[team_matches['toss_winner'] == team])
        toss_rate = toss_wins / total if total > 0 else 0

        # Recent form (last 10 matches)
        recent = team_matches.tail(10)
        recent_wins = len(recent[recent['winner'] == team])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0

        stats.append({
            'team': team,
            'total_matches': total,
            'total_wins': wins,
            'win_rate': round(win_rate, 4),
            'toss_win_rate': round(toss_rate, 4),
            'recent_form': round(recent_form, 4),
        })
    return pd.DataFrame(stats).sort_values('win_rate', ascending=False).reset_index(drop=True)


def compute_player_stats(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate batting stats per player per season."""
    # Merge to get year
    if 'year' not in deliveries.columns:
        merged = deliveries.merge(matches[['id', 'year']], left_on='match_id', right_on='id', how='left')
    else:
        merged = deliveries.copy()

    # Batsman column name normalisation
    bat_col = 'batter' if 'batter' in merged.columns else 'batsman'

    # Group by player
    player_stats = merged.groupby(bat_col).agg(
        total_runs=('batsman_runs', 'sum'),
        balls_faced=('ball', 'count'),
        fours=('batsman_runs', lambda x: (x == 4).sum()),
        sixes=('batsman_runs', lambda x: (x == 6).sum()),
        innings=('match_id', 'nunique'),
    ).reset_index()
    player_stats.columns = ['player', 'total_runs', 'balls_faced', 'fours', 'sixes', 'innings']
    player_stats['strike_rate'] = (player_stats['total_runs'] / player_stats['balls_faced'] * 100).round(2)
    player_stats['avg_runs_per_innings'] = (player_stats['total_runs'] / player_stats['innings']).round(2)
    player_stats = player_stats[player_stats['innings'] >= 5].sort_values('total_runs', ascending=False)
    return player_stats.reset_index(drop=True)


def compute_bowling_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bowling stats per bowler."""
    bowl_col = 'bowler'
    wicket_col = 'player_dismissed' if 'player_dismissed' in deliveries.columns else None

    agg = {
        'total_runs_conceded': ('total_runs', 'sum'),
        'balls_bowled': ('ball', 'count'),
        'matches': ('match_id', 'nunique'),
    }

    bowl_stats = deliveries.groupby(bowl_col).agg(**agg).reset_index()
    bowl_stats.columns = ['bowler', 'runs_conceded', 'balls_bowled', 'matches']
    bowl_stats['overs'] = bowl_stats['balls_bowled'] / 6
    bowl_stats['economy'] = (bowl_stats['runs_conceded'] / bowl_stats['overs']).round(2)

    if wicket_col:
        wickets = deliveries[deliveries[wicket_col].notna() & (deliveries[wicket_col] != 0)].groupby(bowl_col).size().reset_index(name='wickets')
        bowl_stats = bowl_stats.merge(wickets, on='bowler', how='left')
        bowl_stats['wickets'] = bowl_stats['wickets'].fillna(0).astype(int)
        bowl_stats['bowling_avg'] = (bowl_stats['runs_conceded'] / bowl_stats['wickets'].replace(0, np.nan)).round(2)
        bowl_stats['bowling_avg'] = bowl_stats['bowling_avg'].fillna(999)
    else:
        bowl_stats['wickets'] = 0

    bowl_stats = bowl_stats[bowl_stats['matches'] >= 5]
    return bowl_stats.sort_values('wickets', ascending=False).reset_index(drop=True)


def venue_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Wins for batting first vs chasing per venue."""
    df = matches.copy()
    if 'venue' not in df.columns:
        return pd.DataFrame()

    df['toss_winner_won'] = df['toss_winner'] == df['winner']
    df['batting_first_won'] = (
        ((df['toss_winner'] == df['team1']) & (df['toss_decision'] == 'bat') & (df['winner'] == df['team1'])) |
        ((df['toss_winner'] == df['team2']) & (df['toss_decision'] == 'bat') & (df['winner'] == df['team2'])) |
        ((df['toss_winner'] == df['team1']) & (df['toss_decision'] == 'field') & (df['winner'] == df['team2'])) |
        ((df['toss_winner'] == df['team2']) & (df['toss_decision'] == 'field') & (df['winner'] == df['team1']))
    )

    venue_agg = df.groupby('venue').agg(
        total_matches=('id', 'count'),
        batting_first_wins=('batting_first_won', 'sum'),
    ).reset_index()
    venue_agg['chasing_wins'] = venue_agg['total_matches'] - venue_agg['batting_first_wins']
    venue_agg['batting_first_win_pct'] = (venue_agg['batting_first_wins'] / venue_agg['total_matches'] * 100).round(1)
    return venue_agg.sort_values('total_matches', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. ML MODEL TRAINING
# ─────────────────────────────────────────────

def prepare_ml_features(matches: pd.DataFrame) -> tuple:
    """Build feature matrix for match winner prediction."""
    df = matches.copy()

    # Encode teams
    le_team = LabelEncoder()
    all_teams = pd.concat([df['team1'], df['team2'], df['toss_winner']]).unique()
    le_team.fit(all_teams)

    df['team1_enc'] = le_team.transform(df['team1'])
    df['team2_enc'] = le_team.transform(df['team2'])
    df['toss_winner_enc'] = le_team.transform(df['toss_winner'])

    # Toss decision
    df['toss_decision_enc'] = (df['toss_decision'] == 'bat').astype(int)

    # Venue encoding
    if 'venue' in df.columns:
        le_venue = LabelEncoder()
        df['venue_enc'] = le_venue.fit_transform(df['venue'].fillna('Unknown'))
    else:
        df['venue_enc'] = 0

    # Target: 1 if team1 wins, 0 if team2 wins
    df['target'] = (df['winner'] == df['team1']).astype(int)

    features = ['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_enc', 'venue_enc']
    if 'year' in df.columns:
        features.append('year')

    X = df[features]
    y = df['target']
    return X, y, le_team, features


def train_models(matches: pd.DataFrame) -> dict:
    """Train Random Forest, Logistic Regression and XGBoost models."""
    X, y, le_team, feature_names = prepare_ml_features(matches)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
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

    return results, le_team, feature_names


def predict_match(team1: str, team2: str, toss_winner: str, toss_decision: str,
                  venue: str, le_team, models: dict, feature_names: list, matches: pd.DataFrame) -> dict:
    """Predict winner for a new match."""
    try:
        teams_known = list(le_team.classes_)
        if team1 not in teams_known or team2 not in teams_known:
            return {'error': 'Unknown team name'}

        t1_enc = le_team.transform([team1])[0]
        t2_enc = le_team.transform([team2])[0]
        tw_enc = le_team.transform([toss_winner])[0]
        td_enc = 1 if toss_decision == 'bat' else 0

        # Venue — use modal encoding
        venue_map = {}
        if 'venue' in matches.columns:
            from sklearn.preprocessing import LabelEncoder
            le_v = LabelEncoder()
            le_v.fit(matches['venue'].fillna('Unknown'))
            venue_enc = le_v.transform([venue])[0] if venue in le_v.classes_ else 0
        else:
            venue_enc = 0

        row = pd.DataFrame([[t1_enc, t2_enc, tw_enc, td_enc, venue_enc]], columns=['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_enc', 'venue_enc'])
        if 'year' in feature_names:
            row['year'] = 2024

        preds = {}
        for name, res in models.items():
            model = res['model']
            prob = model.predict_proba(row)[0]
            winner = team1 if prob[1] > 0.5 else team2
            confidence = max(prob) * 100
            preds[name] = {'winner': winner, 'confidence': round(confidence, 1), 'prob_team1': round(prob[1]*100, 1)}
        return preds
    except Exception as e:
        return {'error': str(e)}


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_cricket_pipeline():
    """Run the full cricket analytics pipeline and return all artifacts."""
    print("🏏 Loading Cricket Data...")
    matches_raw, deliveries_raw = load_cricket_data()

    print("🧹 Cleaning Data...")
    matches = clean_matches(matches_raw)
    deliveries = clean_deliveries(deliveries_raw)

    print("⚙️  Feature Engineering...")
    team_stats = compute_team_stats(matches)
    player_stats = compute_player_stats(deliveries, matches)
    bowl_stats = compute_bowling_stats(deliveries)
    venue_data = venue_stats(matches)

    print("🤖 Training ML Models...")
    model_results, le_team, feature_names = train_models(matches)

    print("✅ Cricket Pipeline Complete!")
    return {
        'matches': matches,
        'deliveries': deliveries,
        'team_stats': team_stats,
        'player_stats': player_stats,
        'bowl_stats': bowl_stats,
        'venue_stats': venue_data,
        'models': model_results,
        'le_team': le_team,
        'feature_names': feature_names,
    }
