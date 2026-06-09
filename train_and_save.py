"""
Model Training and Data Pre-computation Script
Runs all pipelines, trains models, saves them, and exports lightweight processed data
for rapid loading in the Streamlit app.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

def train_and_save_all():
    print("==================================================")
    print("Starting SportSphere AI Training & Pre-computation")
    print("==================================================")

    # 1. CRICKET
    try:
        from src.cricket_pipeline import run_cricket_pipeline
        cricket_res = run_cricket_pipeline()
        
        # Save Cricket Data
        cricket_res['team_stats'].to_csv("data/processed/cricket_team_stats.csv", index=False)
        cricket_res['player_stats'].to_csv("data/processed/cricket_player_stats.csv", index=False)
        cricket_res['bowl_stats'].to_csv("data/processed/cricket_bowl_stats.csv", index=False)
        cricket_res['venue_stats'].to_csv("data/processed/cricket_venue_stats.csv", index=False)
        cricket_res['matches'].to_csv("data/processed/cricket_matches.csv", index=False)
        
        # Save Cricket ML Models
        with open("models/cricket_models.pkl", "wb") as f:
            pickle.dump({
                'models': cricket_res['models'],
                'le_team': cricket_res['le_team'],
                'feature_names': cricket_res['feature_names']
            }, f)
        print("[OK] Cricket artifacts saved.")
    except Exception as e:
        print(f"[ERROR] Error in Cricket pipeline: {e}")

    # 2. FOOTBALL
    try:
        from src.football_pipeline import run_football_pipeline
        football_res = run_football_pipeline()
        
        # Save Football Data
        football_res['team_performance'].to_csv("data/processed/football_team_performance.csv", index=False)
        football_res['home_away_analysis'].to_csv("data/processed/football_home_away_analysis.csv", index=False)
        football_res['teams'].to_csv("data/processed/football_teams.csv", index=False)
        football_res['leagues'].to_csv("data/processed/football_leagues.csv", index=False)
        
        # Save cleaned matches
        matches_light = football_res['matches'].drop(columns=[c for c in football_res['matches'].columns if 'home_team_api_id' in c or 'away_team_api_id' in c], errors='ignore')
        matches_light.to_csv("data/processed/football_matches.csv", index=False)
        
        # Save Football ML Models
        with open("models/football_models.pkl", "wb") as f:
            pickle.dump({
                'models': football_res['models'],
                'feature_names': football_res['feature_names']
            }, f)
        print("[OK] Football artifacts saved.")
    except Exception as e:
        print(f"[ERROR] Error in Football pipeline: {e}")

    # 3. NBA
    try:
        from src.nba_pipeline import run_nba_pipeline
        nba_res = run_nba_pipeline()
        
        # Save NBA Data
        nba_res['player_efficiency'].to_csv("data/processed/nba_player_efficiency.csv", index=False)
        nba_res['team_ratings'].to_csv("data/processed/nba_team_ratings.csv", index=False)
        nba_res['ranking'].to_csv("data/processed/nba_ranking.csv", index=False)
        nba_res['teams'].to_csv("data/processed/nba_teams.csv", index=False)
        nba_res['players'].to_csv("data/processed/nba_players.csv", index=False)
        
        # Save a lighter games dataset
        games_light = nba_res['games'].drop(columns=[c for c in nba_res['games'].columns if 'TEAM_ID' in c], errors='ignore')
        games_light.to_csv("data/processed/nba_games.csv", index=False)
        
        # Precompute and save player details summary
        player_details_summary = nba_res['games_details'].groupby(['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']).agg(
            avg_pts=('PTS', 'mean'),
            avg_ast=('AST', 'mean'),
            avg_reb=('REB', 'mean'),
            avg_stl=('STL', 'mean'),
            avg_blk=('BLK', 'mean'),
            gp=('GAME_ID', 'count')
        ).reset_index()
        player_details_summary.to_csv("data/processed/nba_player_details_summary.csv", index=False)

        # Save NBA ML Models
        with open("models/nba_models.pkl", "wb") as f:
            pickle.dump({
                'models': nba_res['models'],
                'le_team': nba_res['le_team'],
                'feature_names': nba_res['feature_names'],
                'team_full_map': nba_res['team_full_map'],
                'team_abbr_map': nba_res['team_abbr_map']
            }, f)
        print("[OK] NBA artifacts saved.")
    except Exception as e:
        print(f"[ERROR] Error in NBA pipeline: {e}")

    print("==================================================")
    print("All tasks finished successfully!")
    print("==================================================")

if __name__ == "__main__":
    train_and_save_all()
