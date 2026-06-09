"""
Insights & Cross-Sport Analytics Engine
Precomputes interesting narratives, AI-style insights, and cross-sport statistics
from the processed datasets.
"""

import pandas as pd
import numpy as np

def generate_cricket_insights(matches: pd.DataFrame, team: str) -> list:
    """Generate AI insights for a Cricket team."""
    insights = []
    
    # Filter matches for the team
    team_matches = matches[(matches['team1'] == team) | (matches['team2'] == team)]
    if team_matches.empty:
        return ["No matching data found for this team."]

    # 1. Toss win vs match win
    toss_wins = team_matches[team_matches['toss_winner'] == team]
    toss_loss = team_matches[team_matches['toss_winner'] != team]
    
    tw_mw = len(toss_wins[toss_wins['winner'] == team])
    tl_mw = len(toss_loss[toss_loss['winner'] == team])
    
    tw_pct = (tw_mw / len(toss_wins) * 100) if not toss_wins.empty else 0
    tl_pct = (tl_mw / len(toss_loss) * 100) if not toss_loss.empty else 0
    
    if abs(tw_pct - tl_pct) > 10:
        if tw_pct > tl_pct:
            insights.append(f"Toss reliance is HIGH: wins {tw_pct:.1f}% of matches when winning the toss, compared to only {tl_pct:.1f}% when losing it.")
        else:
            insights.append(f"Toss resilient: performs BETTER when losing the toss (win rate {tl_pct:.1f}%) than when winning it ({tw_pct:.1f}%).")
    else:
        insights.append(f"Toss neutral: Win rate is stable regardless of toss results ({tw_pct:.1f}% on toss wins, {tl_pct:.1f}% on toss losses).")

    # 2. Batting first vs chasing
    # In matches, if toss_decision is 'field' and toss_winner is team, team is chasing (team2).
    # If toss_decision is 'bat' and toss_winner is team, team is batting first (team1).
    def is_chasing(row):
        if row['toss_winner'] == team:
            return row['toss_decision'] == 'field'
        else:
            # opponent won toss, if they chose field, team is batting first; if they chose bat, team is chasing
            return row['toss_decision'] == 'bat'
            
    team_matches = team_matches.copy()
    team_matches['is_chase'] = team_matches.apply(is_chasing, axis=1)
    
    chase_matches = team_matches[team_matches['is_chase']]
    defend_matches = team_matches[~team_matches['is_chase']]
    
    chase_wins = len(chase_matches[chase_matches['winner'] == team])
    defend_wins = len(defend_matches[defend_matches['winner'] == team])
    
    chase_pct = (chase_wins / len(chase_matches) * 100) if not chase_matches.empty else 0
    defend_pct = (defend_wins / len(defend_matches) * 100) if not defend_matches.empty else 0
    
    if abs(chase_pct - defend_pct) > 8:
        if chase_pct > defend_pct:
            insights.append(f"Strong chasing team: win rate is significantly higher when chasing ({chase_pct:.1f}%) compared to defending ({defend_pct:.1f}%).")
        else:
            insights.append(f"Defending specialists: performs better when putting runs on the board first, winning {defend_pct:.1f}% vs chasing {chase_pct:.1f}%.")
    else:
        insights.append(f"Versatile setup: equally comfortable batting first ({defend_pct:.1f}% win rate) or chasing ({chase_pct:.1f}% win rate).")

    # 3. Best Venue
    won_matches = team_matches[team_matches['winner'] == team]
    if not won_matches.empty and 'venue' in won_matches.columns:
        best_venue = won_matches['venue'].value_counts().idxmax()
        venue_wins = won_matches['venue'].value_counts().max()
        insights.append(f"Fortress Venue: Most successful ground is '{best_venue}' with {venue_wins} recorded wins.")

    return insights


def generate_football_insights(matches: pd.DataFrame, team: str) -> list:
    """Generate AI insights for a Football team."""
    insights = []
    
    # Filter matches
    team_matches = matches[(matches['home_team'] == team) | (matches['away_team'] == team)]
    if team_matches.empty:
        return ["No matching data found for this team."]

    # 1. Home vs Away Goals & Wins
    home_games = team_matches[team_matches['home_team'] == team]
    away_games = team_matches[team_matches['away_team'] == team]
    
    home_wins = len(home_games[home_games['result'] == 'H'])
    away_wins = len(away_games[away_games['result'] == 'A'])
    
    home_win_pct = (home_wins / len(home_games) * 100) if not home_games.empty else 0
    away_win_pct = (away_wins / len(away_games) * 100) if not away_games.empty else 0
    
    # Goals
    avg_goals_scored_home = home_games['home_team_goal'].mean()
    avg_goals_scored_away = away_games['away_team_goal'].mean()
    
    insights.append(f"Home Advantage: Win percentage is {home_win_pct:.1f}% at home compared to {away_win_pct:.1f}% on the road.")
    
    if avg_goals_scored_home > avg_goals_scored_away + 0.5:
        insights.append(f"Attacking discrepancy: Scores {avg_goals_scored_home:.2f} goals/game at home vs {avg_goals_scored_away:.2f} away, suggesting a more conservative away setup.")
    else:
        insights.append(f"Consistent firepower: Maintains a steady attack of {avg_goals_scored_home:.2f} goals/game at home and {avg_goals_scored_away:.2f} goals/game away.")

    # 2. Defensive Solidity
    avg_goals_conceded_home = home_games['away_team_goal'].mean()
    avg_goals_conceded_away = away_games['home_team_goal'].mean()
    
    if avg_goals_conceded_away > avg_goals_conceded_home + 0.4:
        insights.append(f"Vulnerable away defence: Concedes {avg_goals_conceded_away:.2f} goals/game away compared to a tighter {avg_goals_conceded_home:.2f} at home.")
    else:
        insights.append(f"Defensive discipline: Defensive record is stable (concedes {avg_goals_conceded_home:.2f} home vs {avg_goals_conceded_away:.2f} away).")

    return insights


def generate_nba_insights(games: pd.DataFrame, team_id: int, team_name: str) -> list:
    """Generate AI insights for an NBA team."""
    insights = []
    
    # Filter games
    team_games = games[(games['home_team_name'] == team_name) | (games['away_team_name'] == team_name)]
    if team_games.empty:
        return ["No matching data found for this team."]

    home_games = team_games[team_games['home_team_name'] == team_name]
    away_games = team_games[team_games['away_team_name'] == team_name]

    # Home vs Away scoring
    avg_pts_home = home_games['PTS_home'].mean()
    avg_pts_away = away_games['PTS_away'].mean()
    
    # Win percentages
    home_wins = len(home_games[home_games['HOME_TEAM_WINS'] == 1])
    away_wins = len(away_games[away_games['HOME_TEAM_WINS'] == 0])
    
    home_win_pct = (home_wins / len(home_games) * 100) if not home_games.empty else 0
    away_win_pct = (away_wins / len(away_games) * 100) if not away_games.empty else 0

    insights.append(f"Arena Factor: Wins {home_win_pct:.1f}% of games in home court, and {away_win_pct:.1f}% of road games.")
    
    if avg_pts_home > avg_pts_away + 3:
        insights.append(f"Fast-paced home offense: Averages {avg_pts_home:.1f} pts per game at home compared to {avg_pts_away:.1f} pts on the road.")
    else:
        insights.append(f"Steady pace: Scores {avg_pts_home:.1f} pts (home) and {avg_pts_away:.1f} pts (away) consistently.")

    # Rebounding
    avg_reb_home = home_games['REB_home'].mean()
    avg_reb_away = away_games['REB_away'].mean()
    
    if avg_reb_home > avg_reb_away + 2:
        insights.append(f"Rebound dominance: Averages {avg_reb_home:.1f} boards at home, significantly out-hustling away games ({avg_reb_away:.1f}).")

    return insights


def load_cross_sport_metrics() -> dict:
    """Precompute cross-sport metrics such as home advantage, consistency, etc."""
    metrics = {}
    
    # 1. Cricket Home win rate (IPL: does batting first or toss win play a larger role?)
    try:
        cr_matches = pd.read_csv("data/processed/cricket_matches.csv")
        cr_total = len(cr_matches)
        
        # Win rate for team1 (historically team1 matches or default home team)
        # In IPL matches, toss decision and home advantage are venue-specific. Let's compute toss advantage:
        toss_wins = cr_matches[cr_matches['toss_winner'] == cr_matches['winner']]
        toss_adv = len(toss_wins) / cr_total
        
        # Batting first win rate
        # we can check target column or recreate batting first
        bat_first_won = (
            ((cr_matches['toss_winner'] == cr_matches['team1']) & (cr_matches['toss_decision'] == 'bat') & (cr_matches['winner'] == cr_matches['team1'])) |
            ((cr_matches['toss_winner'] == cr_matches['team2']) & (cr_matches['toss_decision'] == 'bat') & (cr_matches['winner'] == cr_matches['team2'])) |
            ((cr_matches['toss_winner'] == cr_matches['team1']) & (cr_matches['toss_decision'] == 'field') & (cr_matches['winner'] == cr_matches['team2'])) |
            ((cr_matches['toss_winner'] == cr_matches['team2']) & (cr_matches['toss_decision'] == 'field') & (cr_matches['winner'] == cr_matches['team1']))
        ).sum()
        
        metrics['cricket'] = {
            'toss_win_rate': round(toss_adv * 100, 2),
            'batting_first_win_rate': round((bat_first_won / cr_total) * 100, 2),
            'chasing_win_rate': round((1 - bat_first_won / cr_total) * 100, 2),
            'total_sample': cr_total
        }
    except Exception as e:
        metrics['cricket'] = {'error': str(e)}

    # 2. Football Home advantage
    try:
        fb_matches = pd.read_csv("data/processed/football_matches.csv")
        fb_total = len(fb_matches)
        home_wins = len(fb_matches[fb_matches['result'] == 'H'])
        away_wins = len(fb_matches[fb_matches['result'] == 'A'])
        draws = len(fb_matches[fb_matches['result'] == 'D'])
        
        metrics['football'] = {
            'home_win_rate': round((home_wins / fb_total) * 100, 2),
            'away_win_rate': round((away_wins / fb_total) * 100, 2),
            'draw_rate': round((draws / fb_total) * 100, 2),
            'total_sample': fb_total
        }
    except Exception as e:
        metrics['football'] = {'error': str(e)}

    # 3. NBA Home advantage
    try:
        nba_games = pd.read_csv("data/processed/nba_games.csv")
        nba_total = len(nba_games)
        home_wins = len(nba_games[nba_games['HOME_TEAM_WINS'] == 1])
        
        metrics['nba'] = {
            'home_win_rate': round((home_wins / nba_total) * 100, 2),
            'away_win_rate': round((1 - home_wins / nba_total) * 100, 2),
            'total_sample': nba_total
        }
    except Exception as e:
        metrics['nba'] = {'error': str(e)}
        
    return metrics
