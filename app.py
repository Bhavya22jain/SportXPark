"""
SportSphere AI — Multi-Sport Analytics & Prediction Platform
Streamlit Dashboard Entrypoint
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Set page config
st.set_page_config(
    page_title="SportXPark — Multi-Sport Analytics & Prediction Platform",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
        /* Import fonts */
        @import url('https://api.fontshare.com/v2/css?f[]=clash-grotesk@400,600,700&f[]=satoshi@400,500,700&display=swap');

        /* Base page styling */
        .stApp {
            background-color: #0d0d0d;
            color: #b7ab98;
        }

        /* Noise Texture Overlay */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background-image: url('data:image/svg+xml,%3Csvg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"%3E%3Cfilter id="noiseFilter"%3E%3CfeTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch"/%3E%3C/filter%3E%3Crect width="100%25" height="100%25" filter="url(%23noiseFilter)"/%3E%3C/svg%3E');
            opacity: 0.03;
            pointer-events: none;
            z-index: 9999;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm {
            font-family: 'Clash Grotesk', sans-serif !important;
            color: #b7ab98 !important;
            text-transform: uppercase;
        }
        
        body, p, .stMarkdown p, li, label, .st-emotion-cache-16idsys p {
            font-family: 'Satoshi', sans-serif !important;
        }
        
        /* Fix Top Gap */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }
        .block-container {
            padding-top: 1rem !important;
        }

        /* Accent Colors */
        .coral-text {
            color: #eb5939 !important;
        }

        /* Metric cards */
        .metric-card {
            background: #1a1a1a;
            border: 1px solid rgba(118, 118, 118, 0.12);
            border-radius: 0px;
            padding: 24px;
            box-shadow: none;
            transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            border-color: #b7ab98;
        }
        
        .metric-value {
            font-size: 40px;
            font-weight: 700;
            color: #eb5939;
            margin-bottom: 4px;
            font-family: 'Clash Grotesk', sans-serif;
        }
        
        .metric-label {
            font-size: 14px;
            font-weight: 500;
            color: #b7ab98;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Header titles */
        .gradient-text {
            color: #eb5939;
            font-weight: 800;
        }
        
        /* Sections */
        .section-header {
            border-bottom: 1px solid rgba(118, 118, 118, 0.12);
            padding-bottom: 8px;
            margin-bottom: 20px;
        }

        /* Hero Section Custom Classes */
        .hero-title {
            font-family: 'Clash Grotesk', sans-serif;
            font-size: 100px;
            line-height: 0.9;
            text-transform: uppercase;
            color: #b7ab98;
            margin-bottom: 0px;
            padding-bottom: 0px;
            animation: slideUp 1.15s cubic-bezier(0.16, 1, 0.3, 1);
        }
        
        @keyframes slideUp {
            from { transform: translateY(115%); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .hero-eyebrow {
            font-family: 'Satoshi', sans-serif;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #eb5939;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }

        .hero-eyebrow::after {
            content: "";
            flex-grow: 1;
            height: 1px;
            background: rgba(118, 118, 118, 0.12);
        }

        /* Marquee */
        .marquee-wrapper {
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
            padding: 20px 0;
            border-top: 1px solid rgba(118, 118, 118, 0.12);
            border-bottom: 1px solid rgba(118, 118, 118, 0.12);
            margin: 40px 0;
        }
        
        .marquee-content {
            display: inline-block;
            animation: marquee 30s linear infinite;
            font-family: 'Clash Grotesk', sans-serif;
            font-size: 64px;
            font-weight: 600;
            text-transform: uppercase;
            color: transparent;
            -webkit-text-stroke: 1px rgba(183, 171, 152, 0.5);
        }
        
        .marquee-content:hover {
            animation-play-state: paused;
        }

        .marquee-content span.star {
            color: #eb5939;
            -webkit-text-stroke: 0;
            margin: 0 32px;
            display: inline-block;
            transform: translateY(-10px);
        }

        @keyframes marquee {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
        }
        
        .spinning-badge {
            animation: spin 22s linear infinite;
            width: 200px;
            height: 200px;
        }
        @keyframes spin {
            100% { transform: rotate(360deg); }
        }
        
        .magnetic-dot {
            width: 64px; height: 64px;
            background: #b7ab98;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            color: #0d0d0d; font-weight: bold;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        }
        .magnetic-dot:hover {
            background: #eb5939;
            transform: translate(-50%, -55%);
        }

        /* Sidebar and Inputs */
        [data-testid="stSidebar"] {
            background-color: #0d0d0d !important;
            border-right: 1px solid rgba(118, 118, 118, 0.12) !important;
        }
        
        .stRadio > div[role="radiogroup"] > label {
            background-color: #1a1a1a !important;
            border: 1px solid rgba(118, 118, 118, 0.12) !important;
            padding: 12px 16px !important;
            border-radius: 0px !important;
            margin-bottom: 8px !important;
            transition: all 0.3s ease !important;
        }
        .stRadio > div[role="radiogroup"] > label,
        .stRadio > div[role="radiogroup"] > label p,
        .stRadio > div[role="radiogroup"] > label div {
            color: #FFD700 !important; /* Bright Yellow for clarity */
        }
        .stRadio > div[role="radiogroup"] > label:hover {
            border-color: #FFD700 !important;
            transform: translateX(4px);
        }
        
        .stSelectbox div[data-baseweb="select"] > div, .stTextInput > div > div > input {
            background-color: #1a1a1a !important;
            border: 1px solid rgba(118, 118, 118, 0.12) !important;
            border-radius: 0px !important;
            color: #b7ab98 !important;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: transparent !important;
            color: #b7ab98 !important;
            border: 1px solid #b7ab98 !important;
            border-radius: 50px !important;
            padding: 10px 24px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            font-family: 'Satoshi', sans-serif !important;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        .stButton > button:hover {
            border-color: #eb5939 !important;
            color: #eb5939 !important;
            transform: translateY(-2px) !important;
        }

    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 1. LOAD DATA & MODELS (CACHED)
# ─────────────────────────────────────────────

@st.cache_data
def load_processed_data():
    data = {}
    
    # Cricket Data
    try:
        data['cricket_team_stats'] = pd.read_csv("data/processed/cricket_team_stats.csv")
        data['cricket_player_stats'] = pd.read_csv("data/processed/cricket_player_stats.csv")
        data['cricket_bowl_stats'] = pd.read_csv("data/processed/cricket_bowl_stats.csv")
        data['cricket_venue_stats'] = pd.read_csv("data/processed/cricket_venue_stats.csv")
        data['cricket_matches'] = pd.read_csv("data/processed/cricket_matches.csv")
    except Exception as e:
        st.warning(f"Error loading Cricket datasets: {e}")

    # Football Data
    try:
        data['football_team_perf'] = pd.read_csv("data/processed/football_team_performance.csv")
        data['football_ha'] = pd.read_csv("data/processed/football_home_away_analysis.csv")
        data['football_teams'] = pd.read_csv("data/processed/football_teams.csv")
        data['football_leagues'] = pd.read_csv("data/processed/football_leagues.csv")
        data['football_matches'] = pd.read_csv("data/processed/football_matches.csv")
    except Exception as e:
        st.warning(f"Error loading Football datasets: {e}")

    # NBA Data
    try:
        data['nba_player_eff'] = pd.read_csv("data/processed/nba_player_efficiency.csv")
        data['nba_team_ratings'] = pd.read_csv("data/processed/nba_team_ratings.csv")
        data['nba_ranking'] = pd.read_csv("data/processed/nba_ranking.csv")
        data['nba_teams'] = pd.read_csv("data/processed/nba_teams.csv")
        data['nba_players'] = pd.read_csv("data/processed/nba_players.csv")
        data['nba_games'] = pd.read_csv("data/processed/nba_games.csv")
        data['nba_player_summary'] = pd.read_csv("data/processed/nba_player_details_summary.csv")
    except Exception as e:
        st.warning(f"Error loading NBA datasets: {e}")

    return data


@st.cache_resource
def load_ml_models():
    models = {}
    # Cricket
    try:
        with open("models/cricket_models.pkl", "rb") as f:
            models['cricket'] = pickle.load(f)
    except:
        models['cricket'] = None
        
    # Football
    try:
        with open("models/football_models.pkl", "rb") as f:
            models['football'] = pickle.load(f)
    except:
        models['football'] = None

    # NBA
    try:
        with open("models/nba_models.pkl", "rb") as f:
            models['nba'] = pickle.load(f)
    except:
        models['nba'] = None

    return models


# Initialize data and models
data = load_processed_data()
models = load_ml_models()


# ─────────────────────────────────────────────
# 2. SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

# Initialize session state for navigation
if 'active_page' not in st.session_state:
    st.session_state.active_page = "🏟️ Unified Hub"

with st.sidebar:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0px;' class='gradient-text'>SportXPark</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #ffffff; font-size: 14px;'>Multi-Sport Prediction & Analytics</p>", unsafe_allow_html=True)
    st.write("---")
    
    st.radio(
        "Navigate Platform",
        ["🏟️ Unified Hub", "🏏 Cricket (IPL) Hub", "⚽ Football Hub", "🏀 NBA Hub", "🧠 AI Insights & Cross-Sport"],
        key="active_page"
    )
    app_mode = st.session_state.active_page
    
    st.write("---")
    st.markdown("<div style='font-size: 11px; text-align: center; color: #767676; padding-top: 10px;'>SportXPark v1.0.0<br/>Portfolio Showcase Project</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. UNIFIED HUB
# ─────────────────────────────────────────────

if app_mode == "🏟️ Unified Hub":
    # Hero Section
    st.markdown("""
        <div style="padding: 40px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="width: 65%;">
                    <div class="hero-eyebrow">Multidisciplinary</div>
                    <div style="overflow: hidden;"><h1 class="hero-title">SportXPark</h1></div>
                    <div style="overflow: hidden;"><h1 class="hero-title" style="color: #eb5939;">Platform</h1></div>
                </div>
                <div style="width: 35%; display: flex; justify-content: flex-end; position: relative;">
                    <div style="position: relative; width: 200px; height: 200px;">
                        <svg class="spinning-badge" viewBox="0 0 100 100">
                            <path id="circlePath" d="M 50, 50 m -40, 0 a 40,40 0 1,1 80,0 a 40,40 0 1,1 -80,0" fill="none"/>
                            <text fill="#b7ab98" font-size="12.5" font-family="'Clash Grotesk', sans-serif" letter-spacing="4">
                                <textPath href="#circlePath">OPEN TO WORK • FREELANCE AVAILABLE • </textPath>
                            </text>
                        </svg>
                        <div class="magnetic-dot">→</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="marquee-wrapper">
            <div class="marquee-content">
                CRICKET <span class="star">✦</span> FOOTBALL <span class="star">✦</span> BASKETBALL <span class="star">✦</span> MULTI-SPORT ANALYTICS <span class="star">✦</span> CRICKET <span class="star">✦</span> FOOTBALL <span class="star">✦</span> BASKETBALL <span class="star">✦</span> MULTI-SPORT ANALYTICS
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("Welcome to **SportXPark**, an advanced end-to-end sports analytics platform powered by Machine Learning and Interactive Visualizations. Explore comprehensive data insights across Cricket (IPL), Football (European Leagues), and NBA Basketball.")
    
    st.write("")
    
    # KPI Grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>3</div>
                <div class='metric-label'>Major Sports Modules</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        match_count = len(data.get('cricket_matches', [])) + len(data.get('football_matches', [])) + len(data.get('nba_games', []))
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{match_count:,}</div>
                <div class='metric-label'>Total Match Database</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>9</div>
                <div class='metric-label'>Trained ML Predictors</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>63.2%</div>
                <div class='metric-label'>Peak NBA Prediction Acc</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("---")
    
    st.markdown("### Explore Platform Capabilities")
    
    def change_page(page_name):
        st.session_state.active_page = page_name
        
    col_c, col_f, col_n = st.columns(3)
    with col_c:
        st.markdown("#### 🏏 Cricket Analytics")
        st.write("IPL match outcome prediction, Toss Impact Analysis, Venue performance trends, Batsman Strike-Rates, and Bowler Economy stats.")
            
    with col_f:
        st.markdown("#### ⚽ Football Analytics")
        st.write("European league standings, Match prediction models, Goal distribution analysis, and Home vs Away performance statistics.")
            
    with col_n:
        st.markdown("#### 🏀 NBA Analytics")
        st.write("NBA match outcome predictions, Player Efficiency Ratings (EFF), Offensive vs Defensive team ratings, and Player comparisons.")

    # Place buttons in a new row of columns to force perfect horizontal alignment
    col_btn_c, col_btn_f, col_btn_n = st.columns(3)
    with col_btn_c:
        st.button("Open Cricket Tab", on_click=change_page, args=("🏏 Cricket (IPL) Hub",))
    with col_btn_f:
        st.button("Open Football Tab", on_click=change_page, args=("⚽ Football Hub",))
    with col_btn_n:
        st.button("Open NBA Tab", on_click=change_page, args=("🏀 NBA Hub",))


# ─────────────────────────────────────────────
# 4. CRICKET ANALYTICS HUB
# ─────────────────────────────────────────────

elif app_mode == "🏏 Cricket (IPL) Hub":
    st.markdown("<h1 class='gradient-text'>🏏 Cricket Analytics Hub</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔮 Match Predictor", "📊 Team & Venue Analytics", "🏃 Player Performance"])
    
    # TAB 1: PREDICTION
    with tab1:
        st.subheader("IPL Match Winner Prediction Engine")
        
        if not models.get('cricket'):
            st.error("Cricket prediction models are currently unavailable. Ensure the models are trained and saved.")
        else:
            c_models = models['cricket']['models']
            le_team = models['cricket']['le_team']
            feature_names = models['cricket']['feature_names']
            
            teams = sorted(list(le_team.classes_))
            venues = sorted(data['cricket_venue_stats']['venue'].unique()) if 'cricket_venue_stats' in data else ['Unknown']
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                team1 = st.selectbox("Select Team 1 (Home/Nominal Home)", teams, index=0)
                team2 = st.selectbox("Select Team 2 (Visitor)", teams, index=1 if len(teams) > 1 else 0)
                venue = st.selectbox("Select Venue", venues)
            with col_p2:
                toss_winner = st.selectbox("Select Toss Winner", [team1, team2])
                toss_decision = st.selectbox("Select Toss Decision", ["bat", "field"])
                selected_model_name = st.selectbox("Choose Classification Model", list(c_models.keys()))
                
            if team1 == team2:
                st.error("Error: Team 1 and Team 2 must be different.")
            else:
                if st.button("Predict Match Winner", key="predict_cricket"):
                    # Load model predictor
                    from src.cricket_pipeline import predict_match
                    pred_res = predict_match(
                        team1, team2, toss_winner, toss_decision, venue,
                        le_team, c_models, feature_names, data['cricket_matches']
                    )
                    
                    if 'error' in pred_res:
                        st.error(f"Prediction failed: {pred_res['error']}")
                    else:
                        prediction = pred_res[selected_model_name]
                        winner = prediction['winner']
                        confidence = prediction['confidence']
                        prob_team1 = prediction['prob_team1']
                        
                        st.write("---")
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            st.success(f"### Predicted Winner: **{winner}**")
                            st.metric("Model Confidence", f"{confidence:.1f}%")
                        with col_r2:
                            # Show gauge/probability chart
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = prob_team1,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': f"Probability of {team1} Winning", 'font': {'size': 18}},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#6366f1"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                                        {'range': [50, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                                    ],
                                }
                            ))
                            fig.update_layout(height=280, margin=dict(t=30, b=0, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"})
                            st.plotly_chart(fig, use_container_width=True)
                            
                        # Dynamic narrative insights
                        from src.insights_engine import generate_cricket_insights
                        insights = generate_cricket_insights(data['cricket_matches'], winner)
                        st.markdown("#### AI Analyst Observations")
                        for ins in insights:
                            st.markdown(f"- 💡 {ins}")

    # TAB 2: TEAM & VENUE ANALYTICS
    with tab2:
        st.subheader("Team Performance and Venue Statistics")
        
        # Win Rates of all teams
        if 'cricket_team_stats' in data:
            df_team = data['cricket_team_stats']
            fig_team = px.bar(
                df_team, x='team', y='win_rate',
                title="IPL Team Overall Win Rates",
                labels={'win_rate': 'Win Rate', 'team': 'Team'},
                color='win_rate',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig_team.update_layout(xaxis_tickangle=-45, height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"})
            st.plotly_chart(fig_team, use_container_width=True)
            
            # Show interactive filter for Team Comparison
            st.write("---")
            st.subheader("Compare Team Statistics Side-by-Side")
            col_tc1, col_tc2 = st.columns(2)
            with col_tc1:
                tc1 = st.selectbox("Select Team A", df_team['team'].unique(), index=0)
            with col_tc2:
                tc2 = st.selectbox("Select Team B", df_team['team'].unique(), index=1 if len(df_team['team'].unique()) > 1 else 0)
                
            if tc1 != tc2:
                t1_data = df_team[df_team['team'] == tc1].iloc[0]
                t2_data = df_team[df_team['team'] == tc2].iloc[0]
                
                # Radar or simple comparative bar chart
                comp_df = pd.DataFrame([
                    {"Metric": "Win Rate", tc1: t1_data['win_rate'], tc2: t2_data['win_rate']},
                    {"Metric": "Toss Win Rate", tc1: t1_data['toss_win_rate'], tc2: t2_data['toss_win_rate']},
                    {"Metric": "Recent Form", tc1: t1_data['recent_form'], tc2: t2_data['recent_form']}
                ])
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(x=comp_df['Metric'], y=comp_df[tc1], name=tc1, marker_color="#6366f1"))
                fig_comp.add_trace(go.Bar(x=comp_df['Metric'], y=comp_df[tc2], name=tc2, marker_color="#10b981"))
                fig_comp.update_layout(
                    barmode='group', title=f"Head to Head: {tc1} vs {tc2}",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"}
                )
                st.plotly_chart(fig_comp, use_container_width=True)

        # Venue Analysis
        if 'cricket_venue_stats' in data:
            st.write("---")
            st.subheader("Venue Performance: Bat First vs Chasing wins")
            df_v = data['cricket_venue_stats'].head(15) # Top 15 venues
            
            fig_venue = go.Figure()
            fig_venue.add_trace(go.Bar(x=df_v['venue'], y=df_v['batting_first_wins'], name='Batting First Wins', marker_color='#a855f7'))
            fig_venue.add_trace(go.Bar(x=df_v['venue'], y=df_v['chasing_wins'], name='Chasing Wins', marker_color='#3b82f6'))
            fig_venue.update_layout(
                barmode='stack', title="Top 15 Stadiums: Toss/Innings Impact",
                xaxis_tickangle=-45, height=500,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"}
            )
            st.plotly_chart(fig_venue, use_container_width=True)

    # TAB 3: PLAYER PERFORMANCE
    with tab3:
        st.subheader("Player Batting & Bowling Analysis")
        
        # Batter Stats
        if 'cricket_player_stats' in data:
            st.markdown("#### Batting Leaderboard (Strike Rate vs Total Runs)")
            df_p = data['cricket_player_stats']
            
            # Show top 50
            fig_bat = px.scatter(
                df_p.head(50), x='total_runs', y='strike_rate',
                size='innings', hover_name='player',
                labels={'total_runs': 'Total Runs', 'strike_rate': 'Strike Rate'},
                color='strike_rate',
                color_continuous_scale=px.colors.sequential.Plasma
            )
            fig_bat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"})
            st.plotly_chart(fig_bat, use_container_width=True)

        # Bowler Stats
        if 'cricket_bowl_stats' in data:
            st.write("---")
            st.markdown("#### Bowling Performance (Economy Rate vs Wickets)")
            df_b = data['cricket_bowl_stats']
            
            fig_bowl = px.scatter(
                df_b.head(50), x='wickets', y='economy',
                size='overs', hover_name='bowler',
                labels={'wickets': 'Wickets Taken', 'economy': 'Economy Rate (lower is better)'},
                color='economy',
                color_continuous_scale=px.colors.sequential.Viridis_r
            )
            fig_bowl.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"})
            st.plotly_chart(fig_bowl, use_container_width=True)


# ─────────────────────────────────────────────
# 5. FOOTBALL ANALYTICS HUB
# ─────────────────────────────────────────────

elif app_mode == "⚽ Football Hub":
    st.markdown("<h1 class='gradient-text'>⚽ Football Analytics Hub</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔮 Outcome Predictor", "📊 Standings & Home/Away Trends"])
    
    # TAB 1: PREDICTION
    with tab1:
        st.subheader("European Match Predictor (Home / Draw / Away)")
        
        if not models.get('football'):
            st.error("Football prediction models are currently unavailable. Ensure the models are trained and saved.")
        else:
            f_models = models['football']['models']
            feature_names = models['football']['feature_names']
            
            teams = sorted(list(data['football_team_perf']['team'].unique())) if 'football_team_perf' in data else []
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                home_team = st.selectbox("Select Home Team", teams, index=0)
            with col_f2:
                away_team = st.selectbox("Select Away Team", teams, index=1 if len(teams) > 1 else 0)
                
            if home_team == away_team:
                st.error("Error: Home Team and Away Team must be different.")
            else:
                selected_model_name = st.selectbox("Choose Classification Model", list(f_models.keys()), key="model_football")
                
                if st.button("Predict Match Outcome", key="predict_football"):
                    # Get team attributes to predict
                    model = f_models[selected_model_name]['model']
                    
                    # Fetch metrics from precomputed team stats
                    t_perf = data['football_team_perf']
                    h_api = t_perf[t_perf['team'] == home_team].iloc[0]['team_id']
                    a_api = t_perf[t_perf['team'] == away_team].iloc[0]['team_id']
                    
                    # Dummy/default attributes if team attributes are missing
                    # feature_names: ['home_team_api_id', 'away_team_api_id', 'home_speed', 'home_chance', 'home_def_press', 'home_def_agg', 'away_speed', 'away_chance', 'away_def_press', 'away_def_agg', 'year']
                    # We can load these attributes from the matches dataframe
                    matches = data['football_matches']
                    
                    def get_team_attrs(team_api_id):
                        # Filter matches where they played home
                        h_matches = matches[matches['home_team'] == home_team]
                        if not h_matches.empty:
                            row = h_matches.iloc[0]
                            return [row.get('home_speed', 50), row.get('home_chance', 50), row.get('home_def_press', 50), row.get('home_def_agg', 50)]
                        # Or away
                        a_matches = matches[matches['away_team'] == home_team]
                        if not a_matches.empty:
                            row = a_matches.iloc[0]
                            return [row.get('away_speed', 50), row.get('away_chance', 50), row.get('away_def_press', 50), row.get('away_def_agg', 50)]
                        return [50, 50, 50, 50]
                        
                    home_attrs = get_team_attrs(h_api)
                    away_attrs = get_team_attrs(a_api)
                    
                    row_data = pd.DataFrame([[
                        h_api, a_api,
                        home_attrs[0], home_attrs[1], home_attrs[2], home_attrs[3],
                        away_attrs[0], away_attrs[1], away_attrs[2], away_attrs[3],
                        2016 # Default/final year in sqlite DB
                    ]], columns=feature_names)
                    
                    probs = model.predict_proba(row_data)[0]
                    # Outcomes in model are encoded as: 2=H, 1=D, 0=A
                    # So probs are [P(Away), P(Draw), P(Home)]
                    
                    st.write("---")
                    col_fr1, col_fr2 = st.columns(2)
                    with col_fr1:
                        # Determine winning outcome
                        highest_idx = np.argmax(probs)
                        outcome_map = {2: f"{home_team} Wins (Home)", 1: "Draw Match", 0: f"{away_team} Wins (Away)"}
                        st.success(f"### Predicted Outcome: **{outcome_map[highest_idx]}**")
                        st.metric("Model Probability Score", f"{probs[highest_idx]*100:.1f}%")
                    with col_fr2:
                        # Radial/Bar distribution of results
                        fig_probs = go.Figure(go.Bar(
                            x=['Away Win', 'Draw', 'Home Win'],
                            y=[probs[0]*100, probs[1]*100, probs[2]*100],
                            marker_color=['#ef4444', '#94a3b8', '#10b981']
                        ))
                        fig_probs.update_layout(
                            title="Probability Distribution (%)",
                            yaxis={'range': [0, 100]},
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"}
                        )
                        st.plotly_chart(fig_probs, use_container_width=True)
                        
                    # AI Observations
                    from src.insights_engine import generate_football_insights
                    insights = generate_football_insights(data['football_matches'], home_team)
                    st.markdown("#### AI Analyst Observations")
                    for ins in insights:
                        st.markdown(f"- 💡 {ins}")

    # TAB 2: STANDINGS & HOME/AWAY TRENDS
    with tab2:
        st.subheader("Standings & League Visualizations")
        
        leagues = sorted(list(data['football_leagues']['name'].unique())) if 'football_leagues' in data else []
        selected_league = st.selectbox("Select League", leagues)
        
        # Calculate standings dynamically using league_standings function
        from src.football_pipeline import league_standings
        standings = league_standings(data['football_matches'], selected_league)
        
        if standings.empty:
            st.warning("No standings data available for this league.")
        else:
            st.dataframe(standings.set_index('Rank'), use_container_width=True)
            
            # Standings bar chart
            fig_pts = px.bar(
                standings, x='Team', y='Pts',
                title=f"{selected_league} Points Standings",
                color='Pts', color_continuous_scale='Bluered_r'
            )
            fig_pts.update_layout(xaxis_tickangle=-45, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"})
            st.plotly_chart(fig_pts, use_container_width=True)
            
        # Home vs Away win percentages by league
        if 'football_ha' in data:
            st.write("---")
            st.subheader("Home vs Away Goal & Win Distribution Across Leagues")
            df_ha = data['football_ha']
            
            fig_ha = go.Figure()
            fig_ha.add_trace(go.Bar(x=df_ha['league'], y=df_ha['home_win_pct'], name='Home Win %', marker_color='#10b981'))
            fig_ha.add_trace(go.Bar(x=df_ha['league'], y=df_ha['draw_pct'], name='Draw %', marker_color='#94a3b8'))
            fig_ha.add_trace(go.Bar(x=df_ha['league'], y=df_ha['away_win_pct'], name='Away Win %', marker_color='#ef4444'))
            fig_ha.update_layout(
                barmode='stack', title="Home vs Away Win Breakdown",
                xaxis_tickangle=-45, height=500,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"}
            )
            st.plotly_chart(fig_ha, use_container_width=True)


# ─────────────────────────────────────────────
# 6. NBA ANALYTICS HUB
# ─────────────────────────────────────────────

elif app_mode == "🏀 NBA Hub":
    st.markdown("<h1 class='gradient-text'>🏀 NBA Analytics Hub</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔮 Game Predictor", "📊 Team Offense vs Defense", "🏃 Player Efficiency & Comparison"])
    
    # TAB 1: GAME PREDICTION
    with tab1:
        st.subheader("NBA Game Winner Prediction Engine")
        
        if not models.get('nba'):
            st.error("NBA prediction models are currently unavailable. Ensure the models are trained and saved.")
        else:
            n_models = models['nba']['models']
            le_team = models['nba']['le_team']
            feature_names = models['nba']['feature_names']
            team_full_map = models['nba']['team_full_map']
            team_abbr_map = models['nba']['team_abbr_map']
            
            teams_options = sorted([{"id": tid, "name": name} for tid, name in team_full_map.items()], key=lambda x: x['name'])
            
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                home_team_sel = st.selectbox("Select Home Team", teams_options, format_func=lambda x: x['name'], index=0)
            with col_n2:
                away_team_sel = st.selectbox("Select Away Team", teams_options, format_func=lambda x: x['name'], index=1 if len(teams_options) > 1 else 0)
                
            if home_team_sel['id'] == away_team_sel['id']:
                st.error("Error: Home Team and Away Team must be different.")
            else:
                selected_model_name = st.selectbox("Choose Classification Model", list(n_models.keys()), key="model_nba")
                
                if st.button("Predict Game Winner", key="predict_nba"):
                    # Extract last 10 games rolling averages for features
                    # Feature names: ['HOME_TEAM_enc', 'VISITOR_TEAM_enc', 'SEASON',
                    #                 'roll_pts_home', 'roll_fg_pct_home', 'roll_fg3_pct_home', 'roll_ast_home', 'roll_reb_home', 'roll_win_rate_home',
                    #                 'roll_pts_away', 'roll_fg_pct_away', 'roll_fg3_pct_away', 'roll_ast_away', 'roll_reb_away', 'roll_win_rate_away']
                    model = n_models[selected_model_name]['model']
                    
                    h_enc = le_team.transform([home_team_sel['id']])[0]
                    a_enc = le_team.transform([away_team_sel['id']])[0]
                    
                    # Fetch rolling attributes from games dataset
                    games = data['nba_games']
                    
                    def get_rolling_stats(team_name):
                        # Filter where they played as home/away and pick latest
                        h_games = games[games['home_team_name'] == team_name]
                        a_games = games[games['away_team_name'] == team_name]
                        # Combine and find latest game date
                        all_t_games = pd.concat([h_games, a_games]).sort_values('GAME_DATE_EST')
                        if not all_t_games.empty:
                            latest_row = all_t_games.iloc[-1]
                            # If they were home team in latest, get home_roll stats, else away_roll stats
                            if latest_row['home_team_name'] == team_name:
                                return [
                                    latest_row.get('roll_pts_home', 100), latest_row.get('roll_fg_pct_home', 0.45),
                                    latest_row.get('roll_fg3_pct_home', 0.35), latest_row.get('roll_ast_home', 22),
                                    latest_row.get('roll_reb_home', 42), latest_row.get('roll_win_rate_home', 0.5)
                                ]
                            else:
                                return [
                                    latest_row.get('roll_pts_away', 100), latest_row.get('roll_fg_pct_away', 0.45),
                                    latest_row.get('roll_fg3_pct_away', 0.35), latest_row.get('roll_ast_away', 22),
                                    latest_row.get('roll_reb_away', 42), latest_row.get('roll_win_rate_away', 0.5)
                                ]
                        return [100, 0.45, 0.35, 22, 42, 0.5]
                        
                    h_roll = get_rolling_stats(home_team_sel['name'])
                    a_roll = get_rolling_stats(away_team_sel['name'])
                    
                    row_data = pd.DataFrame([[
                        h_enc, a_enc, 2022, # SEASON
                        h_roll[0], h_roll[1], h_roll[2], h_roll[3], h_roll[4], h_roll[5],
                        a_roll[0], a_roll[1], a_roll[2], a_roll[3], a_roll[4], a_roll[5]
                    ]], columns=feature_names)
                    
                    prob = model.predict_proba(row_data)[0]
                    # prob[1] is probability of Home Team Win
                    winner_name = home_team_sel['name'] if prob[1] > 0.5 else away_team_sel['name']
                    confidence = max(prob) * 100
                    
                    st.write("---")
                    col_nr1, col_nr2 = st.columns(2)
                    with col_nr1:
                        st.success(f"### Predicted Winner: **{winner_name}**")
                        st.metric("Model Confidence Score", f"{confidence:.1f}%")
                    with col_nr2:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob[1]*100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"Probability of {home_team_sel['name']} Winning", 'font': {'size': 16}},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#6366f1"},
                                'steps': [
                                    {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                                    {'range': [50, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                                ],
                            }
                        ))
                        fig.update_layout(height=280, margin=dict(t=30, b=0, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"})
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # AI Insights
                    from src.insights_engine import generate_nba_insights
                    insights = generate_nba_insights(data['nba_games'], home_team_sel['id'], home_team_sel['name'])
                    st.markdown("#### AI Analyst Observations")
                    for ins in insights:
                        st.markdown(f"- 💡 {ins}")

    # TAB 2: TEAM OFFENSE VS DEFENSE
    with tab2:
        st.subheader("Offensive vs Defensive ratings (Average PPG Scored vs Conceded)")
        
        if 'nba_team_ratings' in data:
            df_r = data['nba_team_ratings']
            
            fig_ratings = px.scatter(
                df_r, x='off_rating', y='def_rating',
                hover_name='team', text='team',
                labels={'off_rating': 'Offensive Rating (Points Scored/Game)', 'def_rating': 'Defensive Rating (Points Conceded/Game)'},
                color='net_rating', color_continuous_scale='RdYlGn_r'
            )
            # Standardize quadrant
            fig_ratings.update_layout(
                yaxis={'autorange': "reversed"}, # Lower defensive rating (fewer points conceded) is better
                title="Offensive vs Defensive Efficiency Landscape (Top Right is Elite)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"}
            )
            st.plotly_chart(fig_ratings, use_container_width=True)

    # TAB 3: PLAYER EFFICIENCY
    with tab3:
        st.subheader("Player Efficiency & MVP Rankings")
        
        if 'nba_player_eff' in data:
            df_pe = data['nba_player_eff']
            
            # Show MVP Leaderboard
            st.markdown("#### SportSphere MVP Rating Leaderboard")
            st.dataframe(df_pe[['PLAYER_NAME', 'gp', 'avg_pts', 'avg_eff', 'MVP_score']].head(20).rename(
                columns={'PLAYER_NAME': 'Player', 'gp': 'Games Played', 'avg_pts': 'PPG', 'avg_eff': 'EFF Rating', 'MVP_score': 'SportSphere MVP Index'}
            ), use_container_width=True)
            
            # Comparison section
            st.write("---")
            st.subheader("Player Comparison Radar / Metrics")
            
            players_list = sorted(list(data['nba_player_summary']['PLAYER_NAME'].unique()))
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                p1 = st.selectbox("Select Player A", players_list, index=0)
            with col_comp2:
                p2 = st.selectbox("Select Player B", players_list, index=1 if len(players_list) > 1 else 0)
                
            if p1 != p2:
                p1_data = data['nba_player_summary'][data['nba_player_summary']['PLAYER_NAME'] == p1].iloc[0]
                p2_data = data['nba_player_summary'][data['nba_player_summary']['PLAYER_NAME'] == p2].iloc[0]
                
                # Plotly polar/radar chart comparison
                categories = ['Points/Game', 'Assists/Game', 'Rebounds/Game', 'Steals/Game', 'Blocks/Game']
                p1_metrics = [p1_data['avg_pts'], p1_data['avg_ast'], p1_data['avg_reb'], p1_data['avg_stl'], p1_data['avg_blk']]
                p2_metrics = [p2_data['avg_pts'], p2_data['avg_ast'], p2_data['avg_reb'], p2_data['avg_stl'], p2_data['avg_blk']]
                
                # Normalize metrics for visual radar comparison
                max_values = [35, 12, 15, 3, 3] # Standard peak stats
                p1_norm = [min(val/m*100, 100) for val, m in zip(p1_metrics, max_values)]
                p2_norm = [min(val/m*100, 100) for val, m in zip(p2_metrics, max_values)]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=p1_norm, theta=categories, fill='toself', name=p1, fillcolor='rgba(99, 102, 241, 0.4)', line={'color': '#6366f1'}))
                fig_radar.add_trace(go.Scatterpolar(r=p2_norm, theta=categories, fill='toself', name=p2, fillcolor='rgba(16, 185, 129, 0.4)', line={'color': '#10b981'}))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title=f"Normalized Performance Metric: {p1} vs {p2}",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"}
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Show raw averages in columns
                st.write("")
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.markdown(f"**{p1}** Averages:")
                    st.write(f"- Points: **{p1_data['avg_pts']:.1f}**")
                    st.write(f"- Assists: **{p1_data['avg_ast']:.1f}**")
                    st.write(f"- Rebounds: **{p1_data['avg_reb']:.1f}**")
                with col_stat2:
                    st.markdown(f"**{p2}** Averages:")
                    st.write(f"- Points: **{p2_data['avg_pts']:.1f}**")
                    st.write(f"- Assists: **{p2_data['avg_ast']:.1f}**")
                    st.write(f"- Rebounds: **{p2_data['avg_reb']:.1f}**")


# ─────────────────────────────────────────────
# 7. AI INSIGHTS & CROSS-SPORT
# ─────────────────────────────────────────────

elif app_mode == "🧠 AI Insights & Cross-Sport":
    st.markdown("<h1 class='gradient-text'>🧠 Cross-Sport Analytics & AI Insights</h1>", unsafe_allow_html=True)
    st.markdown("Unify insights across multiple disciplines to study fundamental trends of sports matches, home advantages, scoring distributions, and variance metrics.")
    
    # Load cross-sport data
    from src.insights_engine import load_cross_sport_metrics
    metrics = load_cross_sport_metrics()
    
    st.write("")
    
    st.subheader("1. Home Advantage Matrix")
    st.write("Does playing on home turf offer a uniform advantage across different sports leagues? Let's check win rates.")
    
    # Prepare comparison data
    cross_df = pd.DataFrame([
        {"Sport": "IPL Cricket (Bat First Pct)", "Home/Favor Win Rate (%)": metrics['cricket']['batting_first_win_rate']},
        {"Sport": "European Football (Home Wins)", "Home/Favor Win Rate (%)": metrics['football']['home_win_rate']},
        {"Sport": "NBA Basketball (Home Wins)", "Home/Favor Win Rate (%)": metrics['nba']['home_win_rate']}
    ])
    
    fig_cross = px.bar(
        cross_df, x='Sport', y='Home/Favor Win Rate (%)',
        title="Home/Toss Advantaged Team Win Rate Comparison",
        color='Home/Favor Win Rate (%)', color_continuous_scale='Magma',
        text='Home/Favor Win Rate (%)'
    )
    fig_cross.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"})
    st.plotly_chart(fig_cross, use_container_width=True)
    
    st.markdown("""
        > **Insight Output:**
        > - **NBA Basketball** exhibits the highest home-court advantage (approaching ~59%), likely due to crowd noise, altitude/travel fatigue, and referee bias.
        > - **European Football** shows a solid home advantage (~46% outright home wins, with ~25% draws).
        > - **IPL Cricket** shows that batting first vs chasing is highly balanced (~48% vs ~52%), suggesting that pitch degradation and dew factors level the field over the standard 20 overs.
    """)
    
    st.write("---")
    st.subheader("2. Platform General Summary Stats")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown("**IPL Cricket Info**")
        st.write(f"- Matches analyzed: **{metrics['cricket']['total_sample']:,}**")
        st.write(f"- Toss win influence: **{metrics['cricket']['toss_win_rate']}%** win correlation")
    with col_s2:
        st.markdown("**Football Info**")
        st.write(f"- Matches analyzed: **{metrics['football']['total_sample']:,}**")
        st.write(f"- Away win rate: **{metrics['football']['away_win_rate']}%**")
    with col_s3:
        st.markdown("**NBA Info**")
        st.write(f"- Matches analyzed: **{metrics['nba']['total_sample']:,}**")
        st.write(f"- Road win rate: **{metrics['nba']['away_win_rate']}%**")
