# SportSphere AI — Multi-Sport Analytics & Prediction Platform

SportSphere AI is a production-grade, end-to-end sports analytics and match outcome prediction platform. Built in Python, it integrates machine learning engines, automated feature engineering, and statistical modeling with an interactive web dashboard. It covers **Cricket (IPL)**, **Football (European Leagues)**, and **NBA Basketball** datasets to provide high-fidelity prediction probabilities, player ranking systems (MVP Index), efficiency stats, head-to-head comparisons, and cross-sport analytics.

## 🚀 Key Features

*   **Multi-Sport Prediction Engines**: Leverages multiple machine learning models (XGBoost, Random Forest, Logistic Regression) to predict match winners, complete with confidence dials and probability distributions.
*   **Feature Engineering Pipelines**: Automatically computes complex rolling metrics (10-game team averages, toss impact multipliers, venue fortresses) to capture real-time team momentum without data leakage.
*   **Player Efficiency Rating (EFF & MVP Index)**: Uses standard NBA EFF metrics and designs a custom multi-factor MVP selection index blending efficiency, game volume, and scoring.
*   **Head-to-Head Comparisons**: Side-by-side comparative tools for teams and players, including automated radar charts comparing key athletic metrics.
*   **AI-Generated Observations**: Instantly compiles natural language analytical observations explaining team traits (e.g., toss dependency, home turf biases, chasing vs defending margins).
*   **Unified Cross-Sport Dashboard**: Aggregates home-ground advantage values across Cricket, Football, and NBA.

---

## 🛠️ Tech Stack & Libraries

*   **Core Logic & Analytics**: Python 3.11+, Pandas, NumPy, SciPy
*   **Machine Learning**: Scikit-Learn, XGBoost
*   **Data Visualization**: Plotly, Matplotlib, Seaborn
*   **Frontend UI**: Streamlit (with custom HSL CSS styling and glassmorphism cards)

---

## 📂 Project Structure

```text
sportsphere-ai/
│
├── data/
│   ├── cricket/           # Raw Cricket match & ball data
│   ├── football/          # SQLite European Soccer Database
│   ├── nba/               # NBA games, players, rankings, details
│   └── processed/         # Pre-computed lightweight datasets for Streamlit
│
├── src/
│   ├── cricket_pipeline.py# Data cleaning, feature computation, cricket ML models
│   ├── football_pipeline.py# SQL ingestion, cleaning, league standing, football ML models
│   ├── nba_pipeline.py     # Rolling average, player efficiency, NBA ML models
│   ├── insights_engine.py  # AI insight narrative compiler & cross-sport statistics
│   └── train_and_save.py   # One-click training and pre-computation script
│
├── models/
│   ├── cricket_models.pkl  # Serialized cricket ML models
│   ├── football_models.pkl # Serialized football ML models
│   └── nba_models.pkl      # Serialized NBA ML models
│
├── requirements.txt        # Package dependencies
├── README.md               # Documentation
└── app.py                  # Dashboard Entrypoint
```

---

## ⚙️ How it Works (The Data Pipeline)

### 1. Data Ingestion & Cleaning
- **Cricket**: Cleans IPL matches and ball-by-ball deliveries, normalizing historical team name changes (e.g., Delhi Daredevils ➔ Delhi Capitals) and handling missing cities and umpire attributes.
- **Football**: Connects to the SQLite European Soccer Database, extracts matches, country names, leagues, and merges historical team attributes (e.g., Build-up speed, defence aggression).
- **NBA**: Ingests games, player stats, and league rankings, converting data types, cleaning match dates, and handling missing points or minutes values.

### 2. Feature Engineering
- **Rolling Metrics (NBA)**: Computes 10-game rolling averages for points scored, field-goal %, rebound rates, and win rates, shifted by 1 game to avoid future data leakage.
- **Venue and Toss (Cricket)**: Aggregates historical match outcomes per venue to identify the batting-first vs chasing win percentage. Computes toss win correlation ratios.
- **Player Metrics**: Computes Cricket batter strike-rates, bowler economy rates, and NBA Player Efficiency Rating (EFF) based on scoring, rebounds, assists, steals, blocks, turnovers, and missed shots.

### 3. Model Training & Evaluation
The platform trains three models for each sport module:
- **Logistic Regression**: Linear baseline with high interpretability.
- **Random Forest**: High capability of capturing non-linear feature splits.
- **XGBoost**: Extreme gradient boosting for top-tier classification accuracy.

Models are evaluated on holdout sets (80/20 train/test split) utilizing accuracy score, confusion matrices, and precision/recall metrics.

---

## 🚀 Setup & Execution

### 1. Prerequisites
Install [Python 3.11+](https://www.python.org/downloads/).

### 2. Installation & Library Setup
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run Training & Feature Pre-computation
Before starting the app, run the automated training script to fit all models and output lightweight, cached CSVs. This allows the app to load instantly:
```bash
python src/train_and_save.py
```

### 4. Run the Streamlit Dashboard
Launch the local web server:
```bash
streamlit run app.py
```

---

## 📊 Sample Model Performance Results

- **NBA Game Winner**: Peak validation accuracy **63.2%** (XGBoost)
- **IPL Match Winner**: Peak validation accuracy **54.1%** (Logistic Regression)
- **Football Match Outcome (3-way)**: Peak validation accuracy **50.7%** (XGBoost)
