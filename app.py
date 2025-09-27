import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.models.poisson_model import PoissonMatchModel
from src.feature_engineering import prepare_match_data_for_regression
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import poisson
import seaborn as sns

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("data/premier_league_matches_latest.csv")
    df = prepare_match_data_for_regression(df)
    return df

# Train Poisson model
@st.cache_resource
def train_model(df):
    feature_cols = ["home_attack_strength", "away_attack_strength", "xG_diff"]

    X = df[feature_cols].fillna(0)
    y_home = df["home_goals"]
    y_away = df["away_goals"]

    X_train, _, y_home_train, _, y_away_train, _ = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42
    )

    model = PoissonMatchModel()
    model.train(X_train, y_home_train, y_away_train)
    return model

# Predict outcome
def predict_match(model, df, home_team, away_team, max_goals=6):
    home_last = df[df["home_team"] == home_team].tail(1)
    away_last = df[df["away_team"] == away_team].tail(1)

    if home_last.empty or away_last.empty:
        return None, None, None, None

    feature_row = pd.DataFrame([{
        "home_attack_strength": home_last["home_attack_strength"].values[0],
        "away_attack_strength": away_last["away_attack_strength"].values[0],
        "xG_diff": home_last["xG_diff"].values[0],
    }])

    preds = model.predict(feature_row)
    probs = model.predict_outcome_probabilities(feature_row, max_goals=max_goals)

    # Compute full goal probability matrix
    lambda_home = preds["home_goals_pred"].values[0]
    lambda_away = preds["away_goals_pred"].values[0]

    home_dist = [poisson.pmf(i, lambda_home) for i in range(max_goals+1)]
    away_dist = [poisson.pmf(i, lambda_away) for i in range(max_goals+1)]
    matrix = np.outer(home_dist, away_dist)

    # Most likely scoreline
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    predicted_scoreline = (max_idx[0], max_idx[1])  # (home_goals, away_goals)

    return preds.iloc[0], probs.iloc[0], predicted_scoreline, matrix

# Streamlit UI
st.title("Premier League Match Predictor")

df = load_data()
model = train_model(df)

teams = sorted(df["home_team"].unique())
home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

if st.button("Predict Outcome"):
    if home_team == away_team:
        st.warning("Please choose two different teams!")
    else:
        preds, probs, scoreline, matrix = predict_match(model, df, home_team, away_team)

        if preds is None:
            st.error("Not enough data for this matchup.")
        else:
            st.subheader(f"Prediction: {home_team} vs {away_team}")
            st.write(f"Expected Goals → {home_team}: {preds['home_goals_pred']:.2f}, {away_team}: {preds['away_goals_pred']:.2f}")

            st.metric(label=f"{home_team} Win Probability", value=f"{probs['P_HomeWin']:.1%}")
            st.metric(label="Draw Probability", value=f"{probs['P_Draw']:.1%}")
            st.metric(label=f"{away_team} Win Probability", value=f"{probs['P_AwayWin']:.1%}")

            st.success(f"Most likely scoreline: {home_team} {scoreline[0]} – {scoreline[1]} {away_team}")

            # Plot probability bar chart
            st.subheader("Outcome Probabilities")
            fig, ax = plt.subplots()
            labels = [f"{home_team} Win", "Draw", f"{away_team} Win"]
            values = [probs['P_HomeWin'], probs['P_Draw'], probs['P_AwayWin']]
            ax.bar(labels, values, color=["green", "gray", "red"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            st.pyplot(fig)
