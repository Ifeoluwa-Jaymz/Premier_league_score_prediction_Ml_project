import pandas as pd
from src.models.poisson_model import PoissonMatchModel
from src.feature_engineering import prepare_match_data_for_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def predict_match(model, df, home_team: str, away_team: str):
    """Predict outcome probabilities for a given matchup."""
    # Filter last row for each team (latest form)
    home_last = df[df["home_team"] == home_team].tail(1)
    away_last = df[df["away_team"] == away_team].tail(1)

    if home_last.empty or away_last.empty:
        print(f" Not enough data for {home_team} vs {away_team}")
        return None

    # Build feature row
    feature_row = pd.DataFrame([{
        "home_attack_strength": home_last["home_attack_strength"].values[0],
        "away_attack_strength": away_last["away_attack_strength"].values[0],
        "xG_diff": home_last["xG_diff"].values[0],
    }])

    # Predict outcome probabilities
    probs = model.predict_outcome_probabilities(feature_row)
    print(f"\n Prediction for {home_team} vs {away_team}:")
    print(probs.iloc[0])

    return probs.iloc[0]

def main():
    # Load data
    df = pd.read_csv("data/premier_league_matches_latest.csv")
    df = prepare_match_data_for_regression(df)

    feature_cols = ["home_attack_strength", "away_attack_strength", "xG_diff"]
    X = df[feature_cols].fillna(0)
    y_home = df["home_goals"]
    y_away = df["away_goals"]

    # Train/test split
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42
    )

    # Train Poisson model
    model = PoissonMatchModel()
    model.train(X_train, y_home_train, y_away_train)

    # Evaluate
    preds = model.predict(X_test)
    mae_home = mean_absolute_error(y_home_test, preds["home_goals_pred"])
    mae_away = mean_absolute_error(y_away_test, preds["away_goals_pred"])
    print(" Mean Absolute Error:")
    print(f"   Home Goals: {mae_home:.3f}")
    print(f"   Away Goals: {mae_away:.3f}")

    # Example: Arsenal vs Chelsea
    predict_match(model, df, "Arsenal", "Chelsea")

if __name__ == "__main__":
    main()
