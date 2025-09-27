import pandas as pd

def prepare_match_data_for_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for EPL score prediction using available columns."""

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")

    # Rename xG columns for clarity
    df = df.rename(columns={"xG": "home_xG", "xG.1": "away_xG"})

    # Derive xGA as opponent's xG
    df["home_xGA"] = df["away_xG"]
    df["away_xGA"] = df["home_xG"]

    # Rolling averages for last 5 matches (form)
    for col in ["home_xG", "home_xGA"]:
        df[f"{col}_rolling"] = (
            df.groupby("home_team")[col].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        )
    for col in ["away_xG", "away_xGA"]:
        df[f"{col}_rolling"] = (
            df.groupby("away_team")[col].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        )

    # Attack/defense strength
    df["home_attack_strength"] = df["home_xG_rolling"] - df["home_xGA_rolling"]
    df["away_attack_strength"] = df["away_xG_rolling"] - df["away_xGA_rolling"]

    # xG difference
    df["xG_diff"] = df["home_xG_rolling"] - df["away_xG_rolling"]

    return df
