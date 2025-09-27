import pandas as pd
from pathlib import Path
import datetime

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

URL_SUMMARY = "https://fbref.com/en/comps/9/Premier-League-Stats"
URL_MATCHES = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"


def download_team_summary(url: str, dated_file: Path, latest_file: Path):
    """Download Premier League team summary stats."""
    try:
        tables = pd.read_html(url)
        df = tables[0]
        df.to_csv(dated_file, index=False)
        df.to_csv(latest_file, index=False)
        print(f"Team summary saved to {dated_file} and {latest_file}")
    except Exception as e:
        print(f"Failed to fetch team summary: {e}")


def download_match_results(url: str, dated_file: Path, latest_file: Path):
    """Download Premier League match results and clean them."""
    try:
        tables = pd.read_html(url)
        df = tables[0]

        # Drop rows without scores yet
        df = df[df["Score"].notna()].copy()

        # Split Score column
        df[["home_goals", "away_goals"]] = df["Score"].str.split("–", expand=True).astype(int)

        # Rename for consistency
        df.rename(columns={"Home": "home_team", "Away": "away_team"}, inplace=True)

        # Save both versions
        df.to_csv(dated_file, index=False)
        df.to_csv(latest_file, index=False)
        print(f"Match results saved to {dated_file} and {latest_file}")
    except Exception as e:
        print(f"Failed to fetch match results: {e}")


if __name__ == "__main__":
    today = datetime.date.today().strftime("%Y-%m-%d")

    # File paths
    summary_dated = DATA_DIR / f"premier_league_summary_{today}.csv"
    matches_dated = DATA_DIR / f"premier_league_matches_{today}.csv"

    summary_latest = DATA_DIR / "premier_league_summary_latest.csv"
    matches_latest = DATA_DIR / "premier_league_matches_latest.csv"

    # Run downloads
    download_team_summary(URL_SUMMARY, summary_dated, summary_latest)
    download_match_results(URL_MATCHES, matches_dated, matches_latest)
