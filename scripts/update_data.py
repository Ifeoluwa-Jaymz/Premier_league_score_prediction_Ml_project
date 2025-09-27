import pandas as pd
from pathlib import Path
import datetime

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

URL_SUMMARY = "https://fbref.com/en/comps/9/Premier-League-Stats"
URL_MATCHES = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"


def download_team_summary(url: str, dated_file: Path, latest_file: Path):
    try:
        tables = pd.read_html(url)
        df = tables[0]
        df.to_csv(dated_file, index=False)
        df.to_csv(latest_file, index=False)
        print(f"Team summary saved to {dated_file} and {latest_file}")
    except Exception as e:
        print(f"Failed to fetch team summary: {e}")


def download_match_results(url: str, dated_file: Path, latest_file: Path):
    try:
        tables = pd.read_html(url)
        df = tables[0]
        df = df[df["Score"].notna()].copy()  # drop unplayed fixtures
        df[["home_goals", "away_goals"]] = df["Score"].str.split("–", expand=True).astype(int)
        df.rename(columns={"Home": "home_team", "Away": "away_team"}, inplace=True)

        df.to_csv(dated_file, index=False)
        df.to_csv(latest_file, index=False)
        print(f"Match results saved to {dated_file} and {latest_file}")
    except Exception as e:
        print(f"Failed to fetch match results: {e}")


def update_data():
    """Fetch and save the latest EPL data."""
    today = datetime.date.today().strftime("%Y-%m-%d")

    summary_dated = DATA_DIR / f"premier_league_summary_{today}.csv"
    matches_dated = DATA_DIR / f"premier_league_matches_{today}.csv"

    summary_latest = DATA_DIR / "premier_league_summary_latest.csv"
    matches_latest = DATA_DIR / "premier_league_matches_latest.csv"

    download_team_summary(URL_SUMMARY, summary_dated, summary_latest)
    download_match_results(URL_MATCHES, matches_dated, matches_latest)


if __name__ == "__main__":
    update_data()
