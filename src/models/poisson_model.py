import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import poisson

class PoissonMatchModel:
    """Poisson regression for football scores + outcome probabilities."""

    def __init__(self):
        self.model_home = None
        self.model_away = None
        self.feature_cols = None  # keep track of features

    def train(self, X, y_home, y_away):
        # Add constant column
        X_const = sm.add_constant(X)
        self.feature_cols = X_const.columns

        self.model_home = sm.GLM(y_home, X_const, family=sm.families.Poisson()).fit()
        self.model_away = sm.GLM(y_away, X_const, family=sm.families.Poisson()).fit()

    def _prepare_features(self, X):
        """Ensure test data has same columns as training (incl. constant)."""
        X_const = sm.add_constant(X, has_constant="add")
        # Reorder and fill missing
        X_const = X_const.reindex(columns=self.feature_cols, fill_value=0)
        return X_const

    def predict(self, X):
        X_const = self._prepare_features(X)
        y_home_pred = self.model_home.predict(X_const)
        y_away_pred = self.model_away.predict(X_const)
        return pd.DataFrame({
            "home_goals_pred": y_home_pred,
            "away_goals_pred": y_away_pred
        })

    def predict_outcome_probabilities(self, X, max_goals: int = 10):
        X_const = self._prepare_features(X)
        lambda_home = self.model_home.predict(X_const)
        lambda_away = self.model_away.predict(X_const)

        probs = []
        for l_h, l_a in zip(lambda_home, lambda_away):
            home_dist = [poisson.pmf(i, l_h) for i in range(max_goals+1)]
            away_dist = [poisson.pmf(i, l_a) for i in range(max_goals+1)]
            matrix = np.outer(home_dist, away_dist)

            p_home_win = np.sum(np.tril(matrix, -1))
            p_draw = np.sum(np.diag(matrix))
            p_away_win = np.sum(np.triu(matrix, 1))

            probs.append({
                "P_HomeWin": p_home_win,
                "P_Draw": p_draw,
                "P_AwayWin": p_away_win
            })

        return pd.DataFrame(probs)
