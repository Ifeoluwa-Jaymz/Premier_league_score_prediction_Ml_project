from sklearn.linear_model import LogisticRegression
import numpy as np
from .base_model import BaseModel

class LogisticMatchModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, multi_class="ovr")

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
