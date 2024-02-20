from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier

class Classifier(BaseEstimator):
    def __init__(self):
        self.model = AdaBoostClassifier(n_estimators=50, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
