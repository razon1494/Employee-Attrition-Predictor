import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RatioFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "MonthlyIncome" in X.columns and "TotalWorkingYears" in X.columns:
            X["IncomePerYearExp"] = X["MonthlyIncome"] / (X["TotalWorkingYears"] + 1)

        if "YearsAtCompany" in X.columns and "TotalWorkingYears" in X.columns:
            X["CompanyTenureRatio"] = X["YearsAtCompany"] / (X["TotalWorkingYears"] + 1)

        if "Age" in X.columns and "TotalWorkingYears" in X.columns:
            X["YearsPerAge"] = X["TotalWorkingYears"] / (X["Age"] + 1)

        return X


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.q1_ = X.quantile(0.25)
        self.q3_ = X.quantile(0.75)
        self.iqr_ = self.q3_ - self.q1_
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        lower = self.q1_ - self.factor * self.iqr_
        upper = self.q3_ + self.factor * self.iqr_
        return X.clip(lower=lower, upper=upper, axis=1)
