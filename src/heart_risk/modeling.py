from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_risk.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    logistic = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    forest = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )
    return {
        "logistic_regression": (
            logistic,
            {
                "classifier__C": [0.05, 0.2, 1.0, 5.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            forest,
            {
                "classifier__n_estimators": [150, 300],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_leaf": [1, 3, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }
