import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from heart_risk.config import FEATURES


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    row = [54, 1, 2, 130, 246, 0, 0, 150, 0, 1.2, 1, 0, 3]
    return pd.DataFrame([row, row], columns=FEATURES)


@pytest.fixture
def dummy_artifacts(tmp_path, sample_frame):
    model = DummyClassifier(strategy="prior").fit(sample_frame, [0, 1])
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "test_model",
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
    return model_path, metadata_path
