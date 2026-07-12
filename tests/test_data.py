import numpy as np
import pandas as pd
import pytest

from heart_risk.config import FEATURES, TARGET
from heart_risk.data import clean_data


def test_clean_data_binarizes_target_and_preserves_missing_features():
    row = dict.fromkeys(FEATURES, 1.0)
    frame = pd.DataFrame([{**row, "ca": "?", "severity": 3}])
    result = clean_data(frame)
    assert result.loc[0, TARGET] == 1
    assert np.isnan(result.loc[0, "ca"])


def test_clean_data_rejects_missing_schema():
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_data(pd.DataFrame({"age": [50]}))


def test_clean_data_removes_duplicate_records():
    row = {**dict.fromkeys(FEATURES, 1.0), "severity": 0}
    result = clean_data(pd.DataFrame([row, row]))
    assert len(result) == 1
    assert result.loc[0, TARGET] == 0
