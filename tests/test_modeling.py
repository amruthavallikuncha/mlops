import numpy as np

from heart_risk.modeling import candidate_models


def test_candidate_pipelines_fit_and_return_probabilities(sample_frame):
    labels = np.array([0, 1])
    for pipeline, _ in candidate_models().values():
        pipeline.fit(sample_frame, labels)
        probabilities = pipeline.predict_proba(sample_frame)
        assert probabilities.shape == (2, 2)
