from pathlib import Path

import pandas as pd
import pytest

from src.explain import MODEL_PATH, analyze_url, contributions, load_model, to_frame

SAMPLES = Path(__file__).with_name("sample_urls.csv")

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run python -m src.train first")


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_predicts_all_samples(model):
    pipe, meta = model
    for url in pd.read_csv(SAMPLES)["url"]:
        res = analyze_url(url, pipe, meta)
        assert 0.0 <= res["proba_phishing"] <= 1.0
        assert isinstance(res["reasons"], list)


def test_contributions_sum_to_probability(model):
    pipe, meta = model
    res = analyze_url("https://login-verify.weeblysite.com", pipe, meta)
    bias, contrib = contributions(pipe, to_frame(res["features"]))
    assert abs(bias + contrib.sum() - res["proba_phishing"]) < 1e-6


def test_obvious_cases(model):
    pipe, meta = model
    assert analyze_url("https://emerald-exch-0utlook.firebaseapp.com", pipe, meta)["is_phishing"]
    assert not analyze_url("https://www.google.com", pipe, meta)["is_phishing"]


def test_invalid_url_raises(model):
    pipe, meta = model
    with pytest.raises(ValueError):
        analyze_url("not a url", pipe, meta)
