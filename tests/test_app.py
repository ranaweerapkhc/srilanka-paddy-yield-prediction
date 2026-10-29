"""Smoke tests for the Streamlit app, run in CI before the Docker image is built."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


def run_app():
    return AppTest.from_file(APP, default_timeout=180).run()


def test_app_loads_and_predicts():
    at = run_app()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Predicted yield (EBM)" in labels


def test_future_season_prediction_is_plausible():
    at = run_app()
    at.selectbox[0].set_value("GALLE")
    at.number_input[0].set_value(2026)
    at.run()
    assert not at.exception
    value = next(m.value for m in at.metric if m.label == "Predicted yield (EBM)")
    kg_ha = float(value.split()[0].replace(",", ""))
    assert 1000 < kg_ha < 8000
