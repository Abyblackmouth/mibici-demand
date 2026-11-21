"""Contratos de simulación: etiquetas, causalidad, tipos, detección y alertas."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from threadpoolctl import threadpool_limits

from mibici_demand import config, features
from mibici_demand.monitoring.drift import (
    DEFAULT_PLAN,
    AlertPolicy,
    Binning,
    compare_distributions,
    make_alerts,
    make_scenario,
    metric_deltas,
    performance_alerts,
    population_stability,
    score_predictions,
    verify_scenario,
)


@pytest.fixture
def control():
    times = pd.date_range("2026-09-07", periods=72, freq="h").repeat(4)
    n = len(times)
    frame = pd.DataFrame(
        {
            "estacion": np.tile(np.arange(1, 5, dtype="int64"), 72),
            "hora_inicio": times,
            "origen_pronostico": times.floor("D"),
            "demanda": (np.arange(n) % 5).astype("int64"),
            "hora": times.hour.astype("int8"),
            "dia_semana": times.day_of_week.astype("int8"),
            "mes": times.month.astype("int8"),
            "es_fin_semana": (times.day_of_week >= 5).astype("int8"),
            "lag_24": (np.arange(n) % 7).astype("float32"),
            "lag_168": (np.arange(n) % 11).astype("float32"),
            "fuente_lag_24": times - pd.Timedelta(hours=24),
            "fuente_lag_168": times - pd.Timedelta(hours=168),
        }
    )
    return frame


@pytest.mark.parametrize("family", ["control", "composicion", "faltantes", "combinado"])
def test_escenarios_reproducibles_preservan_etiquetas_tipos_y_fuentes(control, family):
    original = control.copy(deep=True)
    intensity = 0 if family == "control" else 0.3
    one = make_scenario(control, family, intensity)
    two = make_scenario(control, family, intensity)
    assert_frame_equal(one, two, check_exact=True)
    assert_frame_equal(control, original, check_exact=True)
    verify_scenario(control, one, family)
    np.testing.assert_array_equal(one["demanda"], control.iloc[one["fila_fuente"]]["demanda"])
    assert one["lag_24"].dtype == one["lag_168"].dtype == np.dtype("float32")
    for lag in (24, 168):
        assert (
            (one[f"fuente_lag_{lag}"] + pd.Timedelta(hours=1)) <= one["origen_pronostico"]
        ).all()


def test_composicion_sustituye_exactamente_no_pico_por_registros_completos(control):
    source_peak = control["hora"].isin(config.PEAK_HOURS)
    frame = make_scenario(control, "composicion", 0.5)
    replaced = frame["fila_fuente"].ne(frame["posicion_lote"])
    assert replaced.sum() == int((~source_peak).sum() * 0.5)
    assert frame.loc[replaced, "hora"].isin(config.PEAK_HOURS).all()
    assert not source_peak.loc[replaced].any()
    assert frame["hora"].isin(config.PEAK_HOURS).sum() == source_peak.sum() + replaced.sum()


def test_escenarios_no_consultan_etiquetas_para_muestrear_o_ocultar(control):
    changed = control.copy()
    changed["demanda"] = changed["demanda"] * 1000 + 123
    one = make_scenario(control, "combinado", 0.5)
    two = make_scenario(changed, "combinado", 0.5)
    assert_frame_equal(one.drop(columns="demanda"), two.drop(columns="demanda"), check_exact=True)


@pytest.mark.parametrize("family", ["composicion", "faltantes", "combinado"])
def test_intensidades_anidadas_y_semillas_independientes(control, family):
    frames = [make_scenario(control, family, a) for a in (0.1, 0.3, 0.5)]
    previous = set()
    for frame in frames:
        mask = (
            frame["lag_24"].isna()
            if family != "composicion"
            else frame["fila_fuente"].ne(frame["posicion_lote"])
        )
        current = set(frame.index[mask])
        assert previous < current
        previous = current
    other_seed = make_scenario(control, family, 0.5, seed=99)
    assert not frames[-1].equals(other_seed)


@pytest.mark.parametrize("column", ["demanda", "lag_24", "hora", "fuente_lag_168"])
def test_rechaza_registros_adulterados(control, column):
    frame = make_scenario(control, "combinado", 0.3)
    i = frame.index[frame["lag_24"].notna()][0]
    if column == "fuente_lag_168":
        frame.loc[i, column] += pd.Timedelta(hours=1)
    else:
        frame.loc[i, column] += 1
    with pytest.raises(ValueError):
        verify_scenario(control, frame, "combinado")


@pytest.mark.parametrize(
    "family,intensity",
    [("desconocido", 0.3), ("control", 0.1), ("faltantes", -0.1), ("faltantes", 1.1)],
)
def test_rechaza_planes_de_escenario_invalidos(control, family, intensity):
    with pytest.raises(ValueError):
        make_scenario(control, family, intensity)


def test_control_identico_tiene_psi_ks_y_tv_cero_aunque_cambie_n():
    ref = pd.Series([0.0, 0.0, 2.0, 2.0])
    current = pd.concat([ref] * 3, ignore_index=True)
    bins = Binning.from_reference(ref, categorical=False)
    measured, _ = population_stability(ref, current, bins)
    assert measured == {"psi": 0.0, "distancia_ks": 0.0, "distancia_tv": 0.0}


def test_psi_categorico_contra_calculo_conocido_y_categoria_nueva():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    current = pd.Series(["a"] * 90 + ["b"] * 10)
    bins = Binning.from_reference(ref, categorical=True)
    measured, _ = population_stability(ref, current, bins)
    assert measured["psi"] == pytest.approx(0.4 * np.log(9), abs=1e-5)
    assert measured["distancia_tv"] == pytest.approx(0.4)
    unknown, details = population_stability(ref, pd.Series(["nuevo"] * 100), bins)
    assert unknown["psi"] > 1
    assert details[-1]["bin"] == "DESCONOCIDO" and details[-1]["n_actual"] == 100


@pytest.mark.parametrize("value", [4.0, 6.0])
def test_referencia_constante_detecta_cambios_a_ambos_lados(value):
    ref = pd.Series([5.0] * 100)
    bins = Binning.from_reference(ref, categorical=False)
    measured, _ = population_stability(ref, pd.Series([value] * 100), bins)
    assert measured["psi"] > 0.25 and measured["distancia_ks"] == 1


def test_ks_distancia_conocida_sin_pvalor_y_bins_solo_referencia():
    ref = pd.Series([0.0] * 50 + [10.0] * 50)
    current = pd.Series([10.0] * 100)
    bins = Binning.from_reference(ref, categorical=False)
    measured, _ = population_stability(ref, current, bins)
    assert measured["distancia_ks"] == 0.5
    assert "pvalue" not in measured
    assert max(bins.cuts) == 10


def test_faltantes_no_se_confunden_con_drift_de_valores_observados():
    ref = pd.Series([5.0] * 100, dtype="float32")
    current = ref.copy()
    current.iloc[:50] = np.nan
    bins = {"lag_24": Binning.from_reference(ref, categorical=False)}
    rows, _ = compare_distributions(
        pd.DataFrame({"lag_24": ref}), pd.DataFrame({"lag_24": current}), bins, AlertPolicy()
    )
    assert rows[0]["psi"] == 0 and rows[0]["faltantes_actual"] == 0.5
    rows[0]["comparacion"] = "historica"
    same = {"mae": 1.0, "rmse": 2.0, "wape": 0.5, "r2": 0.2}
    alerts = make_alerts(rows, same, same, AlertPolicy())
    assert len(alerts) == 1 and alerts[0]["tipo"] == "calidad"


def test_todos_los_rezagos_faltantes_no_inventan_psi():
    ref = pd.Series([0.0, 1.0, 2.0])
    bins = Binning.from_reference(ref, categorical=False)
    measured, details = population_stability(ref, pd.Series([np.nan] * 3), bins)
    assert measured["psi"] is None and measured["distancia_ks"] is None and not details


def test_metricas_contra_ejemplo_calculable_y_deltas_completos():
    frame = pd.DataFrame({"demanda": [0.0, 2.0, 4.0], "hora": [7, 8, 9]})
    score = score_predictions(frame, np.array([1.0, 2.0, 2.0]))
    assert score == pytest.approx({"mae": 1.0, "rmse": np.sqrt(5 / 3), "wape": 0.5, "r2": 0.375})
    changed = {**score, "mae": 1.2, "r2": -0.125}
    delta = metric_deltas(changed, score)
    assert delta["delta_mae"] == pytest.approx(0.2)
    assert delta["diferencia_absoluta_r2"] == 0.5
    assert delta["delta_relativa_mae"] == pytest.approx(0.2)


def test_metricas_con_denominador_y_varianza_nulos_son_indefinidas():
    frame = pd.DataFrame({"demanda": [0.0, 0.0], "hora": [7, 8]})
    score = score_predictions(frame, np.array([0.0, 1.0]))
    assert score["wape"] is None and score["r2"] is None
    assert metric_deltas(score, score)["delta_relativa_wape"] is None


@pytest.mark.parametrize(
    "prediction", [np.array([-1.0, 1.0]), np.array([np.nan, 1.0]), np.array([1.0])]
)
def test_no_recorta_predicciones_ni_omite_errores(prediction):
    with pytest.raises(ValueError):
        score_predictions(pd.DataFrame({"demanda": [0.0, 1.0], "hora": [7, 8]}), prediction)


def test_alertas_de_desempeno_exigen_cambio_relativo_y_absoluto():
    control = {"mae": 1.0, "rmse": 2.0, "wape": 0.5, "r2": -0.5}
    small = {"mae": 1.09, "rmse": 2.19, "wape": 0.54, "r2": -0.59}
    assert not performance_alerts(small, control, AlertPolicy())
    degraded = {"mae": 1.2, "rmse": 2.4, "wape": 0.6, "r2": -0.7}
    assert {a["variable"] for a in performance_alerts(degraded, control, AlertPolicy())} == {
        "mae",
        "rmse",
        "wape",
        "r2",
    }
    # Un empeoramiento relativo grande pero minúsculo en unidades no dispara.
    tiny = {"mae": 0.001, "rmse": 0.002, "wape": 0.005, "r2": 0.9}
    assert not performance_alerts({**tiny, "mae": 0.002}, tiny, AlertPolicy())


def test_drift_detectado_no_implica_alerta_de_perdida():
    detector = {
        "rol": "entrada",
        "variable": "hora",
        "psi": 0.3,
        "nivel_psi": "alerta",
        "comparacion": "control",
    }
    score = {"mae": 1.0, "rmse": 2.0, "wape": 0.5, "r2": 0.2}
    alerts = make_alerts([detector], score, score, AlertPolicy())
    assert len(alerts) == 1 and alerts[0]["tipo"] == "distribucion"


def test_umbral_de_faltantes_incluye_limite_y_politica_del_plan():
    plan = json.loads(DEFAULT_PLAN.read_text(encoding="utf-8"))
    policy = AlertPolicy(**plan["umbrales"])
    assert policy == AlertPolicy()
    detector = {
        "rol": "entrada",
        "variable": "lag_168",
        "psi": 0,
        "nivel_psi": "sin_alerta",
        "comparacion": "historica",
        "faltantes_actual": 0.05,
        "faltantes_referencia": 0.0,
    }
    score = {"mae": 1.0, "rmse": 2.0, "wape": 0.5, "r2": 0.2}
    assert make_alerts([detector], score, score, policy)[0]["nivel"] == "alerta"


def test_integracion_artefacto_temporal_septiembre_fijo_sin_fit(monkeypatch):
    root = config.REPORTS_DIR / "fase3"
    execution = json.loads((root / "ejecucion.json").read_text(encoding="utf-8"))
    path = config.ROOT / execution["cortes"][0]["modelo_local"]
    model = joblib.load(path)
    saved = pd.read_parquet(config.PROCESSED_DIR / "fase3/predicciones_backtest.parquet")
    base = saved.loc[saved["corte"].eq("2026-09-07 00:00:00")].reset_index(drop=True)
    columns = features.get_feature_columns(base, horizon="24h")

    def no_fit(*args, **kwargs):
        pytest.fail("La simulación intentó ajustar un componente del modelo")

    for step in (model, *model.named_steps.values()):
        monkeypatch.setattr(type(step), "fit", no_fit)
    before = joblib.hash(model)
    with threadpool_limits(limits=4):
        prediction = model.predict(base[columns])
        missing = make_scenario(base, "faltantes", 1.0)
        missing_prediction = model.predict(missing[columns])
    np.testing.assert_allclose(prediction, base["hist_gbm_poisson"], rtol=1e-12, atol=1e-12)
    assert np.isfinite(missing_prediction).all() and (missing_prediction >= 0).all()
    assert missing["lag_24"].dtype == np.dtype("float32")
    assert joblib.hash(model) == before
