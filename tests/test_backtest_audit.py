"""Controles independientes de la auditoría; no entrenan modelos."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mibici_demand.modeling.audit_backtest import (
    replace_numeric_preserving_dtype,
    snapshot_originals,
    squared_error_concentration,
    verify_composition,
    verify_labels_and_lags,
)
from mibici_demand.temporal import BacktestSpec, build_fold


def test_perturbacion_controlada_no_cambia_el_tipo_ni_las_otras_variables():
    frame = pd.DataFrame(
        {
            "lag_168": pd.Series([50], dtype="float32"),
            "lag_24": pd.Series([0], dtype="float32"),
            "estacion": pd.Series([46], dtype="int16"),
        }
    )
    original = frame.copy()
    same = replace_numeric_preserving_dtype(frame, "lag_168", 50.0)
    pd.testing.assert_frame_equal(same, original)
    changed = replace_numeric_preserving_dtype(frame, "lag_168", 1.0)
    assert changed["lag_168"].dtype == np.dtype("float32")
    pd.testing.assert_frame_equal(
        changed.drop(columns="lag_168"), original.drop(columns="lag_168")
    )
    pd.testing.assert_frame_equal(frame, original)


def test_uno_por_ciento_se_redondea_hacia_arriba_y_conserva_todo_el_denominador():
    prediction = np.r_[10.0, 9.0, np.ones(99)]
    result = squared_error_concentration(np.zeros(101), prediction)
    assert result["n_top"] == 2
    assert result["sse_total"] == 280
    assert result["sse_top"] == 181
    assert result["proporcion_sse_top"] == 181 / 280


def test_error_nulo_no_tiene_concentracion_definida():
    result = squared_error_concentration([1, 2], [1, 2])
    assert result["sse_total"] == 0
    assert result["proporcion_sse_top"] is None


@pytest.mark.parametrize(
    "actual,predicted,fraction",
    [
        ([], [], 0.01),
        ([1], [1, 2], 0.01),
        ([1], [np.inf], 0.01),
        ([np.nan], [1], 0.01),
        ([1], [2], 0),
        ([1], [2], 1.1),
    ],
)
def test_concentracion_rechaza_entradas_que_invalidarian_la_estadistica(
    actual, predicted, fraction
):
    with pytest.raises(ValueError):
        squared_error_concentration(actual, predicted, fraction)


@pytest.fixture(scope="module")
def audit_sample():
    times = pd.date_range("2026-08-01", "2026-09-28", freq="h", inclusive="left")
    rows = [
        (t + pd.Timedelta(minutes=5), t + pd.Timedelta(minutes=15), s, 3 - s)
        for t in times
        for s in [1, 2]
    ]
    trips = pd.DataFrame(
        rows, columns=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
    )
    spec = BacktestSpec(train_days=7, evaluation_days=1)
    _, evaluated, audit = build_fold(trips, spec.timestamps[0], spec)
    return trips, evaluated, audit, spec


def test_reconstruccion_independiente_confirma_etiquetas_y_rezagos(audit_sample):
    trips, evaluated, _, _ = audit_sample
    result = verify_labels_and_lags(trips, evaluated)
    assert result["etiquetas_verificadas"] == len(evaluated)
    assert result["rezagos_verificados"] == 2 * len(evaluated)
    assert result["discrepancias"] == 0


@pytest.mark.parametrize("column", ["demanda", "lag_24", "lag_168"])
def test_reconstruccion_detecta_valores_adulterados(audit_sample, column):
    trips, evaluated, _, _ = audit_sample
    poisoned = evaluated.copy()
    poisoned.loc[0, column] += 1
    with pytest.raises(ValueError):
        verify_labels_and_lags(trips, poisoned)


def test_reconstruccion_detecta_fuente_futura_aunque_el_valor_del_lag_parezca_valido(audit_sample):
    trips, evaluated, _, _ = audit_sample
    poisoned = evaluated.copy()
    poisoned.loc[0, "fuente_lag_24"] = poisoned.loc[0, "origen_pronostico"]
    with pytest.raises(ValueError, match="Fuente temporal"):
        verify_labels_and_lags(trips, poisoned)


def test_datos_posteriores_no_definen_la_cohorte(audit_sample):
    trips, evaluated, audit, spec = audit_sample
    future = trips.loc[trips["Inicio_del_viaje"].ge(spec.timestamps[0])].copy()
    future["Origen_Id"] = 999
    future["Destino_Id"] = 999
    poisoned = pd.concat([trips, future], ignore_index=True)
    verify_composition(poisoned, evaluated, audit, spec)


def test_rejilla_no_admite_excluir_una_fila_segun_su_resultado(audit_sample):
    trips, evaluated, audit, spec = audit_sample
    with pytest.raises(ValueError, match="composición"):
        verify_composition(trips, evaluated.iloc[1:], audit, spec)


def test_instantanea_no_sobrescribe_resultados_anteriores(tmp_path):
    root = tmp_path / "reports"
    root.mkdir()
    (root / "ejecucion.json").write_text(json.dumps({"id": "original"}), encoding="utf-8")
    source = root / "metricas_cortes.csv"
    source.write_text("mae\n1.0129\n", encoding="utf-8")
    snapshot_originals(root, root / "auditoria")
    snapshot_originals(root, root / "auditoria")
    source.write_text("mae\n0.001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No se sobrescribe"):
        snapshot_originals(root, root / "auditoria")
    assert (root / "auditoria/originales/original/metricas_cortes.csv").read_text(
        encoding="utf-8"
    ) == "mae\n1.0129\n"
