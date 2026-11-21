"""Fase 3: ataques de futuro, disponibilidad real y particiones por timestamp."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from mibici_demand import config, dataset, features
from mibici_demand.modeling.backtest import evaluation_metrics, fit_fold
from mibici_demand.temporal import BacktestSpec, build_fold, causal_daily_frame, prepare_data


@pytest.fixture(scope="module")
def trips():
    rows = []
    for timestamp in pd.date_range("2026-08-01", "2026-09-28", freq="h", inclusive="left"):
        for station in (1, 2):
            for i in range(1 + (timestamp.day + timestamp.hour + station) % 3):
                start = timestamp + pd.Timedelta(minutes=2 * i)
                rows.append((start, start + pd.Timedelta(minutes=10), station, 3 - station))
    return pd.DataFrame(
        rows, columns=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
    )


def test_solo_dos_meses_con_historia_suficiente():
    spec = BacktestSpec()
    assert spec.periods == ["2026-08", "2026-09"]
    assert spec.data_start == pd.Timestamp("2026-08-02")
    assert spec.data_end == pd.Timestamp("2026-09-28")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cutoffs": ("2026-09-07", "2026-09-14")},
        {"cutoffs": ("2026-09-14", "2026-09-07", "2026-09-21")},
        {"cutoffs": ("2026-09-07", "2026-09-07", "2026-09-21")},
        {"cutoffs": ("2026-09-07", "2026-09-10", "2026-09-21")},
        {"cutoffs": ("2026-09-07 12:00", "2026-09-15", "2026-09-23")},
        {"cutoffs": ("NaT", "2026-09-14", "2026-09-21")},
        {"label_delay_hours": 0},
        {"label_delay_hours": 25},
        {"train_days": 3},
    ],
)
def test_rechaza_configuraciones_que_rompen_el_protocolo(kwargs):
    with pytest.raises(ValueError):
        BacktestSpec(**kwargs)


def test_todos_los_registros_de_un_timestamp_permanecen_juntos(trips):
    spec = BacktestSpec()
    cutoff = spec.timestamps[0]
    train, evaluation, audit = build_fold(trips, cutoff, spec)
    assert train["hora_inicio"].max() == pd.Timestamp("2026-09-05 23:00")
    assert evaluation["hora_inicio"].min() == cutoff
    assert evaluation["hora_inicio"].max() < pd.Timestamp("2026-09-14")
    assert set(train["hora_inicio"]).isdisjoint(evaluation["hora_inicio"])
    assert train.groupby("hora_inicio")["estacion"].nunique().eq(2).all()
    assert evaluation.groupby("hora_inicio")["estacion"].nunique().eq(2).all()
    assert pd.Timestamp(audit["max_fin_viaje_etiquetas_train"]) < cutoff
    assert train["origen_pronostico"].max() < cutoff


def test_mutar_futuro_no_cambia_train_cohorte_ni_primera_emision(trips):
    spec = BacktestSpec()
    cutoff = spec.timestamps[0]
    train, evaluation, audit = build_fold(trips, cutoff, spec)
    future = trips[trips["Inicio_del_viaje"] >= cutoff].copy()
    future["Origen_Id"] = 999
    poisoned = pd.concat([trips, *([future] * 5)], ignore_index=True)
    changed_train, changed_eval, changed_audit = build_fold(poisoned, cutoff, spec)
    assert_frame_equal(train, changed_train)
    assert audit["ids_estaciones"] == changed_audit["ids_estaciones"]
    assert audit["horas_servicio"] == changed_audit["horas_servicio"]
    assert 999 not in changed_eval["estacion"].unique()
    columns = features.get_feature_columns(evaluation, horizon="24h")
    assert_frame_equal(
        evaluation.loc[evaluation["origen_pronostico"] == cutoff, columns],
        changed_eval.loc[changed_eval["origen_pronostico"] == cutoff, columns],
    )


def test_un_viaje_abierto_no_entra_en_el_lag_aunque_su_inicio_sea_pasado(trips):
    origin = pd.Timestamp("2026-09-07")
    first_seen = pd.Series({1: pd.Timestamp("2026-08-01")})
    args = ([1], origin, origin + pd.Timedelta(days=1), [23], first_seen)
    base = causal_daily_frame(trips, *args)
    late = pd.DataFrame(
        [
            {
                "Inicio_del_viaje": origin - pd.Timedelta(minutes=30),
                "Fin_del_viaje": origin + pd.Timedelta(minutes=10),
                "Origen_Id": 1,
                "Destino_Id": 2,
            }
        ]
    )
    changed = causal_daily_frame(pd.concat([trips, late]), *args)
    assert changed.loc[0, "lag_24"] == base.loc[0, "lag_24"]
    late["Fin_del_viaje"] = origin - pd.Timedelta(minutes=1)
    completed = causal_daily_frame(pd.concat([trips, late]), *args)
    assert completed.loc[0, "lag_24"] == base.loc[0, "lag_24"] + 1
    late["Fin_del_viaje"] = origin
    at_origin = causal_daily_frame(pd.concat([trips, late]), *args)
    assert at_origin.loc[0, "lag_24"] == base.loc[0, "lag_24"]


def test_observaciones_de_evaluacion_solo_entran_al_pronostico_siguiente(trips):
    origin = pd.Timestamp("2026-09-07")
    first_seen = pd.Series({1: pd.Timestamp("2026-08-01")})
    args = ([1], origin, origin + pd.Timedelta(days=2), [8], first_seen)
    base = causal_daily_frame(trips, *args)
    extra = pd.DataFrame(
        [
            {
                "Inicio_del_viaje": origin + pd.Timedelta(hours=8),
                "Fin_del_viaje": origin + pd.Timedelta(hours=8, minutes=15),
                "Origen_Id": 1,
                "Destino_Id": 2,
            }
        ]
    )
    changed = causal_daily_frame(pd.concat([trips, extra]), *args)
    assert changed.loc[0, "lag_24"] == base.loc[0, "lag_24"]
    assert changed.loc[1, "lag_24"] == base.loc[1, "lag_24"] + 1
    for lag in (24, 168):
        assert (
            changed[f"fuente_lag_{lag}"] + pd.Timedelta(hours=1) <= changed["origen_pronostico"]
        ).all()


def test_tres_pipelines_independientes_y_perfil_solo_del_train(trips):
    spec = BacktestSpec(train_days=7, evaluation_days=1)
    pipelines, evals = [], []
    for cutoff in spec.timestamps:
        train, evaluation, _ = build_fold(trips, cutoff, spec)
        pipeline, columns = fit_fold(train, threads=1, params={"max_iter": 2})
        pipelines.append(pipeline)
        evals.append(evaluation)
        actual = pipeline.named_steps["perfil_estacion"].station_profiles_.set_index("estacion")[
            "est_demanda_media"
        ]
        expected = train.groupby("estacion")["demanda"].mean()
        assert np.allclose(actual.loc[expected.index], expected)
        assert set(columns) == {
            "estacion",
            "hora",
            "dia_semana",
            "mes",
            "es_fin_semana",
            "lag_24",
            "lag_168",
        }
        prediction = pipeline.predict(evaluation[columns])
        assert np.isfinite(prediction).all()
        # Tanto el imputador como el escalador se ajustaron solo a train enriquecido.
        enriched = pipeline.named_steps["perfil_estacion"].transform(train[columns])
        numeric = pipeline.named_steps["preprocesador"].transformers_[0]
        imputer = numeric[1].named_steps["imputador"]
        scaler = numeric[1].named_steps["escalador"]
        assert np.allclose(scaler.mean_, imputer.transform(enriched[numeric[2]]).mean(axis=0))
    assert len({id(p) for p in pipelines}) == 3
    assert len({id(p.named_steps["preprocesador"]) for p in pipelines}) == 3
    assert set(evals[0]["hora_inicio"]).isdisjoint(evals[1]["hora_inicio"])


def test_wape_y_r2_indefinidos_no_se_reportan_como_exito():
    y = pd.Series([0, 0])
    metrics = evaluation_metrics(y, np.array([1, 2]), pd.Series([8, 9]))
    assert metrics["mae"] == 1.5
    assert metrics["wape"] is None
    assert metrics["wape_horas_pico"] is None
    assert metrics["r2"] is None


def test_preparacion_no_asume_ceros_si_faltan_archivos(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "archivo": f"datos_abiertos_2026_{month}.csv",
                "url": "https://example.invalid/data",
                "bytes": 1,
                "sha256": "x",
            }
            for month in ("08", "09")
        ]
    ).to_csv(manifest, index=False)
    with pytest.raises(FileNotFoundError, match="Faltan datos mensuales"):
        prepare_data(
            BacktestSpec(),
            tmp_path / "raw",
            tmp_path / "processed",
            tmp_path / "reports",
            download_missing=False,
            expected_manifest=manifest,
        )


def test_preparacion_rechaza_fuentes_modificadas(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for month in ("08", "09"):
        name = f"datos_abiertos_2026_{month}.csv"
        (raw / name).write_text("no son los datos esperados", encoding="utf-8")
        rows.append(
            {"archivo": name, "url": "https://example.invalid/data", "bytes": 1, "sha256": "x"}
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="Integridad incorrecta"):
        prepare_data(
            BacktestSpec(),
            raw,
            tmp_path / "processed",
            tmp_path / "reports",
            download_missing=False,
            expected_manifest=manifest,
        )


def test_descarga_sin_catalogo_respeta_la_ventana_y_no_toca_el_manifiesto_original(
    tmp_path, monkeypatch
):
    import requests

    requested = []
    csv_url = "https://mibici.net/datos_abiertos_2026_08.csv"
    catalog_url = "https://mibici.net/nomenclatura_2026_10.csv"

    class Response:
        text = f'<a href="{csv_url}">Ago</a><a href="{catalog_url}">Cat</a>'

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, size):
            yield b"datos"

    def get(url, **kwargs):
        requested.append(url)
        return Response()

    monkeypatch.setattr(requests, "get", get)
    original_hash = dataset.sha256_file(config.RAW_DIR / "manifest.csv")
    manifest = dataset.download("2026-08", "2026-08", tmp_path, include_catalog=False)
    assert requested == [config.OPEN_DATA_URL, csv_url]
    assert manifest["archivo"].tolist() == ["datos_abiertos_2026_08.csv"]
    assert dataset.sha256_file(config.RAW_DIR / "manifest.csv") == original_hash
