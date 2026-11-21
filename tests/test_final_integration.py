"""Contratos adicionales del candidato, recuperación y CV por timestamps."""

from copy import deepcopy
from hashlib import sha256
import json
import subprocess

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from scripts.artifacts import restore
from scripts.verify_docker import container_service, docker, values
from scripts.verify_final import IMAGE

from mibici_demand import api, config
from mibici_demand.modeling.train import timestamp_cv_splits
from mibici_demand.release import default_model_path, read_release


@pytest.fixture(scope="module")
def payload():
    return json.loads(
        (config.ROOT / "examples/predict_request_final.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def final_client():
    with TestClient(api.create_app(default_model_path())) as connection:
        yield connection


def test_release_es_copia_exacta_del_ultimo_corte_sin_cambiar_historico():
    release = read_release()
    assert sha256(default_model_path().read_bytes()).hexdigest() == release["sha256"]
    assert (
        default_model_path().read_bytes() == (config.ROOT / release["source_model"]).read_bytes()
    )
    assert not release["produccion_aprobada"]
    assert (
        sha256((config.ROOT / "models/best_pipeline.joblib").read_bytes()).hexdigest()
        == release["historico_sha256"]
    )


def test_api_final_declara_hash_uso_y_tipos_del_snapshot(final_client, payload):
    response = final_client.get("/health")
    assert response.status_code == 200
    release = read_release()
    assert response.json()["model"]["release_id"] == release["release_id"]
    assert response.json()["model"]["uso"] == release["uso"]
    frame = api.feature_frame(
        api.PredictRequest.model_validate(payload), final_client.app.state.loaded
    )
    assert frame.dtypes.astype(str).to_dict() == release["input_dtypes"]
    assert final_client.post("/predict", json=payload).status_code == 200


def test_api_final_no_predice_origen_anterior_a_entrenamiento(final_client, payload):
    body = deepcopy(payload)
    body["origen_pronostico"] = "2026-09-12T00:00:00-06:00"
    for row in body["observaciones"]:
        row["hora_inicio"] = row["hora_inicio"].replace("2026-09-26", "2026-09-12")
    assert final_client.post("/predict", json=body).status_code == 422


@pytest.mark.parametrize("field,value", [("lag_24", None), ("lag_168", -1), ("hora", 8)])
def test_api_final_conserva_rechazo_de_faltantes_y_contradicciones(
    final_client, payload, field, value
):
    body = deepcopy(payload)
    body["observaciones"][0][field] = value
    assert final_client.post("/predict", json=body).status_code == 422


def test_hash_configurado_incorrecto_impide_cargar_modelo(monkeypatch):
    monkeypatch.setenv("MODEL_SHA256", "0" * 64)
    with TestClient(api.create_app(default_model_path())) as connection:
        assert connection.get("/health").status_code == 503


def test_cv_purga24h_y_no_separa_estaciones_mismo_timestamp():
    times = pd.Series(pd.date_range("2026-08-01", periods=240, freq="h").repeat(7))
    for train, valid in timestamp_cv_splits(times, n_splits=3):
        assert set(times.iloc[train]).isdisjoint(set(times.iloc[valid]))
        assert times.iloc[train].max() < times.iloc[valid].min() - pd.Timedelta(hours=24)
        assert times.iloc[train].value_counts().eq(7).all()
        assert times.iloc[valid].value_counts().eq(7).all()


def test_cv_rechaza_historia_sin_etiquetas_maduras():
    with pytest.raises(ValueError):
        timestamp_cv_splits(
            pd.Series(pd.date_range("2026-09-01", periods=12, freq="h")), n_splits=3
        )


def test_monitoreo_final_usa_mismo_sha_y_periodo_posterior():
    result = json.loads(
        (config.ROOT / "reports/fase3/integracion/drift_final/ejecucion.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["procedencia"]["modelo_sha256"] == read_release()["sha256"]
    assert result["procedencia"]["modelo"] == read_release()["model_path"]
    assert result["procedencia"]["monitoreo_inicio"] == read_release()["corte"]
    assert not result["procedencia"]["octubre_consumido"]
    assert len(result["escenarios"]) == 10


@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_evidencia_reentrenamiento_limpio_tres_cortes(platform):
    result = json.loads(
        (config.ROOT / f"reports/fase3/integracion/repro_{platform}/resultado.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["estado"] == "reproducido"
    assert len(result["cortes"]) == 3 and all(
        r["metricas_en_tolerancia"] for r in result["cortes"]
    )
    assert not result["promocion_ejecutada"]


def test_artefactos_recuperables_sin_sobrescribir(tmp_path, monkeypatch):
    root = config.ROOT
    expected = default_model_path().read_bytes()
    (tmp_path / "references").mkdir()
    (tmp_path / "references/artifact_bundle.json").write_bytes(
        (root / "references/artifact_bundle.json").read_bytes()
    )
    monkeypatch.setattr(config, "ROOT", tmp_path)
    count = restore(root / "artifacts/mibici-fase3-v1.zip", tmp_path)
    assert count > 6
    assert (tmp_path / "models/final/pipeline.joblib").read_bytes() == expected
    (tmp_path / "models/final/pipeline.joblib").write_bytes(b"distinto")
    with pytest.raises(ValueError, match="No sobrescribir"):
        restore(root / "artifacts/mibici-fase3-v1.zip", tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        ".venv/example.txt",
        "data/snapshots/a.parquet",
        "models/final/pipeline.joblib",
        "mlflow_integracion.db",
        ".env.example.secret",
        "credentials_private.json",
        "artifacts/test.zip",
    ],
)
def test_publicacion_excluye_binarios_datos_y_credenciales(path):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=config.ROOT, check=False)
    assert result.returncode == 0


@pytest.fixture(scope="module")
def final_docker():
    # Este bloque no omite silenciosamente: construir la imagen es un requisito.
    docker("image", "inspect", IMAGE)
    with container_service(IMAGE) as service:
        yield service


def test_docker_final_equivale_a_windows_sin_volumenes(final_docker, final_client, payload):
    import requests

    response = requests.post(final_docker.url + "/predict", json=payload, timeout=10)
    assert response.status_code == 200
    assert response.json()["model"]["artifact_sha256"] == read_release()["sha256"]
    np.testing.assert_allclose(
        values(response.json()),
        values(final_client.post("/predict", json=payload).json()),
        rtol=1e-12,
        atol=1e-12,
    )
    assert final_docker.info["Mounts"] == [] and final_docker.info["HostConfig"]["ReadonlyRootfs"]
