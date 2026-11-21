"""Integración contra un contenedor real; requiere motor Linux e imagen ya construida."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import shutil

import joblib
import numpy as np
import pytest
import requests
from scripts.verify_docker import (
    IMAGE,
    assert_isolation,
    container_service,
    direct_frame,
    docker,
    native_probe,
    values,
)
from threadpoolctl import threadpool_limits

from mibici_demand import config


@pytest.fixture(scope="module")
def service():
    if not shutil.which("docker"):
        pytest.skip("Docker CLI no instalado")
    try:
        assert docker("info", "--format", "{{.OSType}}") == "linux"
        docker("image", "inspect", IMAGE)
    except (RuntimeError, AssertionError) as error:
        pytest.skip(f"Construir la imagen y activar Docker Linux: {error}")
    with container_service() as running:
        yield running


@pytest.fixture(scope="module")
def payload():
    return json.loads((config.ROOT / "examples/predict_request.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def probe(service):
    return native_probe(service)


def test_docker_health_y_hash_del_mismo_modelo(service):
    response = requests.get(service.url + "/health", timeout=5)
    assert response.status_code == 200 and response.json()["model_available"]
    assert (
        response.json()["model"]["artifact_sha256"]
        == sha256((config.MODELS_DIR / "best_pipeline.joblib").read_bytes()).hexdigest()
    )
    assert service.info["State"]["Health"]["Status"] == "healthy"


def test_docker_predict_equivale_a_python_windows_y_referencia_previa(service, payload):
    response = requests.post(service.url + "/predict", json=payload, timeout=10)
    assert response.status_code == 200, response.text
    model = joblib.load(config.MODELS_DIR / "best_pipeline.joblib")
    with threadpool_limits(limits=4):
        reference = model.predict(direct_frame(payload))
    np.testing.assert_allclose(values(response.json()), reference, rtol=1e-12, atol=1e-12)
    previous = json.loads(
        (config.ROOT / "examples/predict_response.json").read_text(encoding="utf-8")
    )
    np.testing.assert_allclose(values(response.json()), values(previous), rtol=1e-12, atol=1e-12)
    assert response.json()["predicciones"][-1]["diagnostico"]["unknown_station"] is True


def test_docker_swagger_y_openapi(service):
    response = requests.get(service.url + "/docs", timeout=5)
    assert response.status_code == 200 and "SwaggerUIBundle" in response.text
    assert (
        "post"
        in requests.get(service.url + "/openapi.json", timeout=5).json()["paths"]["/predict"]
    )


@pytest.mark.parametrize(
    "field,value", [("hora", 8), ("lag_24", -1), ("estacion", True), ("lag_168", "50")]
)
def test_docker_rechaza_datos_invalidos_sin_debilitar_esquema(service, payload, field, value):
    body = deepcopy(payload)
    body["observaciones"][0][field] = value
    assert requests.post(service.url + "/predict", json=body, timeout=5).status_code == 422


def test_docker_sin_volumenes_venv_datos_credenciales_o_mlflow(service, probe):
    assert_isolation(service, probe)


def test_docker_contrato_numerico_y_versiones_fijadas(probe):
    assert probe["python"] == "3.13.0" and probe["versions"]["scikit-learn"] == "1.9.1"
    assert (
        probe["input_dtypes"]["lag_24"] == "float32"
        and probe["input_dtypes"]["latitud"] == "float64"
    )
    assert probe["versions"]["pyarrow"] == "26.0.1"


def test_docker_http_e_inferencia_dentro_del_contenedor(probe):
    assert probe["health_status"] == probe["docs_status"] == probe["predict_status"] == 200
    assert probe["invalid_status"] == 422 and probe["swagger_ui"]
    np.testing.assert_allclose(
        values(probe["http_response"]), probe["direct_predictions"], rtol=1e-12, atol=1e-12
    )
