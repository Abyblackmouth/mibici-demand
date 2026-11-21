"""Integración HTTP y equivalencia con artefactos reales, sin reentrenamiento."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from mibici_demand import api, config


@pytest.fixture(scope="module")
def historical_path():
    path = config.MODELS_DIR / "best_pipeline.joblib"
    if not path.is_file():
        pytest.skip("Proporcionar models/best_pipeline.joblib; no se descarga ni entrena.")
    return path


@pytest.fixture(scope="module")
def client(historical_path):
    with TestClient(api.create_app(historical_path)) as connection:
        assert connection.get("/health").status_code == 200
        yield connection


def direct_frame(body: dict) -> pd.DataFrame:
    """Referencia independiente del constructor de la API, con tipos offline."""
    records = []
    for row in body["observaciones"]:
        ts = pd.Timestamp(row["hora_inicio"])
        ts = (
            ts.tz_localize("America/Mexico_City")
            if ts.tz is None
            else ts.tz_convert("America/Mexico_City")
        )
        records.append(
            {
                "estacion": row["estacion"],
                "hora": ts.hour,
                "dia_semana": ts.day_of_week,
                "mes": ts.month,
                "es_fin_semana": int(ts.day_of_week >= 5),
                "lag_24": row["lag_24"],
                "lag_168": row["lag_168"],
                **{k: row[k] for k in ["latitud", "longitud", "capacidad"]},
            }
        )
    return pd.DataFrame(records).astype(
        {
            "estacion": "int16",
            "hora": "int8",
            "dia_semana": "int8",
            "mes": "int8",
            "es_fin_semana": "int8",
            "lag_24": "float32",
            "lag_168": "float32",
            "latitud": "float64",
            "longitud": "float64",
            "capacidad": "float64",
        }
    )


def test_health_incluye_hash_contrato_y_uso_de_referencia(client, historical_path):
    body = client.get("/health").json()
    assert body["status"] == "ready" and body["model_available"]
    digest = sha256(historical_path.read_bytes()).hexdigest()
    assert body["model"]["artifact_sha256"] == digest
    assert body["model"]["model_version"] == "sha256:" + digest
    assert body["model"]["uso"] == "referencia_integracion_no_validado_prospectivamente"
    assert body["model"]["input_dtypes"]["lag_24"] == "float32"
    assert body["model"]["input_dtypes"]["latitud"] == "float64"


def test_swagger_y_openapi_disponibles(client):
    assert "SwaggerUIBundle" in client.get("/docs").text
    schema = client.get("/openapi.json").json()
    assert "post" in schema["paths"]["/predict"]
    assert "get" in schema["paths"]["/health"]


def test_predict_preserva_orden_identificadores_y_no_negatividad(client):
    response = client.post("/predict", json=api.EXAMPLE_REQUEST)
    assert response.status_code == 200, response.text
    rows = response.json()["predicciones"]
    assert [r["estacion"] for r in rows] == [2, 46, 999]
    assert all(r["demanda_predicha"] >= 0 and np.isfinite(r["demanda_predicha"]) for r in rows)
    assert rows[1]["hora_inicio"] == "2026-09-12T21:00:00-06:00"


def test_api_equivale_a_inferencia_directa_y_recarga_no_cambia_salida(client, historical_path):
    frame = direct_frame(api.EXAMPLE_REQUEST)
    first, second = joblib.load(historical_path), joblib.load(historical_path)
    digest = sha256(historical_path.read_bytes()).hexdigest()
    with threadpool_limits(limits=4):
        before, after = first.predict(frame), second.predict(frame)
    values = np.array(
        [
            r["demanda_predicha"]
            for r in client.post("/predict", json=api.EXAMPLE_REQUEST).json()["predicciones"]
        ]
    )
    np.testing.assert_array_equal(before, after)
    np.testing.assert_allclose(values, before, rtol=1e-12, atol=1e-12)
    assert not np.array_equal(values, values.astype("float32").astype("float64"))
    assert sha256(historical_path.read_bytes()).hexdigest() == digest


def test_entrada_repetida_produce_respuesta_identica(client):
    assert (
        client.post("/predict", json=api.EXAMPLE_REQUEST).json()
        == client.post("/predict", json=api.EXAMPLE_REQUEST).json()
    )


def test_estacion_desconocida_usa_el_comportamiento_del_pipeline(client, historical_path):
    model = joblib.load(historical_path)
    assert 999 not in model.named_steps["perfil_estacion"].station_profiles_["estacion"].to_list()
    row = client.post("/predict", json=api.EXAMPLE_REQUEST).json()["predicciones"][-1]
    assert row["estacion"] == 999 and row["diagnostico"]["unknown_station"]
    assert row["demanda_predicha"] >= 0


def test_tipos_del_frame_coinciden_con_contrato_offline(client):
    frame = api.feature_frame(
        api.PredictRequest.model_validate(api.EXAMPLE_REQUEST), client.app.state.loaded
    )
    assert frame.dtypes.astype(str).to_dict() == api.INPUT_DTYPES
    pd.testing.assert_frame_equal(frame, direct_frame(api.EXAMPLE_REQUEST)[frame.columns])


@pytest.mark.parametrize(
    "field,value",
    [
        ("estacion", True),
        ("estacion", "46"),
        ("estacion", 0),
        ("estacion", 32768),
        ("lag_24", -1),
        ("lag_168", "50"),
        ("lag_24", True),
        ("lag_24", 1e40),
        ("latitud", 91),
        ("longitud", -181),
        ("capacidad", 0),
        ("capacidad", "15"),
        ("hora_inicio", "no-fecha"),
        ("hora_inicio", 1790000000),
        ("hora_inicio", "2026-09-12T08:30:00-06:00"),
        ("hora_inicio", "2026-09-13T00:00:00-06:00"),
        ("hora_inicio", "2026-09-11T23:00:00-06:00"),
        ("hora", 8),
        ("mes", 9),
        ("dia_semana", 5),
        ("demanda", 4),
        ("lag_1", 0),
    ],
)
def test_rechaza_datos_invalidos_o_contradictorios(client, field, value):
    body = deepcopy(api.EXAMPLE_REQUEST)
    body["observaciones"][0][field] = value
    response = client.post("/predict", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["msg"]


@pytest.mark.parametrize("field", ["lag_24", "latitud", "longitud", "capacidad"])
def test_campos_requeridos_por_esquema_o_artefacto(client, field):
    body = deepcopy(api.EXAMPLE_REQUEST)
    del body["observaciones"][0][field]
    assert client.post("/predict", json=body).status_code == 422


def test_nan_e_infinito_se_rechazan_con_error_serializable(client):
    for number in ["NaN", "Infinity", "-Infinity"]:
        body = json.dumps(api.EXAMPLE_REQUEST).replace('"lag_24": 1.0', f'"lag_24": {number}', 1)
        response = client.post(
            "/predict", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"]


def test_lote_vacio_origen_no_diario_y_duplicados_se_rechazan(client):
    for variant in ["empty", "origin", "duplicate"]:
        body = deepcopy(api.EXAMPLE_REQUEST)
        if variant == "empty":
            body["observaciones"] = []
        elif variant == "origin":
            body["origen_pronostico"] = "2026-09-12T01:00:00-06:00"
        else:
            body["observaciones"].append(body["observaciones"][0])
        assert client.post("/predict", json=body).status_code == 422


def test_utc_y_hora_local_derivan_el_mismo_calendario(client):
    body = deepcopy(api.EXAMPLE_REQUEST)
    body["origen_pronostico"] = "2026-09-12T06:00:00Z"
    for row in body["observaciones"]:
        row["hora_inicio"] = pd.Timestamp(row["hora_inicio"]).tz_convert("UTC").isoformat()
    assert (
        client.post("/predict", json=body).json()
        == client.post("/predict", json=api.EXAMPLE_REQUEST).json()
    )


def test_modelo_se_carga_una_sola_vez_en_inicio(historical_path, monkeypatch):
    original = api.joblib.load
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(api.joblib, "load", counted)
    with TestClient(api.create_app(historical_path)) as connection:
        for _ in range(2):
            assert connection.get("/health").status_code == 200
            assert connection.post("/predict", json=api.EXAMPLE_REQUEST).status_code == 200
    assert len(calls) == 1


def test_artefacto_ausente_no_se_reintenta_por_solicitud(tmp_path, monkeypatch):
    def forbidden_load(*args, **kwargs):
        pytest.fail("No debe cargar un archivo ausente ni reintentar por solicitud")

    monkeypatch.setattr(api.joblib, "load", forbidden_load)
    with TestClient(api.create_app(tmp_path / "missing.joblib")) as connection:
        for _ in range(2):
            assert connection.get("/health").status_code == 503
            assert connection.post("/predict", json=api.EXAMPLE_REQUEST).status_code == 503
        assert connection.get("/docs").status_code == 200


def test_artefacto_incompatible_informa_no_disponible(tmp_path):
    path = tmp_path / "not_pipeline.joblib"
    joblib.dump({"not": "a_pipeline"}, path)
    with TestClient(api.create_app(path)) as connection:
        assert connection.get("/health").status_code == 503


@pytest.mark.parametrize("output", [[-1, 1, 2], [np.nan, 1, 2], [np.inf, 1, 2], [1]])
def test_salidas_invalidas_no_se_ocultan_con_recortes(client, monkeypatch, output):
    monkeypatch.setattr(
        client.app.state.loaded.pipeline, "predict", lambda frame: np.array(output)
    )
    response = client.post("/predict", json=api.EXAMPLE_REQUEST)
    assert response.status_code == 500
    assert "no se alteraron" in response.json()["detail"]


def test_model_path_alternativo_conserva_438_sin_limitar_y_alerta(monkeypatch):
    execution_path = config.REPORTS_DIR / "fase3/ejecucion.json"
    if not execution_path.exists():
        pytest.skip("Requiere los artefactos locales del backtesting auditado")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    path = config.ROOT / execution["cortes"][0]["modelo_local"]
    if not path.exists():
        pytest.skip("Requiere el pipeline del primer corte")
    monkeypatch.setenv("MODEL_PATH", str(path))
    observation = {
        "estacion": 46,
        "hora_inicio": "2026-09-12T21:00:00-06:00",
        "lag_24": 0.0,
        "lag_168": 50.0,
        "capacidad": 15.0,
    }
    body = {"origen_pronostico": "2026-09-12T00:00:00-06:00", "observaciones": [observation]}
    with TestClient(api.create_app()) as connection:
        response = connection.post("/predict", json=body)
        assert response.status_code == 200, response.text
        row = response.json()["predicciones"][0]
        assert row["demanda_predicha"] == pytest.approx(438.011742808447, rel=1e-12)
        assert row["diagnostico"]["extreme_prediction"] is True
        assert response.json()["model"]["artifact_sha256"] == sha256(path.read_bytes()).hexdigest()
        del body["observaciones"][0]["capacidad"]
        no_reference = connection.post("/predict", json=body).json()["predicciones"][0]
        assert no_reference["demanda_predicha"] == row["demanda_predicha"]
        assert no_reference["diagnostico"]["extreme_prediction"] is None
