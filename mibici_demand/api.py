r"""Servicio MiBici a 24 h; candidato técnico común sin aprobación productiva.

PowerShell: .venv\Scripts\python.exe -m uvicorn mibici_demand.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha256
from io import BytesIO
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal
import warnings
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from sklearn.exceptions import InconsistentVersionWarning
from threadpoolctl import threadpool_limits

from mibici_demand import config, features
from mibici_demand.release import artifact_contract, default_model_path

SERVICE_VERSION = "0.2.0"
LOCAL_TIMEZONE = ZoneInfo("America/Mexico_City")
MODEL_USAGE = "referencia_integracion_no_validado_prospectivamente"
INPUT_DTYPES = {
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
BASE_FEATURES = {"estacion", "hora", "dia_semana", "mes", "es_fin_semana", "lag_24", "lag_168"}
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Lag = Annotated[
    float, Field(strict=True, ge=0, le=float(np.finfo(np.float32).max), allow_inf_nan=False)
]
logger = logging.getLogger(__name__)

EXAMPLE_REQUEST = {
    "origen_pronostico": "2026-09-12T00:00:00-06:00",
    "observaciones": [
        {
            "estacion": 2,
            "hora_inicio": "2026-09-12T08:00:00-06:00",
            "lag_24": 1.0,
            "lag_168": 1.0,
            "latitud": 20.6744,
            "longitud": -103.362224,
            "capacidad": 15.0,
        },
        {
            "estacion": 46,
            "hora_inicio": "2026-09-12T21:00:00-06:00",
            "lag_24": 0.0,
            "lag_168": 50.0,
            "latitud": 20.6744,
            "longitud": -103.362224,
            "capacidad": 15.0,
        },
        {
            "estacion": 999,
            "hora_inicio": "2026-09-12T08:00:00-06:00",
            "lag_24": 1.0,
            "lag_168": 1.0,
            "latitud": 20.6744,
            "longitud": -103.362224,
            "capacidad": 15.0,
        },
    ],
}


def local_hour(value: Any) -> datetime:
    if isinstance(value, str):
        if "T" not in value:
            raise ValueError("Usa una fecha ISO 8601 completa: AAAA-MM-DDTHH:00:00[-06:00].")
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Fecha ISO 8601 inválida.") from exc
    if not isinstance(value, datetime):
        # Pydantic convierte ValueError en 422; TypeError escaparía como error interno.
        raise ValueError("Se requiere una fecha ISO 8601, no un timestamp numérico.")  # noqa: TRY004
    local = (
        value.replace(tzinfo=LOCAL_TIMEZONE)
        if value.tzinfo is None
        else value.astimezone(LOCAL_TIMEZONE)
    )
    if local.minute or local.second or local.microsecond:
        raise ValueError(
            "La fecha debe identificar una hora completa (minutos y segundos en cero)."
        )
    return local


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    estacion: Annotated[
        StrictInt,
        Field(
            ge=1,
            le=int(np.iinfo(np.int16).max),
            description="ID positivo representable en int16; puede ser una estación desconocida.",
        ),
    ]
    hora_inicio: datetime = Field(
        description="Hora objetivo ISO 8601; se normaliza a America/Mexico_City. Sin zona se interpreta como hora local."
    )
    lag_24: Lag = Field(
        description="Conteo disponible al emitir el pronóstico para la misma hora del día anterior; conversión explícita a float32."
    )
    lag_168: Lag = Field(
        description="Conteo disponible al emitir el pronóstico para la misma hora de la semana anterior; conversión explícita a float32."
    )
    latitud: Annotated[Number, Field(ge=-90, le=90)] | None = Field(
        default=None, description="float64. Obligatoria si el artefacto fue ajustado con latitud."
    )
    longitud: Annotated[Number, Field(ge=-180, le=180)] | None = Field(
        default=None, description="float64. Obligatoria si el artefacto fue ajustado con longitud."
    )
    capacidad: Annotated[Number, Field(gt=0)] | None = Field(
        default=None,
        description="float64. Anclajes; requerida por el histórico. También referencia diagnóstica, nunca límite de predicción.",
    )

    _normalize_hour = field_validator("hora_inicio", mode="before")(local_hour)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_REQUEST})

    origen_pronostico: datetime = Field(
        description="Emisión diaria a las 00:00 locales; las observaciones deben estar dentro de las siguientes 24 horas."
    )
    observaciones: list[Observation] = Field(min_length=1)
    _normalize_origin = field_validator("origen_pronostico", mode="before")(local_hour)

    @model_validator(mode="after")
    def validate_horizon(self):
        if self.origen_pronostico.hour:
            raise ValueError("El origen del pronóstico diario debe ser a las 00:00 locales.")
        try:
            end = self.origen_pronostico + timedelta(days=1)
        except OverflowError as exc:
            raise ValueError("Fecha fuera del rango de un pronóstico de 24 horas.") from exc
        seen = set()
        for row in self.observaciones:
            if not self.origen_pronostico <= row.hora_inicio < end:
                raise ValueError(
                    "Cada hora_inicio debe pertenecer a [origen_pronostico, origen + 24h)."
                )
            key = (row.estacion, row.hora_inicio)
            if key in seen:
                raise ValueError("No se permiten observaciones estación/hora duplicadas.")
            seen.add(key)
        return self


class ArtifactInfo(BaseModel):
    model_version: str
    artifact_sha256: str
    uso: Literal[
        "referencia_integracion_no_validado_prospectivamente",
        "candidato_temporal_integracion_no_aprobado_produccion",
    ] = MODEL_USAGE
    release_id: str | None = None
    forecast_origin_min: datetime | None = None
    model_feature_columns: list[str]
    required_observation_fields: list[str]
    input_dtypes: dict[str, str]
    fitted_imputer_dtype: str


class Diagnostic(BaseModel):
    extreme_prediction: bool | None
    reference_capacity: float | None
    criterion: Literal["prediccion_mayor_que_capacidad", "sin_referencia_de_capacidad"]
    unknown_station: bool


class Prediction(BaseModel):
    estacion: int
    hora_inicio: datetime
    demanda_predicha: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    diagnostico: Diagnostic


class PredictResponse(BaseModel):
    model: ArtifactInfo
    horizonte: Literal["24h"] = "24h"
    origen_pronostico: datetime
    predicciones: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    service_version: str = SERVICE_VERSION
    model_available: bool
    model: ArtifactInfo | None
    detail: str | None = None


@dataclass
class LoadedModel:
    pipeline: Any
    info: ArtifactInfo
    known_stations: set[int]
    lock: Lock = field(default_factory=Lock)


def load_model(path: Path) -> LoadedModel:
    # El hash identifica exactamente los bytes deserializados, aunque el archivo cambie después.
    payload = path.read_bytes()
    digest = sha256(payload).hexdigest()
    expected_digest = os.getenv("MODEL_SHA256")
    if expected_digest and digest != expected_digest:
        raise ValueError("El SHA-256 no coincide con MODEL_SHA256.")
    release = artifact_contract(path, digest)
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        pipeline = joblib.load(BytesIO(payload))
    steps = getattr(pipeline, "named_steps", {})
    if not {"perfil_estacion", "preprocesador", "modelo"}.issubset(steps):
        raise ValueError("Se requiere un pipeline MiBici con perfil, preprocesador y modelo.")
    profile = steps["perfil_estacion"]
    preprocessor = steps["preprocesador"]
    fitted_columns = list(preprocessor.feature_names_in_)
    columns = [name for name in fitted_columns if name not in features.PROFILE_COLUMNS]
    if not BASE_FEATURES.issubset(columns) or set(columns) - INPUT_DTYPES.keys():
        raise ValueError(
            "El artefacto no es compatible con el contrato de variables del horizonte 24h."
        )
    metadata = [c for c in config.STATION_META_COLS if c in columns]
    info = ArtifactInfo(
        model_version=f"sha256:{digest}",
        artifact_sha256=digest,
        model_feature_columns=columns,
        required_observation_fields=["estacion", "hora_inicio", "lag_24", "lag_168", *metadata],
        input_dtypes={
            name: (release["input_dtypes"] if release else INPUT_DTYPES)[name] for name in columns
        },
        uso=release["uso"] if release else MODEL_USAGE,
        release_id=release["release_id"] if release else None,
        forecast_origin_min=local_hour(release["corte"].replace(" ", "T")) if release else None,
        fitted_imputer_dtype=str(
            preprocessor.named_transformers_["numericas"]
            .named_steps["imputador"]
            .statistics_.dtype
        ),
    )
    return LoadedModel(pipeline, info, {int(s) for s in profile.station_profiles_["estacion"]})


def feature_frame(payload: PredictRequest, loaded: LoadedModel) -> pd.DataFrame:
    records, errors = [], []
    required_metadata = [
        c for c in config.STATION_META_COLS if c in loaded.info.model_feature_columns
    ]
    for i, row in enumerate(payload.observaciones):
        for column in required_metadata:
            if getattr(row, column) is None:
                errors.append(
                    {
                        "loc": ["body", "observaciones", i, column],
                        "msg": f"Este artefacto requiere {column}; consulta /health.",
                        "type": "missing_model_feature",
                    }
                )
        date = row.hora_inicio
        records.append(
            {
                "estacion": row.estacion,
                "hora": date.hour,
                "dia_semana": date.weekday(),
                "mes": date.month,
                "es_fin_semana": int(date.weekday() >= 5),
                "lag_24": row.lag_24,
                "lag_168": row.lag_168,
                "latitud": row.latitud,
                "longitud": row.longitud,
                "capacidad": row.capacidad,
            }
        )
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    # Solo se envían columnas usadas en fit: el modelo sin metadatos mantiene su bloque float32.
    return pd.DataFrame(records, columns=loaded.info.model_feature_columns).astype(
        loaded.info.input_dtypes
    )


def create_app(model_path: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            configured = Path(model_path or os.getenv("MODEL_PATH") or default_model_path())
            path = configured if configured.is_absolute() else config.ROOT / configured
            application.state.loaded = load_model(path)
            application.state.load_error = None
        except Exception:
            logger.exception("No se pudo cargar el artefacto configurado en MODEL_PATH")
            application.state.loaded = None
            application.state.load_error = "Modelo no disponible; revisa MODEL_PATH, el archivo y la compatibilidad del pipeline."
        yield
        application.state.loaded = None

    application = FastAPI(
        title="MiBici: inferencia de demanda por estación y hora",
        version=SERVICE_VERSION,
        description="Candidato temporal para integración, sin aprobación productiva. /health declara hash, uso y corte mínimo. Calendario derivado de la fecha; rezagos construidos antes de la emisión. MODEL_PATH permite comprobar referencias históricas por separado.",
        lifespan=lifespan,
    )
    application.state.loaded = None
    application.state.load_error = "El ciclo de inicio del servicio todavía no se ha ejecutado."

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        messages = {
            "missing": "Campo obligatorio.",
            "extra_forbidden": "Campo no permitido; no envíes calendario manual, etiquetas ni rezagos de otro horizonte.",
            "int_type": "Debe ser un entero JSON, no texto ni booleano.",
            "float_type": "Debe ser un número JSON, no texto ni booleano.",
            "finite_number": "Debe ser un número finito.",
            "too_short": "El lote debe contener al menos una observación.",
        }
        detail = [
            {
                "loc": list(e["loc"]),
                "msg": messages.get(e["type"], e["msg"].removeprefix("Value error, ")),
                "type": e["type"],
            }
            for e in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @application.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        summary="Estado y disponibilidad del artefacto",
    )
    def health(request: Request, response: Response):
        loaded = request.app.state.loaded
        if loaded is None:
            response.status_code = 503
        return HealthResponse(
            status="ready" if loaded else "unavailable",
            model_available=loaded is not None,
            model=loaded.info if loaded else None,
            detail=request.app.state.load_error,
        )

    @application.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            503: {"description": "Modelo no disponible"},
            500: {"description": "La inferencia no produjo conteos finitos no negativos"},
        },
        summary="Predice un lote estación/hora dentro de las siguientes 24 horas",
    )
    def predict(payload: PredictRequest, request: Request):
        loaded = request.app.state.loaded
        if loaded is None:
            raise HTTPException(status_code=503, detail=request.app.state.load_error)
        if (
            loaded.info.forecast_origin_min
            and payload.origen_pronostico < loaded.info.forecast_origin_min
        ):
            raise HTTPException(
                status_code=422,
                detail="El origen es anterior al corte de entrenamiento del candidato; no usar información futura para predecir el pasado.",
            )
        frame = feature_frame(payload, loaded)
        try:
            with loaded.lock, threadpool_limits(limits=4):
                raw = np.asarray(loaded.pipeline.predict(frame), dtype=np.float64)
            if raw.shape != (len(frame),) or not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError("Salida inválida del modelo.")
        except Exception as error:
            logger.exception("Falló la inferencia del artefacto %s", loaded.info.artifact_sha256)
            raise HTTPException(
                status_code=500,
                detail="El modelo no produjo un conteo finito no negativo por observación; no se alteraron sus salidas para ocultarlo.",
            ) from error
        predictions = []
        for row, value in zip(payload.observaciones, raw):
            reference = row.capacidad
            predictions.append(
                Prediction(
                    estacion=row.estacion,
                    hora_inicio=row.hora_inicio,
                    demanda_predicha=float(value),
                    diagnostico=Diagnostic(
                        extreme_prediction=bool(value > reference)
                        if reference is not None
                        else None,
                        reference_capacity=reference,
                        criterion="prediccion_mayor_que_capacidad"
                        if reference is not None
                        else "sin_referencia_de_capacidad",
                        unknown_station=row.estacion not in loaded.known_stations,
                    ),
                )
            )
        return PredictResponse(
            model=loaded.info,
            origen_pronostico=payload.origen_pronostico,
            predicciones=predictions,
        )

    return application


app = create_app()
