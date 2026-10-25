"""Rutas y parámetros del proyecto. Todo lo que se puede ajustar vive aquí."""

from __future__ import annotations

import os
from pathlib import Path

try:  # lee MLFLOW_TRACKING_URI y otras variables del archivo .env, si existe
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"  # datos originales: no se modifican nunca
INTERIM_DIR = ROOT / "data" / "interim"  # resultados intermedios por mes y por perfil de limpieza
PROCESSED_DIR = ROOT / "data" / "processed"  # tablas finales de entrenamiento
FIGURES_DIR = ROOT / "reports" / "figures"

# Nombres que usa la plantilla Cookiecutter Data Science (los módulos de modeling/ los importan)
PROJ_ROOT = ROOT
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = RAW_DIR
INTERIM_DATA_DIR = INTERIM_DIR
PROCESSED_DATA_DIR = PROCESSED_DIR
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

OPEN_DATA_URL = "https://mibici.net/en/open-data/"

# Contrato de los archivos mensuales de viajes
TRIP_COLUMNS = [
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]

# Reglas de limpieza (cada umbral se justifica en el README)
MIN_TRIP_SECONDS = (
    60  # un viaje más corto que esto, con el mismo origen y destino, es un anclaje fallido
)
MAX_TRIP_SECONDS = 24 * 3600  # más de un día no es un viaje real
BIRTH_YEAR_MIN = 1920  # año de nacimiento plausible: de 1920 a (año del viaje - 10)
MIN_AGE = 10

# Horas del día sin servicio: las que concentran menos de este porcentaje de los viajes del mes
CLOSED_HOUR_SHARE = 0.001

LAGS = (1, 24, 168)  # rezagos de demanda en horas
ROLLING_HOURS = 24

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = "mibici-fase1-versiones-de-datos"
