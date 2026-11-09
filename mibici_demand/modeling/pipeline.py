"""Pipeline de modelado y preprocesamiento con Scikit-Learn para la Fase 2 de MiBici.

Este módulo encapsula en un único `sklearn.pipeline.Pipeline` reproducible todas las
transformaciones de ingeniería de características y preprocesamiento previas al estimador:

1. **Enriquecimiento por estación sin fuga de datos (`StationProfileTransformer`)**:
   Calcula exclusivamente sobre el conjunto de entrenamiento la demanda media histórica,
   la proporción de viajes en horas pico (`config.PEAK_HOURS`) y los 3 componentes
   principales (`pc1`, `pc2`, `pc3`) del perfil semanal de 168 horas (`station_profile_pca`).
   Durante la inferencia o validación, las estaciones nuevas o con poco historial
   (riesgo H14 en `references/hallazgos_de_datos.md`) se imputan con la mediana global
   del entrenamiento.

2. **Preprocesamiento diferencial por tipo de variable y familia de modelo (`ColumnTransformer`)**:
   - **Variables numéricas** (rezagos válidos del horizonte `config.HORIZON_LAGS[horizon]`,
     estadísticas/componentes del perfil de estación y metadatos geográficos/capacidad
     `config.STATION_META_COLS`): imputación por mediana (`SimpleImputer(strategy="median")`)
     seguida de estandarización (`StandardScaler()`).
   - **Variables de calendario cíclicas/categóricas** (`["hora", "dia_semana", "mes"]`):
     codificación one-hot (`OneHotEncoder(handle_unknown="ignore")`).
   - **Indicador binario** (`["es_fin_semana"]`): paso directo (`"passthrough"`).
   - **Identificador de estación** (`["estacion"]`): codificación ordinal con soporte para
     estaciones desconocidas (`OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`)
     aplicada únicamente en modelos basados en árboles (`"hist_gbm_poisson"` y `"random_forest"`),
     evitando introducir un orden lineal artificial en modelos lineales (`"ridge"` y `"poisson"`).

3. **Estimador de regresión de conteos**:
   Soporta cuatro algoritmos adaptados a la distribución sesgada y no negativa de viajes
   por estación-hora (43.4% de ceros, H12 en `references/hallazgos_de_datos.md`):
   - `"ridge"`: regresión lineal con regularización L2 (referencia lineal rápida).
   - `"poisson"`: modelo lineal generalizado (GLM) con enlace logarítmico y pérdida Poisson.
   - `"hist_gbm_poisson"`: Gradient Boosting basado en histogramas con pérdida Poisson.
   - `"random_forest"`: ensamble paralelo de árboles de decisión.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from mibici_demand import config, features

TREE_MODELS = frozenset({"hist_gbm_poisson", "random_forest"})
LINEAR_MODELS = frozenset({"ridge", "poisson"})
SUPPORTED_MODELS = TREE_MODELS | LINEAR_MODELS


class _AvailableColumnSelector:
    """Selector de columnas compatible con `ColumnTransformer` que filtra las columnas presentes en `X`.

    Permite declarar las columnas numéricas esperadas (incluyendo `config.STATION_META_COLS` cuando
    `include_station_meta=True`) sin fallar si un `DataFrame` de prueba no incluye el catálogo
    externo de estaciones, y a la vez se comporta como secuencia/lista para inspección estática.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = list(columns)

    def __call__(self, df: pd.DataFrame) -> list[str]:
        if hasattr(df, "columns"):
            return [col for col in self.columns if col in df.columns]
        return list(self.columns)

    def __iter__(self):
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def __getitem__(self, item: int | slice) -> Any:
        return self.columns[item]

    def __contains__(self, item: object) -> bool:
        return item in self.columns

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _AvailableColumnSelector):
            return self.columns == other.columns
        if isinstance(other, (list, tuple)):
            return self.columns == list(other)
        return False

    def __repr__(self) -> str:
        return repr(self.columns)


def _build_estimator(model_name: str, params: dict[str, Any] | None = None) -> Any:
    """Instancia el estimador de Scikit-Learn configurado con hiperparámetros base y sobrescrituras.

    Parameters
    ----------
    model_name : str
        Identificador del algoritmo: `"ridge"`, `"poisson"`, `"hist_gbm_poisson"` o `"random_forest"`.
    params : dict[str, Any] | None, optional
        Diccionario de hiperparámetros adicionales o de sobrescritura para el estimador.

    Returns
    -------
    estimator
        Estimador de regresión de Scikit-Learn listo para integrarse al `Pipeline`.
    """
    custom_params = dict(params) if params else {}

    if model_name == "ridge":
        defaults: dict[str, Any] = {"random_state": config.RANDOM_SEED}
        defaults.update(custom_params)
        return Ridge(**defaults)

    if model_name == "poisson":
        defaults = {"max_iter": 300}
        defaults.update(custom_params)
        return PoissonRegressor(**defaults)

    if model_name == "hist_gbm_poisson":
        defaults = {
            "loss": "poisson",
            "random_state": config.RANDOM_SEED,
        }
        defaults.update(custom_params)
        return HistGradientBoostingRegressor(**defaults)

    if model_name == "random_forest":
        defaults = {
            "n_jobs": -1,
            "random_state": config.RANDOM_SEED,
        }
        defaults.update(custom_params)
        return RandomForestRegressor(**defaults)

    raise ValueError(
        f"Modelo '{model_name}' no reconocido. Algoritmos soportados: {sorted(SUPPORTED_MODELS)}"
    )


def build_model_pipeline(
    model_name: str,
    params: dict[str, Any] | None = None,
    horizon: str = config.DEFAULT_HORIZON,
    include_station_meta: bool = True,
) -> Pipeline:
    """Construye el `Pipeline` completo de preprocesamiento y modelado para predecir demanda horaria.

    El pipeline ejecuta tres etapas secuenciales que garantizan la ausencia de fuga de datos
    (data leakage) entre entrenamiento, validación y prueba:

    1. `("perfil_estacion", StationProfileTransformer())`:
       Calcula en `fit(X, y)` la demanda media por estación (`est_demanda_media`), la proporción
       de demanda en horas pico (`est_prop_hora_pico`) y los 3 componentes principales del perfil
       semanal (`pc1`, `pc2`, `pc3`). En `transform(X)` une estos atributos por `estacion` y
       rellena estaciones nuevas (riesgo H14) con la mediana global del entrenamiento.
    2. `("preprocesador", ColumnTransformer(...))`:
       - Escala e imputa con la mediana las variables numéricas: rezagos válidos del horizonte
         (`config.HORIZON_LAGS[horizon]`), componentes del perfil de estación y, si
         `include_station_meta=True`, los metadatos de estación (`latitud`, `longitud`, `capacidad`).
       - Codifica con `OneHotEncoder(handle_unknown="ignore")` las variables `["hora", "dia_semana", "mes"]`.
       - Pasa `["es_fin_semana"]` e incluye `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`
         sobre `["estacion"]` únicamente cuando `model_name` pertenece a los modelos basados en árboles.
    3. `("modelo", estimator)`:
       Ajusta el algoritmo seleccionado (`"ridge"`, `"poisson"`, `"hist_gbm_poisson"` o `"random_forest"`).

    Parameters
    ----------
    model_name : str
        Nombre del algoritmo a entrenar:
        - `"ridge"`: `Ridge(random_state=config.RANDOM_SEED)`
        - `"poisson"`: `PoissonRegressor(max_iter=300)`
        - `"hist_gbm_poisson"`: `HistGradientBoostingRegressor(loss="poisson", random_state=config.RANDOM_SEED)`
        - `"random_forest"`: `RandomForestRegressor(n_jobs=-1, random_state=config.RANDOM_SEED)`
    params : dict[str, Any] | None, optional
        Hiperparámetros específicos para sobrescribir los valores por defecto del estimador.
    horizon : str, default=config.DEFAULT_HORIZON
        Horizonte operativo de predicción (`"24h"` o `"1h"`). Controla qué rezagos de
        `config.HORIZON_LAGS` se permiten para evitar fuga de información temporal.
    include_station_meta : bool, default=True
        Si es `True`, incorpora las columnas de `config.STATION_META_COLS` (`latitud`,
        `longitud`, `capacidad`) al bloque numérico del preprocesador.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline de Scikit-Learn con los pasos `perfil_estacion`, `preprocesador` y `modelo`.
    """
    if horizon not in config.HORIZON_LAGS:
        raise ValueError(
            f"Horizonte '{horizon}' inválido. Horizontes configurados: {list(config.HORIZON_LAGS)}"
        )

    estimator = _build_estimator(model_name=model_name, params=params)

    numeric_columns: list[str] = list(config.HORIZON_LAGS[horizon]) + list(
        features.PROFILE_COLUMNS
    )
    if include_station_meta:
        numeric_columns.extend(config.STATION_META_COLS)

    numeric_transformer = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
        ]
    )

    transformers: list[tuple[str, Any, Any]] = [
        ("numericas", numeric_transformer, _AvailableColumnSelector(numeric_columns)),
        (
            "calendario",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["hora", "dia_semana", "mes"],
        ),
        ("fin_semana", "passthrough", ["es_fin_semana"]),
    ]

    if model_name in TREE_MODELS:
        transformers.append(
            (
                "estacion",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ["estacion"],
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    return Pipeline(
        steps=[
            ("perfil_estacion", features.StationProfileTransformer()),
            ("preprocesador", preprocessor),
            ("modelo", estimator),
        ]
    )
