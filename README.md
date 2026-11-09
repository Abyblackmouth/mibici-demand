# MiBici Demand

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicción de demanda de viajes por estación y hora con los datos abiertos de MiBici, el sistema de bicicletas públicas del Área Metropolitana de Guadalajara.

Proyecto del Equipo 7 para el curso TC5044 Operaciones de Aprendizaje Automático (Tecnológico de Monterrey). El repositorio cubre la **Fase 1** (de los datos a una tabla de entrenamiento versionada) y la **Fase 2** (modelos, ajuste de hiperparámetros, seguimiento de experimentos y registro de modelos con MLflow).

## El problema

Predecir cuántos viajes se iniciarán en cada estación en cada hora, para anticipar dónde y cuándo redistribuir bicicletas. La variable objetivo (`demanda`) cuenta los viajes cuya estación de origen es *s* y cuya hora de inicio cae en la hora *t*. Es una regresión de conteos con horizonte de las 24 horas del día siguiente. El planteamiento y el ML Canvas están en [`references/planteamiento_del_problema.md`](references/planteamiento_del_problema.md).

## Inicio rápido

Requisitos: Python 3.10 o superior, unos 8 GB de memoria y espacio para los datos crudos (3 GB) y las tablas derivadas.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q                                   # 17 pruebas
python -m mibici_demand.cli descargar                 # archivos mensuales y catálogo de estaciones
python -m mibici_demand.cli procesar                  # limpia mes por mes: nunca hay más de un mes en memoria
python -m mibici_demand.cli construir                 # tabla estación-hora con calendario y rezagos
python -m mibici_demand.cli registrar --version v1    # versiones de datos en MLflow
python -m mibici_demand.cli entrenar --registrar-mlflow   # ajusta, compara y registra modelos
python -m mibici_demand.cli predecir                  # predicciones por lotes con el mejor modelo
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`make data` encadena descargar, procesar y construir; `make train` y `make predict` corren el entrenamiento y la predicción; `make test` y `make lint` corren las pruebas y el formato. Usa siempre `python -m` para que MLflow corra con el mismo intérprete. Si hace falta otra base de MLflow, se define `MLFLOW_TRACKING_URI` en un archivo `.env`.

Opciones de `entrenar`: `--horizonte 24h|1h`, `--max-train-months` (meses de entrenamiento, 12 por defecto), `--modelos`, `--n-iter` y `--etiqueta` (qué versión de la tabla leer).

## Estructura del repositorio

Generada con Cookiecutter Data Science; el primer commit es la plantilla intacta.

```
├── Makefile           <- Comandos: data, train, predict, test, lint y los pasos del pipeline
├── README.md
├── data
│   ├── raw            <- CSV originales, inmutables; no se suben (solo manifest.csv)
│   ├── interim        <- Resultados de la limpieza por mes
│   ├── processed      <- Tablas de entrenamiento y predicciones
│   └── external       <- Sin uso
├── docs               <- Documentación (mkdocs)
├── mibici_demand      <- Paquete del proyecto
│   ├── config.py, dataset.py, cleaning.py, features.py, tracking.py, plots.py, cli.py
│   └── modeling       <- pipeline.py, train.py y predict.py
├── models             <- best_pipeline.joblib (lo genera `make train`)
├── notebooks          <- EDA (0.01) y modelado (0.02)
├── references         <- Planteamiento, datos, hallazgos, refactorización y resultados de la Fase 2
├── reports            <- Resultados de los modelos y figuras
├── tests              <- test_pipeline.py
├── pyproject.toml
└── requirements.txt
```

## Módulos

| Módulo | Responsabilidad |
|---|---|
| `dataset.py` | Descubre y descarga los archivos, escribe el manifiesto (nombre, enlace, tamaño y SHA-256) y lee cada mes con tipos explícitos |
| `cleaning.py` | `TripCleaner`: cinco reglas de limpieza, cada una con su conteo |
| `features.py` | Tabla estación-hora, rezagos sin fuga, selección de variables por horizonte y `StationProfileTransformer` |
| `modeling/pipeline.py` | `Pipeline` de scikit-learn: perfil de estación, preprocesamiento y estimador |
| `modeling/train.py` | Partición temporal, ajuste con `TimeSeriesSplit`, evaluación y comparación de modelos |
| `modeling/predict.py` | Predicción por lotes y presión de redistribución (demanda predicha entre capacidad) |
| `tracking.py` | Registro de versiones de datos, de corridas de modelos y del campeón en MLflow |
| `plots.py` | Figuras del EDA y de la comparación de modelos |
| `cli.py` | Línea de comandos del pipeline |
| `config.py` | Rutas, umbrales, semillas, horizontes y nombres de experimento y modelo |

## Pipeline de scikit-learn

Un solo `Pipeline` encadena tres pasos, de modo que el preprocesamiento se ajusta solo con los datos de entrenamiento de cada partición:

1. `perfil_estacion`: calcula por estación la demanda media, la proporción de viajes en horas pico y tres componentes principales del perfil semanal, solo con el entrenamiento; a las estaciones nuevas les imputa la mediana global.
2. `preprocesador`: imputa y estandariza las variables numéricas, codifica en *one-hot* hora, día de la semana y mes, y codifica la estación de forma ordinal solo en los modelos de árboles.
3. `modelo`: ridge, regresión de Poisson, gradient boosting con pérdida de Poisson o Random Forest.

Con el horizonte de 24 horas solo se usan los rezagos de 24 y 168 horas; el rezago de 1 hora y la media de 24 horas quedan para el horizonte de una hora.

## Evaluación y seguimiento

- **Partición:** cronológica, nunca al azar. Entrenamiento, 7 días de validación y los últimos 7 días como prueba.
- **Ajuste:** `RandomizedSearchCV` con `TimeSeriesSplit`; el campeón se elige por el error de **validación**.
- **Métricas:** MAE, RMSE, WAPE y R², globales y en horas pico, contra dos referencias (la misma hora de la semana anterior y del día anterior).
- **MLflow:** experimento `mibici-fase2-modelado`, una corrida por modelo con sus parámetros y métricas, los ensayos del ajuste como corridas anidadas, y el modelo registrado como `mibici-demanda-estacion-hora` con el alias `champion`.

El detalle, con las cifras y sus límites, está en [`references/resultados_fase2.md`](references/resultados_fase2.md).

## Resultados de la Fase 2 (prueba, últimos 7 días)

| Modelo | MAE | RMSE | WAPE | WAPE horas pico |
|---|---|---|---|---|
| `hist_gbm_poisson` (campeón) | 1.103 | 1.751 | 0.510 | 0.447 |
| `random_forest` | 1.137 | 1.806 | 0.526 | 0.459 |
| `ridge` | 1.233 | 1.935 | 0.571 | 0.502 |
| Referencia: misma hora de la semana anterior | 1.367 | 2.273 | 0.633 | 0.558 |
| `poisson` | 1.498 | 4.346 | 0.693 | 0.665 |
| Referencia: misma hora del día anterior | 1.518 | 2.624 | 0.702 | 0.654 |

## Limpieza y tabla de entrenamiento

Las reglas, los umbrales y lo encontrado en los datos están en [`references/hallazgos_de_datos.md`](references/hallazgos_de_datos.md); las columnas, en [`references/diccionario_datos.md`](references/diccionario_datos.md).

- Los valores atípicos de demanda no se eliminan: son picos de hora pico. Se eliminan solo los errores.
- Cada estación existe solo entre su primera y su última aparición; las horas sin servicio cuentan como ceros para los rezagos, pero no son filas de entrenamiento.

## Roles y su interacción

| Rol | Actividades | Interacción |
|---|---|---|
| Data Engineer | Descarga, limpieza, tablas intermedias y la tabla de entrenamiento con sus rezagos y metadatos de estación | Entrega al Data Scientist la tabla con el esquema acordado |
| Data Scientist | Planteamiento, EDA, elección de algoritmos y métricas, referencias de comparación | Define con el ML Engineer los rezagos válidos del horizonte |
| ML Engineer | Pipeline de scikit-learn, ajuste con validación temporal, seguimiento en MLflow y registro del campeón | Recibe la tabla del Data Engineer y entrega el modelo registrado |
| Software Engineer | Módulos, línea de comandos, pruebas y revisión de código | Revisa los cambios de todos |
| Site Reliability Engineer | Entorno reproducible, límites de memoria (ventana de entrenamiento y filas de ajuste), semillas y base de MLflow | Acuerda con el ML Engineer los límites de memoria |
