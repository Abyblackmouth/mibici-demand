# MiBici Demand

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Prediccion de demanda de viajes por estacion y hora con datos abiertos de MiBici

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici_demand and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici_demand   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici_demand a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

<!-- pipeline-mibici -->
## Pipeline de datos de MiBici (Fase 1)

Lleva los datos abiertos de MiBici desde los CSV mensuales hasta una tabla de entrenamiento de demanda por estación y hora, con cada versión de los datos registrada en MLflow.

**Problema.** Predecir cuántos viajes se iniciarán en cada estación en cada hora, para anticipar dónde y cuándo redistribuir bicicletas. Objetivo (`demanda`): viajes cuya estación de origen es *s* y cuya hora de inicio cae en la hora *t*.

### Uso

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m mibici_demand.cli descargar                 # los 142 archivos mensuales (unos 3 GB) y el catálogo de estaciones
python -m mibici_demand.cli procesar                  # limpia mes por mes: nunca hay más de un mes en memoria
python -m mibici_demand.cli construir                 # tabla estación-hora con calendario y rezagos
python -m mibici_demand.cli figuras                   # gráficas del EDA en reports/figures
python -m mibici_demand.cli registrar --version v1    # versiones de datos en MLflow
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Para comparar ventanas de años se guarda cada una con una etiqueta:

```bash
python -m mibici_demand.cli construir --desde 2022-01 --etiqueta desde2022
python -m mibici_demand.cli registrar --version v2-desde2022 --etiqueta desde2022
python -m mibici_demand.cli figuras --etiqueta desde2022
```

`make data` encadena descargar, procesar y construir. Usa siempre `python -m` para que MLflow corra con el mismo intérprete que registró los datos.

### Qué hace cada módulo

| Módulo | Responsabilidad |
|---|---|
| `dataset.py` | Descubre los archivos en la página, los descarga sin modificarlos, escribe el manifiesto (nombre, url, tamaño, sha256) y lee cada mes con tipos explícitos |
| `cleaning.py` | `TripCleaner`: cinco reglas de limpieza, cada una con su conteo |
| `features.py` | Tabla estación-hora, rejilla activa, rezagos sin fuga, preprocesamiento y PCA |
| `tracking.py` | Registro de cada versión de datos en MLflow |
| `plots.py` | Figuras del EDA |
| `cli.py` | Línea de comandos del pipeline |

### Versionado de datos en MLflow

`registrar` crea un run por cada versión, en el experimento `mibici-fase1-versiones-de-datos`, con el dataset como entrada (`mlflow.log_input`), los parámetros que lo definen, métricas descriptivas y los archivos de documentación:

| Dataset | Qué registra |
|---|---|
| `archivos_crudos` | El manifiesto de los CSV descargados: nombre, enlace, tamaño y SHA-256 de cada uno |
| `viajes_limpios` | El reporte de limpieza por mes y el total de filas que toca cada regla |
| `demanda_estacion_hora` | La tabla de entrenamiento: filas, estaciones, proporción de ceros, varianza del PCA y el baseline estacional |

Para la tabla de entrenamiento, MLflow guarda el esquema y una huella de las **primeras 10,000 filas** (la tabla completa tiene millones y no cabe en memoria junto con sus copias). La versión completa queda identificada con el parámetro `sha256_tabla` (la huella del archivo Parquet) y el número real de filas en `filas_tabla_completa`.

### Reglas de limpieza y de la rejilla

Las reglas, los umbrales y lo que se encontró en los 142 archivos están en [`references/hallazgos_de_datos.md`](references/hallazgos_de_datos.md); el diccionario de datos, en [`references/diccionario_datos.md`](references/diccionario_datos.md).

- Los valores atípicos de demanda no se eliminan: son picos de hora pico. Se eliminan solo los errores.
- Cada estación existe solo entre su primera y su última aparición; las horas sin servicio cuentan como ceros para los rezagos pero no son filas de entrenamiento.
- Alcance de la Fase 1: hasta la tabla de entrenamiento y el baseline estacional. Los modelos corresponden a la Fase 2.

### Roles (borrador)

| Rol | Actividades de la Fase 1 | Interacción |
|---|---|---|
| Data Engineer | Descarga, manifiesto, limpieza por mes, tablas intermedias, rejilla y versiones de datos | Acuerda con el Data Scientist qué es dato válido |
| Data Scientist | Planteamiento y ML Canvas, EDA, definición del objetivo y de las variables, baseline | Recibe la tabla del Data Engineer y define qué variables necesita |
| Software Engineer | Estructura del repositorio, módulos y clases, pruebas, revisión de código | Revisa los cambios de todos |
| Site Reliability Engineer | Entorno reproducible, requisitos, `.gitignore`, servidor y base de MLflow, control de memoria y disco | Define con el Data Engineer el procesamiento por mes |
| ML Engineer | Versionado en MLflow, comparación de versiones, preparación de la tabla para la Fase 2 | Conecta los datos con el entrenamiento y la futura API |
