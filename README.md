# MiBici Demand

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicción de demanda de viajes por estación y hora con los datos abiertos de MiBici, el sistema de bicicletas públicas del Área Metropolitana de Guadalajara.

Proyecto del Equipo 7 para el curso TC5044 Operaciones de Aprendizaje Automático (Tecnológico de Monterrey). Este repositorio corresponde a la **Fase 1**: del análisis del problema a una tabla de entrenamiento versionada con MLflow.

## El problema

Predecir cuántos viajes se iniciarán en cada estación en cada hora, para anticipar dónde y cuándo redistribuir bicicletas. La variable objetivo (`demanda`) cuenta los viajes cuya estación de origen es *s* y cuya hora de inicio cae en la hora *t*. Es una regresión de conteos, con horizonte de las 24 horas del día siguiente. El planteamiento completo y el ML Canvas están en [`references/planteamiento_del_problema.md`](references/planteamiento_del_problema.md).

## Inicio rápido

Requisitos: Python 3.10 o superior, unos 8 GB de memoria y espacio para los datos crudos (3.1 GB) y las tablas derivadas.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q                                   # 14 pruebas
python -m mibici_demand.cli descargar                 # los 142 archivos mensuales y el catálogo de estaciones
python -m mibici_demand.cli procesar                  # limpia mes por mes: nunca hay más de un mes en memoria
python -m mibici_demand.cli construir                 # tabla estación-hora con calendario y rezagos
python -m mibici_demand.cli figuras                   # gráficas del EDA en reports/figures
python -m mibici_demand.cli registrar --version v1    # versiones de datos en MLflow
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Para comparar ventanas de años, cada una se guarda con una etiqueta:

```bash
python -m mibici_demand.cli construir --desde 2022-01 --etiqueta desde2022
python -m mibici_demand.cli registrar --version v2-desde2022 --etiqueta desde2022
python -m mibici_demand.cli figuras --etiqueta desde2022
```

`make data` encadena descargar, procesar y construir; `make test` y `make lint` corren las pruebas y el formato. Usa siempre `python -m` para que MLflow corra con el mismo intérprete que registró los datos. Si hace falta otra base de MLflow, se define `MLFLOW_TRACKING_URI` en un archivo `.env`.

## Estructura del repositorio

Generada con Cookiecutter Data Science; el primer commit es la plantilla intacta.

```
├── Makefile           <- Comandos: data, descargar, procesar, construir, figuras, registrar, test, lint
├── README.md
├── data
│   ├── raw            <- CSV originales, inmutables; no se suben (solo manifest.csv)
│   ├── interim        <- Resultados de la limpieza por mes
│   ├── processed      <- Tablas de entrenamiento
│   └── external       <- Sin uso en esta fase
├── docs               <- Documentación (mkdocs)
├── mibici_demand      <- Paquete del proyecto
│   ├── config.py, dataset.py, cleaning.py, features.py, tracking.py, plots.py, cli.py
│   └── modeling       <- Esqueleto de la plantilla, sin uso en esta fase
├── models             <- Sin contenido en esta fase
├── notebooks          <- 0.01-eq7-eda-demanda-estacion-hora.ipynb
├── references         <- Planteamiento, diccionario de datos, hallazgos y refactorización
├── reports/figures    <- Figuras del EDA (historia completa y ventana desde 2022)
├── tests              <- test_pipeline.py
├── pyproject.toml
└── requirements.txt
```

## Módulos

| Módulo | Responsabilidad |
|---|---|
| `dataset.py` | Descubre los archivos en la página, los descarga sin modificarlos, escribe el manifiesto (nombre, enlace, tamaño y SHA-256) y lee cada mes con tipos explícitos |
| `cleaning.py` | `TripCleaner`: cinco reglas de limpieza, cada una con su conteo |
| `features.py` | Tabla estación-hora, rejilla activa, rezagos sin fuga, preprocesamiento y PCA |
| `tracking.py` | Registro de cada versión de datos en MLflow |
| `plots.py` | Figuras del EDA |
| `cli.py` | Línea de comandos del pipeline |
| `config.py` | Rutas, umbrales y parámetros |

## Versionado de datos en MLflow

`registrar` crea un registro (run) por cada versión, en el experimento `mibici-fase1-versiones-de-datos`, con el dataset como entrada, los parámetros que lo definen, métricas descriptivas y los archivos de documentación.

| Dataset | Qué registra |
|---|---|
| `archivos_crudos` | El manifiesto de los CSV descargados: nombre, enlace, tamaño y SHA-256 de cada uno |
| `viajes_limpios` | El reporte de limpieza por mes y el total de filas que toca cada regla |
| `demanda_estacion_hora` | La tabla de entrenamiento: filas, estaciones, proporción de ceros, varianza del PCA y el baseline estacional |

Para la tabla de entrenamiento, MLflow guarda el esquema y una huella de las primeras 10,000 filas (la tabla completa tiene millones y no cabe en memoria junto con sus copias). La versión completa queda identificada con el parámetro `sha256_tabla` (la huella del archivo Parquet) y con el número real de filas en `filas_tabla_completa`.

## Limpieza y tabla de entrenamiento

Las reglas, los umbrales y lo encontrado en los 142 archivos están en [`references/hallazgos_de_datos.md`](references/hallazgos_de_datos.md); las columnas, en [`references/diccionario_datos.md`](references/diccionario_datos.md).

- Los valores atípicos de demanda no se eliminan: son picos de hora pico. Se eliminan solo los errores.
- Cada estación existe solo entre su primera y su última aparición; las horas sin servicio cuentan como ceros para los rezagos, pero no son filas de entrenamiento.
- Con el horizonte de 24 horas solo se pueden usar los rezagos de 24 y 168 horas; el de 1 hora y la media de 24 horas quedan reservados para un horizonte de una hora.

## Resultados de la Fase 1

| Resultado | Valor |
|---|---|
| Archivos procesados | 142 meses, 39,078,984 filas |
| Filas tras la limpieza | 37,870,722 (se elimina el 3.09%) |
| Tabla de entrenamiento | 22,651,305 filas con la historia completa; 11,917,036 desde 2022 |
| Baseline (misma hora de la semana anterior, última semana) | MAE 1.25, RMSE 2.08 y WAPE 0.64 |

## Roles y su interacción

| Rol | Actividades de la Fase 1 | Interacción |
|---|---|---|
| Data Engineer | Descarga, manifiesto, limpieza por mes, tablas intermedias, rejilla y versiones de datos | Acuerda con el Data Scientist qué es dato válido |
| Data Scientist | Planteamiento y ML Canvas, EDA, definición del objetivo y de las variables, baseline | Recibe la tabla del Data Engineer y define qué variables necesita |
| Software Engineer | Estructura del repositorio, módulos y clases, pruebas, revisión de código | Revisa los cambios de todos |
| Site Reliability Engineer | Entorno reproducible, requisitos, `.gitignore`, base de MLflow, control de memoria y disco | Define con el Data Engineer el procesamiento por mes |
| ML Engineer | Versionado en MLflow, comparación de versiones y preparación de la tabla de entrenamiento | Conecta los datos versionados con quien entrena y evalúa |
