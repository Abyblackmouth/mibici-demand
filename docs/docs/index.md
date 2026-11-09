# MiBici Demand

## Descripción

Predicción de demanda de viajes por estación y hora con los datos abiertos de MiBici. El proyecto lleva los datos desde los CSV mensuales hasta una tabla de entrenamiento versionada con MLflow (Fase 1) y, sobre esa tabla, compara modelos con un pipeline de scikit-learn y registra los experimentos y el modelo campeón en MLflow (Fase 2). Curso TC5044, Equipo 7.

## Comandos

El Makefile reúne los puntos de entrada del proyecto:

| Comando | Qué hace |
|---|---|
| `make requirements` | Instala las dependencias |
| `make data` | Descarga, procesa y construye la tabla de entrenamiento |
| `make descargar`, `make procesar`, `make construir` | Cada paso por separado |
| `make figuras` | Genera las figuras del análisis exploratorio |
| `make registrar` | Registra la versión de los datos en MLflow |
| `make train` | Ajusta, compara y registra los modelos en MLflow |
| `make predict` | Predicción por lotes con el mejor modelo |
| `make test` | Corre las pruebas |
| `make lint` y `make format` | Revisan y aplican el formato del código |

El detalle de cada paso y de los datos está en `README.md` y en la carpeta `references/` del repositorio.

## Seguimiento de experimentos

Los modelos se registran en el experimento `mibici-fase2-modelado` de MLflow, con una corrida por modelo (parámetros, métricas y archivos), los ensayos del ajuste de hiperparámetros como corridas anidadas, y el mejor modelo en el Model Registry como `mibici-demanda-estacion-hora` con el alias `champion`. Las cifras y sus límites están en `references/resultados_fase2.md`.
