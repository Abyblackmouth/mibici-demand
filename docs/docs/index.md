# MiBici Demand

## Descripción

Predicción de demanda de viajes por estación y hora con los datos abiertos de MiBici. El proyecto lleva los datos desde los CSV mensuales hasta una tabla de entrenamiento versionada con MLflow (Fase 1 del curso TC5044, Equipo 7).

## Comandos

El Makefile reúne los puntos de entrada del proyecto:

| Comando | Qué hace |
|---|---|
| `make requirements` | Instala las dependencias |
| `make data` | Descarga, procesa y construye la tabla de entrenamiento |
| `make descargar`, `make procesar`, `make construir` | Cada paso por separado |
| `make figuras` | Genera las figuras del análisis exploratorio |
| `make registrar` | Registra la versión de los datos en MLflow |
| `make test` | Corre las pruebas |
| `make lint` y `make format` | Revisan y aplican el formato del código |

El detalle de cada paso y de los datos está en `README.md` y en la carpeta `references/` del repositorio.
