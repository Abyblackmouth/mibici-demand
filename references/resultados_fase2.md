# Resultados de la Fase 2: modelos, ajuste y registro

Predicción de la demanda de viajes por estación y hora, con horizonte de las 24 horas del día siguiente. Los resultados salen de `python -m mibici_demand.cli entrenar --registrar-mlflow` y están en `reports/resultados_modelos.csv` y `.json`.

## Diseño del experimento

| Elemento | Decisión |
|---|---|
| Objetivo | `demanda`: viajes iniciados en la estación en la hora |
| Variables | Calendario (hora, día de la semana, mes, fin de semana), estación, metadatos de la estación (latitud, longitud, capacidad), perfil histórico de la estación (demanda media, proporción en horas pico, 3 componentes principales) y rezagos de 24 y 168 horas |
| Partición | Cronológica, nunca al azar: entrenamiento con los últimos 6 meses previos a la validación (1,556,606 filas), validación de 7 días (65,520 filas) y prueba de los últimos 7 días (65,046 filas) |
| Pipeline | Un `Pipeline` de scikit-learn: perfil de estación calculado solo con el entrenamiento, preprocesamiento por tipo de variable y estimador |
| Ajuste de hiperparámetros | `RandomizedSearchCV` con `TimeSeriesSplit` de 3 pliegues, sobre las filas más recientes (hasta 300,000) |
| Reentrenamiento | El mejor modelo de cada familia se reentrena con entrenamiento más validación y se evalúa en la prueba |
| Elección del campeón | Por error absoluto medio en **validación**; la prueba solo se reporta |
| Métricas | MAE, RMSE, WAPE y R², globales y en horas pico (7 a 9 y 17 a 19) |
| Referencias | La misma hora de la semana anterior (`lag_168`) y la misma hora del día anterior (`lag_24`) |

## Por qué estos algoritmos

- **Ridge:** referencia lineal rápida.
- **Regresión de Poisson:** modelo lineal generalizado con enlace logarítmico, pensado para conteos no negativos.
- **Gradient boosting con pérdida de Poisson (`hist_gbm_poisson`):** captura relaciones no lineales entre hora, día, estación y rezagos, y maneja millones de filas dentro de la memoria disponible.
- **Random Forest:** ensamble de árboles como segunda familia no lineal.

## Resultados (prueba, últimos 7 días; ordenados por MAE de validación)

| Modelo | Tipo | MAE val | MAE prueba | RMSE prueba | WAPE prueba | R² prueba | WAPE horas pico |
|---|---|---|---|---|---|---|---|
| `hist_gbm_poisson` | ml | 1.117 | 1.103 | 1.751 | 0.510 | 0.727 | 0.447 |
| `random_forest` | ml | 1.144 | 1.137 | 1.806 | 0.526 | 0.710 | 0.459 |
| `ridge` | ml | 1.215 | 1.233 | 1.935 | 0.571 | 0.667 | 0.502 |
| `baseline_estacional_168h` | baseline | 1.389 | 1.367 | 2.273 | 0.633 | 0.540 | 0.558 |
| `poisson` | ml | 1.492 | 1.498 | 4.346 | 0.693 | -0.681 | 0.665 |
| `baseline_ayer_24h` | baseline | 1.505 | 1.518 | 2.624 | 0.702 | 0.387 | 0.654 |

El mejor modelo es `hist_gbm_poisson`: su WAPE de prueba es 0.510, contra 0.633 de la mejor referencia (`lag_168`), una mejora del 19.3%; en horas pico el WAPE baja de 0.558 a 0.447. Es el mismo ganador tanto por validación como por prueba.

## Límites que conviene tener presentes

- **Ventana de entrenamiento.** Todas las corridas usaron los últimos 6 meses (`max_train_months = 6`), por lo que las corridas etiquetadas `completa` y `desde2022` usaron exactamente el mismo conjunto de entrenamiento y sus métricas son idénticas. La etiqueta solo indica de qué tabla se leyó. Para comparar ventanas hay que variar `--max-train-months`.
- **Comparación entre algoritmos.** Para acotar la memoria, el Random Forest se reentrena con las últimas 250,000 filas, mientras que el boosting usa todo el entrenamiento; la comparación no es del todo pareja.
- **Ajuste ligero.** La búsqueda aleatoria probó solo 3 o 4 combinaciones por modelo (40 ensayos anidados en total) sobre un máximo de 300,000 filas, así que es un ajuste inicial y no exhaustivo.
- **Regresión de Poisson.** Rinde peor que la referencia (R² negativo) y su RMSE (4.35) es mucho mayor que su MAE (1.50), señal de errores muy grandes en algunos casos; no se investigó la causa. Queda como resultado negativo documentado.
- **Semana de prueba.** La prueba son los últimos 7 días de la tabla usada en este entrenamiento; las cifras de las referencias no son comparables una a una con las de la Fase 1 si la tabla termina en otra fecha.
- **Seguimiento.** La base de MLflow guarda rutas absolutas de la máquina donde se entrenó, así que no se versiona; para repetir los experimentos se corre el comando de entrenamiento.
