# Fase 3: simulación ejecutada de Data Drift y desempeño

Estado: **EJECUTADO**. Fecha UTC: `2026-11-21T02:07:52.085184+00:00`. Rama `fase3`; Python 3.13.0. No se entrenó ni reemplazó ningún modelo.

## Datos y modelo fijo

Se eligió el **último corte cronológico disponible**, sin seleccionar la semana por sus resultados. El comportamiento anómalo de ese corte ya era conocido; este experimento no es una nueva validación ciega. No se usa `models/best_pipeline.joblib` porque su periodo de entrenamiento no acredita ausencia de información posterior.

Se carga `models/final/pipeline.joblib` del backtesting `20261119T195229_3cc516c1` (run MLflow de origen `063ac14e3a8d478c8366d9c72b2ac536`). Su SHA-256 es `e355fb092ca5c6d3f8498369001a803c9afc399b82bdd379ee61b902809af372`. Se conserva el mismo objeto, pesos, perfil de estación, PCA, imputador, escalador y codificadores en todos los escenarios. El hash del estado antes/después coincide: `17a63eca7ed7d08e0a2acff206e634de`.

| Conjunto | Periodo local semiabierto | Filas | Uso |
|---|---|---:|---|
| Referencia histórica | [2026-08-23 00:00:00, 2026-09-20 00:00:00) | 261,020 | Distribuciones de entrenamiento; no se publican métricas in-sample |
| Control etiquetado | [2026-09-21 00:00:00, 2026-09-28 00:00:00) | 65,520 | Métricas y base íntegra de simulación |

Zona `America/Mexico_City`; cohorte fija de 468 estaciones y 20 horas de servicio por día, determinada solo con el pasado del corte. El número de estaciones con historia puede variar dentro del entrenamiento, según la elegibilidad causal heredada; no se fuerza una referencia balanceada retrospectivamente. Las fechas son locales sin tz, conforme a los CSV originales.

Las fuentes son los CSV abiertos de MiBici de agosto/septiembre de 2026, limpiados anteriormente con `TripCleaner`. Se reutiliza `data/processed/fase3/viajes_limpios.parquet`, verificando su hash de la ejecución original, y `build_fold` reconstruye la referencia/control; se exige coincidencia exacta con el parquet de backtesting antes de simular. La procedencia completa y hashes están en `drift_final/ejecucion.json`. No se vuelve a descargar ni limpiar ni se lee octubre. El parquet original tiene datos hasta el 28 de septiembre, pero la lectura para esta simulación filtra comienzos anteriores al 28 de septiembre; su hash completo solo verifica integridad.

## Disponibilidad temporal y etiquetas

El modelo se ajustó antes del corte `2026-09-21 00:00:00` con 28 días, fin exclusivo `2026-09-20 00:00:00` y purga de 24 h. Último fin de viaje de las etiquetas de entrenamiento: `2026-09-20 00:36:42`. Se reutiliza `verify_composition` y la reconstrucción independiente `verify_labels_and_lags` sobre referencia y monitoreo: **cero discrepancias**. No se vuelve a llamar a `fit`.

Cada origen es la medianoche del día objetivo; `lag_24` y `lag_168` solo cuentan viajes terminados antes de ese origen y de horas cerradas. Calendario, cohorte y horario no se determinan con etiquetas futuras. No se usan `lag_1`, medias del día objetivo, metadatos actuales ni el campeón histórico. Cada escenario conserva los timestamps de origen/fuentes de sus registros. La selección de horas pico consulta solo `hora` y RNG, nunca demanda.

Las etiquetas finales se utilizan **solo después de observarlas** para medir desempeño offline. Se declara disponibilidad conservadora del lote completo desde `2026-09-29 00:00:00` (24 h después del fin exclusivo); máximo fin de viaje leído `2026-09-28 00:19:07`. Los CSV públicos mensuales no constituyen un feed diario real: la disponibilidad operativa se simula, como en el backtesting. Octubre sigue sin consumirse y se conserva la salvedad previa sobre su trazabilidad histórica.

Tipos originales de entrada: `estacion: int64`, `hora: int8`, `dia_semana: int8`, `mes: int8`, `es_fin_semana: int8`, `lag_24: float32`, `lag_168: float32`. Se verifica que el muestreo y NaN no cambien esos tipos; en particular los rezagos permanecen float32 y las predicciones float64.

## Plan fijado antes de puntuar

`references/plan_drift.json` fue guardado a `2026-11-21T01:25:16Z` antes de ejecutar esta simulación; SHA-256 `387c5b1c08d16af7ad5975ccf4fb657f9fd7671b58b316afea9934c720fa46e1`. Se copia a `drift_final/plan_predeclarado.json` antes de cargar datos. No se ajustaron umbrales ni intensidades tras observar resultados. El backtesting previo ya era conocido; esto no supone calibración prospectiva de alertas.

- **A. Control:** todas las observaciones, sin modificación. Se recalculan las predicciones y se comparan con las anteriores.
- **B. Composición:** se sustituye `floor(a*N_no_pico)` posiciones no pico por filas pico completas muestreadas con reemplazo. Horas pico heredadas: 7, 8, 9, 17, 18 y 19. La fracción pico resultante es `p + floor(a*N_no_pico)/N`; no se confunde la intensidad con el porcentaje final de horas pico.
- **C. Faltantes:** se ocultan **ambos** rezagos en `floor(a*N)` posiciones elegidas sin consultar X o y (MCAR en la población de solicitudes). Simula una interrupción conjunta de disponibilidad del feature store; el fenómeno físico y su demanda no se alteran. Solo se sustituye el valor recibido por NaN.
- **D. Progresivo:** B y C se ejecutan por separado al 10%, 30% y 50%; además se combina composición seguida de faltantes a esas tres intensidades. El combinado no identifica por sí solo qué mecanismo causa el cambio.

Semilla 42 para composición y 43 para faltantes, con `numpy.random.default_rng`; sustituciones/máscaras anidadas entre intensidades. Se mantiene el tamaño del lote. `fila_fuente` enlaza cada fila con el control original y `posicion_lote` identifica cada solicitud simulada. Se comprueba igualdad exacta de todas las columnas con su fila fuente, permitiendo únicamente NaN en rezagos en C/D. El muestreo tiene duplicados: no crea viajes nuevos ni nuevas observaciones independientes. Es una mezcla de solicitudes, no una nueva serie de viajes; no se recalculan rezagos a partir del lote remuestreado.

## Detectores y umbrales

PSI = Σ(qᵢ−pᵢ)·ln(qᵢ/pᵢ). Para rezagos/demanda se fijan cuantiles de la referencia (deciles, eliminando cortes repetidos), límites de soporte y colas. Para calendario/estación se fijan categorías históricas y una categoría desconocida. No se redefine el binning con monitoreo, ni siquiera al comparar un escenario con control. Se suma 1e-6 a cada probabilidad y se renormaliza para manejar bins vacíos. `bins_referencia.json` y `distribuciones_bins.csv` permiten auditar los cálculos.

El PSI es **condicional a valores observados**: NaN se mide aparte para no llamar cambio de demanda a un fallo de captura. Se informa PSI contra entrenamiento y contra control para separar cambio natural de la semana y perturbación añadida. Se complementa con distancia TV discretizada y distancia empírica KS para numéricas; no se usan p-valores ni se afirma significación, porque hay conteos discretos, dependencia estación/tiempo y duplicados. Las hipótesis de la [prueba KS de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) no se cumplen como prueba IID en este diseño. La literatura primaria [Haas y Sibbald, 2024](https://journals.sagepub.com/doi/10.3233/DS-240059) también advierte que los puntos de corte de estabilidad deben depender del caso de uso.

| Señal | Vigilancia / alerta predeclarada | Acción propuesta |
|---|---|---|
| Distribución de entrada | PSI ≥0.10 / ≥0.25 | Revisar calendario y mezcla; segmentar y monitorizar; no inferir degradación por PSI |
| Calidad de rezagos | Faltantes ≥1% / ≥5% | Investigar captura/feature store; corregir causa y recuperar valores |
| Pérdida MAE | Aumento ≥10% y ≥0.05 viajes | Revisar errores/segmentos y causas de datos |
| Pérdida RMSE | Aumento ≥10% y ≥0.10 viajes | Investigar errores extremos sin suprimirlos |
| Pérdida WAPE | Aumento ≥10% y ≥0.02 (2 puntos porcentuales) | Revisar precisión relativa al volumen |
| Pérdida R² | Descenso ≥0.10 unidades | Revisar junto con errores y varianza de y |

Son reglas **ilustrativas de ingeniería**, sin calibración de falsos positivos ni aprobación operativa, no conclusiones estadísticas ni SLO. Los mínimos absolutos evitan alertar solo por cambios diminutos relativos. Si el control tiene error cero, se usa únicamente el mínimo absoluto (el cambio relativo no está definido). La demanda se compara como diagnóstico diferido; no activa una alerta de drift de entradas. Una alerta solo propone investigar/monitorizar/revisar el modelo y evaluar un eventual reentrenamiento con nueva validación temporal; aquí no se ejecuta ninguno ni se cambia el artefacto desplegado.

## Resultados ejecutados

Diferencia máxima control frente al backtesting original: **0**, tolerancia 1e-12. Se mantienen predicciones extremas y todas las filas; no se recorta, elimina ni ajusta ninguna para favorecer resultados. MAE/RMSE en viajes/estación/hora, WAPE como proporción, R² puede ser negativo.

| Escenario | N únicos fuente / N lote | Pico | Faltantes por lag | MAE | RMSE | WAPE | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 65,520 / 65,520 | 30.00% | 0% | 1.0788 | 1.8094 | 0.5615 | 0.6492 |
| composicion_10 | 60,934 / 65,520 | 37.00% | 0% | 1.1108 | 1.8587 | 0.5551 | 0.6503 |
| composicion_30 | 51,761 / 65,520 | 51.00% | 0% | 1.1693 | 1.9400 | 0.5417 | 0.6561 |
| composicion_50 | 42,588 / 65,520 | 65.00% | 0% | 1.2303 | 2.0200 | 0.5323 | 0.6546 |
| faltantes_10 | 65,520 / 65,520 | 30.00% | 10% | 1.1010 | 1.8799 | 0.5731 | 0.6213 |
| faltantes_30 | 65,520 / 65,520 | 30.00% | 30% | 1.1524 | 2.0479 | 0.5999 | 0.5506 |
| faltantes_50 | 65,520 / 65,520 | 30.00% | 50% | 1.2004 | 2.1789 | 0.6249 | 0.4912 |
| combinado_10 | 60,934 / 65,520 | 37.00% | 10% | 1.1339 | 1.9309 | 0.5667 | 0.6226 |
| combinado_30 | 51,761 / 65,520 | 51.00% | 30% | 1.2558 | 2.2111 | 0.5817 | 0.5533 |
| combinado_50 | 42,588 / 65,520 | 65.00% | 50% | 1.3811 | 2.4496 | 0.5975 | 0.4920 |

| Escenario | Δ MAE | Δ RMSE | Δ WAPE (pp) | Δ R² | Δ MAE relativo | Δ RMSE relativo |
|---|---:|---:|---:|---:|---:|---:|
| control | +0.0000 | +0.0000 | +0.00 | +0.0000 | +0.00% | +0.00% |
| composicion_10 | +0.0320 | +0.0493 | -0.64 | +0.0011 | +2.97% | +2.73% |
| composicion_30 | +0.0905 | +0.1306 | -1.98 | +0.0069 | +8.39% | +7.22% |
| composicion_50 | +0.1515 | +0.2106 | -2.92 | +0.0054 | +14.05% | +11.64% |
| faltantes_10 | +0.0222 | +0.0705 | +1.16 | -0.0279 | +2.06% | +3.89% |
| faltantes_30 | +0.0736 | +0.2385 | +3.83 | -0.0986 | +6.83% | +13.18% |
| faltantes_50 | +0.1216 | +0.3695 | +6.33 | -0.1579 | +11.28% | +20.42% |
| combinado_10 | +0.0551 | +0.1215 | +0.51 | -0.0266 | +5.11% | +6.72% |
| combinado_30 | +0.1770 | +0.4017 | +2.02 | -0.0959 | +16.40% | +22.20% |
| combinado_50 | +0.3023 | +0.6402 | +3.60 | -0.1572 | +28.02% | +35.38% |

Δ = escenario − control. `metricas_escenarios.csv` conserva valores completos, diferencia con signo, diferencia absoluta (magnitud) y relativa de las cuatro métricas; el denominador relativo es |control|, y cero/indefinidos se dejan NA. El porcentaje de cambio de un R² negativo no es una mejora porcentual interpretable; la alerta usa caída en unidades. WAPE/R² se dejan NA si falta denominador/varianza.

## Distribuciones, calidad y alertas observadas

| Escenario | Drift vs historia | Drift añadido vs control | Calidad | Pérdida de desempeño | Métricas que activan pérdida |
|---|---|---|---|---|---|
| control | sí | no | no | no | ninguna |
| composicion_10 | sí | no | no | no | ninguna |
| composicion_30 | sí | sí | no | no | ninguna |
| composicion_50 | sí | sí | no | sí | mae, rmse |
| faltantes_10 | sí | no | sí | no | ninguna |
| faltantes_30 | sí | no | sí | sí | rmse |
| faltantes_50 | sí | no | sí | sí | mae, rmse, wape, r2 |
| combinado_10 | sí | no | sí | no | ninguna |
| combinado_30 | sí | sí | sí | sí | mae, rmse |
| combinado_50 | sí | sí | sí | sí | mae, rmse, r2 |

Resumen PSI/KS de entradas y máximos (valores completos por variable/comparación en `detectores_drift.csv`):

| Escenario | PSI máximo vs historia (entrada) | PSI máximo vs control (entrada) | KS lag_24 vs control | KS lag_168 vs control | Predicción máxima | SSE del 1% peor |
|---|---:|---:|---:|---:|---:|---:|
| control | 4.1994 | 0.0000 | 0.0000 | 0.0000 | 60.1095 | 34.66% |
| composicion_10 | 4.1994 | 0.0221 | 0.0123 | 0.0097 | 60.1095 | 34.46% |
| composicion_30 | 4.1994 | 0.1864 | 0.0365 | 0.0297 | 60.1095 | 34.13% |
| composicion_50 | 4.1994 | 0.5134 | 0.0610 | 0.0490 | 39.7489 | 33.29% |
| faltantes_10 | 4.1994 | 0.0000 | 0.0007 | 0.0010 | 60.1095 | 36.79% |
| faltantes_30 | 4.1994 | 0.0000 | 0.0010 | 0.0008 | 60.1095 | 40.99% |
| faltantes_50 | 4.1994 | 0.0001 | 0.0032 | 0.0016 | 60.1095 | 42.54% |
| combinado_10 | 4.1994 | 0.0221 | 0.0129 | 0.0103 | 60.1095 | 36.53% |
| combinado_30 | 4.1994 | 0.1864 | 0.0361 | 0.0286 | 60.1095 | 40.21% |
| combinado_50 | 4.1994 | 0.5134 | 0.0610 | 0.0463 | 38.9274 | 39.72% |

El control puede activar drift histórico, por ejemplo por el cambio calendario agosto→septiembre, aunque contra sí mismo tiene PSI cero y ninguna pérdida respecto a sí mismo. Esa alerta natural no significa automáticamente avería ni degradación del modelo. Las alertas de `alertas_acciones.csv` separan tipo, comparación, severidad, valor y acción.

- **composicion, 50%:** MAE cambia +14.05%, RMSE +11.64%, WAPE -2.92 pp y R² +0.0054. Pérdida según reglas: sí. No se presupone monotonicidad; se conservan todas las intensidades.
- **faltantes, 50%:** MAE cambia +11.28%, RMSE +20.42%, WAPE +6.33 pp y R² -0.1579. Pérdida según reglas: sí. No se presupone monotonicidad; se conservan todas las intensidades.
- **combinado, 50%:** MAE cambia +28.02%, RMSE +35.38%, WAPE +3.60 pp y R² -0.1572. Pérdida según reglas: sí. No se presupone monotonicidad; se conservan todas las intensidades.

El primer corte anterior mostró RMSE elevado y R² negativo por extremos (estación 46); el control de este candidato es una semana distinta y conserva sus propios resultados. Si faltantes/remuestreo reducen un extremo, RMSE puede mejorar mientras MAE o WAPE empeoran; eso no convierte datos faltantes en una solución. `errores_mayores.csv` y `sse_top1_proporcion` permiten examinarlo sin eliminar filas. El cambio de composición también altera la distribución de y y su varianza, por lo que R²/WAPE entre mezclas tienen denominadores distintos. Son riesgos sobre cada mezcla de solicitudes, no estimaciones causales del cambio físico de demanda. El fallo de captura sí conserva el mismo y por posición y permite contraste pareado de errores.

Para separar los mecanismos se reutiliza la predicción control de **cada fila fuente** en la mezcla simulada, antes de ocultar rezagos. Todas las filas sin fallo deben conservar esa predicción (tolerancia 1e-12). `descomposicion_errores.csv` descompone el cambio total en efecto mecánico de mezcla y efecto adicional de pérdida de entradas, en MAE y MSE; no se suman diferencias de RMSE porque la raíz no es aditiva. Es una descomposición del experimento, sin inferir efectos sobre demanda física.

| Escenario | Δ MAE por mezcla | Δ MAE adicional por faltantes | Δ MSE por mezcla | Δ MSE adicional por faltantes |
|---|---:|---:|---:|---:|
| control | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| composicion_10 | +0.0320 | +0.0000 | +0.1809 | +0.0000 |
| composicion_30 | +0.0905 | +0.0000 | +0.4898 | +0.0000 |
| composicion_50 | +0.1515 | +0.0000 | +0.8065 | +0.0000 |
| faltantes_10 | +0.0000 | +0.0222 | +0.0000 | +0.2600 |
| faltantes_30 | +0.0000 | +0.0736 | +0.0000 | +0.9198 |
| faltantes_50 | +0.0000 | +0.1216 | +0.0000 | +1.4737 |
| combinado_10 | +0.0320 | +0.0230 | +0.1809 | +0.2736 |
| combinado_30 | +0.0905 | +0.0864 | +0.4898 | +1.1254 |
| combinado_50 | +0.1515 | +0.1508 | +0.8065 | +1.9202 |

![Distribuciones](drift_final/distribuciones_comparativas.png)

![Desempeño según intensidad](drift_final/desempeno_intensidad.png)

![Detectores PSI](drift_final/detectores_psi.png)

![Calidad y colas de predicción](drift_final/calidad_y_predicciones.png)

## Reproducción y pruebas

Desde PowerShell en la raíz, rama fase3; usa exclusivamente el entorno existente:

```powershell
& .\.venv\Scripts\python.exe -m mibici_demand.monitoring.drift --threads 4 --plan references/plan_drift_final.json --modelo models/final/pipeline.joblib --salida reports/fase3/integracion/drift_final
& .\.venv\Scripts\python.exe -m pytest tests -q --junitxml=reports/fase3/drift/pytest_drift.xml
& .\.venv\Scripts\python.exe -m mibici_demand.monitoring.drift --actualizar-reporte
```

No se instalan dependencias nuevas: NumPy/Pandas/SciPy/Matplotlib/joblib ya están en `.venv`; sus versiones y hashes de código/plan constan en `ejecucion.json`. No requiere un servidor MLflow ni registra/modifica sus corridas anteriores. Para ejecutar también las diez pruebas Docker, mantener Docker Desktop Linux activo y la imagen `mibici-demand:fase3` construida previamente. Las salidas nuevas de este bloque se escriben solo en `reports/fase3/drift/` y este informe; README/documentación y código nuevo se gestionan aparte.

La simulación está ejecutada; la suite de este bloque todavía no se ha registrado. No se declara aprobada antes de ejecutarla.

Se verificaron hashes de 96 artefactos/resultados previos antes y después: todos sin cambios. El artefacto histórico desplegado tampoco se modificó. `predicciones_escenarios.parquet` conserva trazabilidad de cada solicitud simulada, sus etiquetas y predicciones originales.

## Limitaciones y recomendaciones

- Una semana, una semilla y perturbaciones controladas; no hay replicación que estime incertidumbre, potencia o falsas alarmas. La dependencia temporal y los duplicados impiden tratar N como muestras IID. Es una demostración offline reproducible, no validación productiva ni prueba de concept drift P(y|X).
- La referencia de 28 días se hereda sin optimizarla; su suficiencia frente a seis meses sigue sin demostrar. Mes y cambios de cohorte/historia pueden dominar PSI; se deben monitorizar segmentos y calendarios comparables en una futura calibración anterior al periodo de aceptación.
- Las métricas comparan etiquetas completas y no se omiten casos difíciles. En composición la mezcla objetivo cambia; en faltantes se mantiene la misma demanda. No se incrementan rezagos ni demandas con etiquetas incompatibles ni se modifica calendario sin su timestamp.
- **Contrato API:** FastAPI actual rechaza rezagos faltantes/NaN con 422. Este experimento llama directamente al pipeline para estudiar la tolerancia del imputador; no demuestra que esas solicitudes sean aceptadas en producción. En operación deben vigilarse también rechazos, cobertura y latencia, sin abrir silenciosamente el esquema.
- No hay ingestión en vivo, scheduler, alarmas externas, reentrenamiento automático ni sustitución del modelo. Calidad requiere corregir el pipeline de datos; drift sin pérdida requiere investigar/monitorizar; pérdida persistente con datos correctos justifica revisar el modelo y planear reentrenamiento con nueva validación temporal y holdout verdaderamente no examinado.
- No se usan viajes de octubre, no se modifican backtesting/API/Docker y no se publica nada. No se prepara el PDF final. Las alertas y mejoras aparentes no rehabilitan el extremo previamente auditado ni acreditan fiabilidad operativa.
