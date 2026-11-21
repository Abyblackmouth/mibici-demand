# Auditoría de integración final — MiBici MLOps

Estado: **VERIFICADO LOCALMENTE**, rama `fase3`, Python 3.13.0. **161 pruebas aprobadas**, sin fallos, errores ni omisiones; 42.922 s. La preparación no equivale a publicación ni a aprobación productiva.

## Alcance y evidencia conservada

Se revisaron módulos, interfaces, rutas, dependencias, notebooks/documentación, manifiestos, resultados y pruebas de las tres fases. La evaluación histórica conserva su significado original: no se reescriben cifras ni se atribuye validación prospectiva al campeón antiguo. Se releen solo los CSV de agosto/septiembre, nunca octubre, con limpieza independiente y reconstrucción de etiquetas/rezagos.

`integracion/auditoria_comprobaciones.json` documenta 115 archivos previos de evidencia/modelos/ejemplos/datos con hashes sin cambios. `baseline_evidencias.json` conserva esos hashes; la CV corregida y otros cambios de código tienen hashes nuevos, no se simula que sean el código original de los reportes. Se mantuvieron las 140 pruebas anteriores; el único cambio en una prueba histórica es formato de `test_pipeline.py`, con **AST idéntico al original** comprobado.

## Flujo de extremo a extremo

| Etapa | Código/contrato | Comprobación y límite |
|---|---|---|
| Datos | `dataset.py`, manifiesto de Fase 1 | Solo dos meses necesarios; hashes verificados, lectura sin octubre |
| Limpieza | `TripCleaner`, reglas independientes de auditoría | Relectura de fuentes y reconstrucción de etiquetas; no se imputa demanda faltante como cero de toda una red sin registros |
| Variables | `features.py`, `temporal.py` | Orígenes diarios, horas cerradas y viajes terminados; cohorte/horario con pasado, sin catálogo actual |
| Entrenamiento | `build_model_pipeline`, snapshot v1 | Tres ajustes independientes, pesos/estadísticas solo train, seed 42 y parámetros fijos |
| MLflow | Tracking y nuevo Registry | Métricas originales contrastadas en lectura; candidato recargado y joblib exacto recuperado por SHA |
| Selección | `model_release.json` | Último corte causal por cronología, no argmin de test ni optimización de ventana |
| Serving | `api.py`, `release.py` | Misma identidad/tipos, carga una vez, origen mínimo, Swagger y validaciones 422 |
| Docker | Dockerfile, locks | Binario común, sin rutas Windows/volúmenes/datos/MLflow; usuario 10001 y filesystem solo lectura |
| Monitoreo | `monitoring/drift.py` | Mismo candidato y ventana posterior; originales del 7/9 y nuevos del 21/9 separados |

Desde crudo se comprobaron 196,000 etiquetas y 392,000 rezagos, con cero discrepancias. Se recalcularon las nueve combinaciones corte/predictor y las cuatro métricas de **20 escenarios** (10 previos + 10 del candidato), sin cambiar los reportes originales.

## Problemas encontrados y correcciones

1. **Modelos distintos en serving y monitoreo:** se introdujo un manifiesto y una copia inmutable del último artefacto temporal; el candidato es ahora común. El histórico se conserva en su ruta y una imagen separada para sus diez pruebas de integración.
2. **Tipos no uniformes:** el histórico usa estación int16/metadatos float64; el temporal se entrenó con estación int64 y rezagos float32. La API mantiene el contrato histórico al cargarlo y usa los tipos exactos del manifiesto para el candidato. No se convierten rezagos a float64 para suavizar el hallazgo anterior.
3. **Posibilidad de inferir hacia el pasado:** el candidato rechaza orígenes anteriores al 21 de septiembre. Los ejemplos nuevos son del 26, con rezagos de datos causales; no se reutiliza una petición del 12 contra pesos entrenados después.
4. **Sobrescrituras al repetir comandos:** backtesting y CLI de entrenamiento crean directorios únicos; los nuevos reportes de integración/monitoreo tienen salidas propias. No se sustituyen métricas previas al repetir verificaciones.
5. **CV legado por filas:** podía dividir estaciones de una misma hora. Se sustituyó por división sobre timestamps únicos y purga de 24 h, con pruebas nuevas. No se reoptimizó Fase 2 ni se reescribieron sus métricas; sus supuestos históricos de cohorte/disponibilidad siguen sin ser una validación prospectiva.
6. **Registro MLflow:** los intentos iniciales revelaron WinError 5 al limpiar código con atributos de OneDrive y formato Skops incompatible con este pipeline; se empaqueta un staging limpio con CloudPickle explícito. Se corrigió también convertir a Path la ruta devuelta por download_artifacts. Los runs fallidos se conservan; el run final FINISHED, su alias y el SHA recuperado se verificaron.
7. **Publicación/rutas obsoletas:** README con índice e instrucciones actuales; páginas de docs identificadas como legado; Python del Makefile y backend de construcción fijados. Git excluye ahora todos los pesos/snapshots y cachés; nada se borró como supuesto requisito para publicar.
8. **Imagen antigua por ID:** el ID del primer bloque Docker no estaba disponible para reetiquetarlo. Sus evidencias originales se conservan; se construyó y verificó otra imagen histórica con el mismo joblib y código actual, sin afirmar identidad binaria con aquella imagen ausente.

## Modelo común y criterio de liberación

Candidato: **mibici-temporal-20260921-v1**, `models/final/pipeline.joblib`. SHA-256 `e355fb092ca5c6d3f8498369001a803c9afc399b82bdd379ee61b902809af372`. Origen `models/fase3/20261119T195229_3cc516c1/pipeline_20260921.joblib`, run `063ac14e3a8d478c8366d9c72b2ac536`. Entrenamiento [2026-08-23 00:00:00, 2026-09-20 00:00:00) con 24 h de maduración; corte 2026-09-21 00:00:00. Es una copia de bytes, no un nuevo ajuste ni el modelo reentrenado de las pruebas limpias.

Se usa una política cronológica para unificar el paquete técnico, no una competición entre semanas. Las métricas de septiembre ya eran conocidas; no se presentan como prueba final ciega ni se usan para escoger hiperparámetros. Criterios explícitos de revisión: causalidad, identidad/contrato, comparación con ambos baselines, salidas finitas sin recortes y ausencia de fallos extremos en el procedimiento. La evidencia cumple integración y causalidad; **no cumple una demostración de robustez general** porque el primer corte empeora RMSE/R² y no existe holdout nuevo certificado. Estos son criterios de revisión de esta auditoría, no criterios retrospectivamente presentados como predeclarados antes del backtesting.

Por tanto, el modelo es **desplegable técnicamente para la demostración académica**, con `produccion_aprobada=false`. No se promueve a champion productivo ni se afirma superioridad al campeón de Fase 2 comparando métricas de periodos distintos. El dato de octubre no se consume; su condición de holdout virgen sigue sin poder certificarse por la categoría 10 aprendida en el histórico.

Registro nuevo: `mibici-fase3-candidato` v2, alias `candidate`, run `f130b2a5fa9c44f5a1a2ac8c004a7125`, en `mlflow_integracion.db`. Se recuperó el joblib exacto con su SHA y se recargó la envoltura sklearn de Registry para 65,520 filas: diferencia máxima 0. El champion y la base histórica no se modifican.

## Reproducibilidad real del entrenamiento

`references/data_snapshot.json` fija seis archivos, fuente, columnas/tipos, periodos, parámetros, seed 42, threads 4 y tolerancia absoluta/relativa de métricas 1e-6 fijada antes de ejecutar. No se hace búsqueda ni selección sobre las evaluaciones. Se vuelve a ajustar exactamente el builder de fit_fold con esos parámetros; se compara también con los resultados originales y se guardan modelos nuevos en directorios separados.

| Entorno limpio | Cortes ajustados | Máxima diferencia de predicción | Máxima diferencia de métricas |
|---|---:|---:|---:|
| windows, Python 3.13.0 | 3 | 0 | 0 |
| linux, Python 3.13.0 | 3 | 3.552713678800501e-15 | 0 |

Windows usó `.venv/repro-clean`, sin paquetes del entorno principal, con lock y pip check. Linux usó `Dockerfile.train`, base por digest y wheels fijadas, snapshot incluido y cero volúmenes. La diferencia residual de predicciones Linux está a escala de redondeo y muy por debajo de la tolerancia; las métricas resultaron idénticas. Los hashes de reserialización Linux pueden diferir por entorno/representación, por lo que no se promete igualdad de todos los binarios.

Esto **sí demuestra ajuste reproducible de Fase 3 desde snapshot**, además de la portabilidad del binario. No demuestra reproducción exacta de toda la búsqueda de seis meses de Fase 2: falta su snapshot original, periodos/ejecución completos y no se consumió octubre para reconstruirlo. La limpieza desde crudo se verifica separadamente con fuentes y reconstrucción independiente; la imagen de entrenamiento parte deliberadamente del snapshot preparado.

## Serving Windows y Linux ejecutado

Imagen final `mibici-demand:fase3`: `sha256:254752e18ea1c5145fdc78ca21dfa6e99a7b7db21e582f4334761820301db74f`. Build y run ejecutados; healthy. `/health`, `/predict`, `/docs` devuelven 200 en ambos entornos y las entradas inválidas 422. Inference nativa Linux, API Windows y API Docker tienen diferencia máxima **0** con Python directo; SHA del artefacto idéntico. Sin volúmenes, rutas Windows, venv, datos históricos o MLflow; UID 10001 y rootfs solo lectura.

| Estación | Predicción idéntica en los cuatro caminos |
|---|---:|
| 2 | 2.728705012827481 |
| 46 | 20.720247569805895 |
| 999 | 1.0691387843421751 |

Petición/Respuesta: `examples/predict_request_final.json`, `examples/predict_response_final.json`; prueba detallada en `integracion/inferencia_final.json`. Los tres valores comprueban transporte/precisión, no aceptación de calidad predictiva.

## Resultados y límites metodológicos

| Corte original | MAE modelo | RMSE modelo | WAPE modelo | R² modelo |
|---|---:|---:|---:|---:|
| 2026-09-07 | 1.0129 | 3.8942 | 0.6786 | −1.3431 |
| 2026-09-14 | 1.0843 | 1.9051 | 0.6338 | 0.5429 |
| 2026-09-21 | 1.0788 | 1.8094 | 0.5615 | 0.6492 |

La tabla completa con ambos baselines y las mismas observaciones está en `metricas_cortes.csv`; sus 36 métricas globales se contrastaron con SQLite original en modo lectura. Se mantiene el extremo 438.0117 de estación 46 y la sensibilidad a float32/float64: ninguna salida se limita para ocultarlo. El extremo pertenece al primer modelo temporal, no se adjudica al campeón ni al candidato actual.

Los diez escenarios originales conservan sus métricas. Los diez del candidato nuevo se ejecutaron sobre septiembre 21–27 en `integracion/drift_final/`. Su control reproduce MAE 1.0788/RMSE 1.8094/WAPE 0.5615/R² 0.6492. Por ejemplo, combinado 50% produce MAE 1.3811/RMSE 2.4496/WAPE 0.5975/R² 0.4920 y activa pérdida de desempeño, además de calidad/distribución. No se mezclan estos resultados con los anteriores para mejorar el relato.

**WAPE y composición:** WAPE = Σ|y−ŷ| / Σy; al aumentar horas pico también cambia Σy. Puede bajar aunque aumente el error medio absoluto. N constante no implica mezcla/denominador comparable. Las comparaciones se interpretan como riesgo en cada mezcla, y se descomponen errores usando las predicciones originales de las mismas filas fuente. R² también depende de la varianza de y.

**PSI y control:** el entrenamiento contiene agosto/septiembre y el monitoreo solo septiembre. El control original tiene PSI de mes 12.4618, aun con PSI contra sí mismo cero. Es un cambio de calendario esperado, dominado por categorías/proporciones y suavizado; no prueba degradación. PSI 0.10/0.25 y demás umbrales son ilustrativos, no calibración de falsas alarmas ni SLO aprobados. Se separan distribución, faltantes y desempeño.

**Faltantes:** se ocultan mediciones manteniendo demanda física/etiquetas; el pipeline imputa con train. API actual **rechaza** esas peticiones con 422; el experimento no demuestra aceptación operativa de NaN. No se alteran rezagos medidos con etiquetas incompatibles ni se ejecuta reentrenamiento automático.

**Partición:** 28 días ofrecen cuatro ciclos semanales y limitan descarga/cómputo, pero no se ha demostrado su suficiencia frente a seis meses. No se elige ventana con los resultados de evaluación. Se documenta como limitación y se propone comparación futura anidada temporal predeclarada, más un holdout no examinado. Fuentes mensuales solo simulan un feed diario; la aplicación real requiere disponibilidad operacional equivalente.

## Profesor, roles y rúbrica

La Fase 1 pidió cortes sucesivos: están ejecutados y reproducidos con información anterior a cada corte. Objetivo, destinatario y horizonte están en ML Canvas/README. La Fase 2 pidió justificar partición considerando distribución/variabilidad/sesgos: la auditoría conserva el resultado negativo, estudia composición/cobertura/historia desigual/extremos y reconoce que 28 días no constituyen un óptimo validado. Se añadió el índice navegable solicitado.

`matriz_rubrica_final.md` cubre seis criterios (15+15+20+20+15+15=100), requisitos obligatorios, evidencias y brechas, basándose en la transcripción del equipo pendiente de cotejo con Canvas. `integracion/roles_evidencias.md` separa responsabilidades funcionales de autoría Git comprobada. Falta validar asignaciones/aportes personales de Fase 3 y redactar una reflexión real del equipo; no se inventan colaboraciones.

## Publicación y recuperación

Inventario exacto propuesto: `integracion/archivos_github.txt` (205 rutas al ejecutar el inventario; regenerarlo tras cambios documentales). Fuentes, locks, manifiestos, notebooks y evidencia ligera se suben cuando se autorice. Pesos, snapshots, venv, DB/almacenes MLflow, credenciales, cachés y temporales permanecen excluidos. Se aplicó detector limitado de formatos de tokens/claves; no es una auditoría exhaustiva de seguridad.

Paquete separado `mibici-fase3-v1.zip` (16,793,052 bytes), SHA-256 `755b11dcec416efb6f5666af2e0406e8c67db31f9cadb1ef837481d161e2585b`, 14 miembros. Restauración íntegra y rechazo de sobrescrituras probados. `download_url=null`: no se inventa una ubicación publicada. Propuesta: asset versionado en GitHub Releases/almacenamiento autorizado, manifiesto junto a fuentes y recuperación por SHA; imagen en DockerHub con etiqueta/digest reales después de autorización.

## Comandos y comprobaciones

Los comandos completos están en README. Se ejecutaron prepare_release, builds de entrenamiento/serving/histórico, reproducción Windows/Linux, registro/recuperación MLflow, verify_final, simulación con plan final, pytest completo y auditoría independiente. La suite se lanzó desde `.venv/compatibility-check` para confinar temporales:

```powershell
& ..\Scripts\python.exe -m pytest ..\..\tests -q --rootdir ..\.. --basetemp integration-full-tests -o cache_dir=cache --junitxml=..\..\reports\fase3\integracion\pytest_final.xml
```

Ruff cubre explícitamente paquete, scripts y tests; no se usa la configuración de descubrimiento que omitía los dos últimos directorios. pip check pasó en entorno principal y limpio. Se registran cuatro advertencias existentes de dependencias (TestClient/httpx y MLflow/SQLAlchemy/Pandas/esquema entero); no se silenciaron ni se flexibilizaron criterios para aprobar. Se corrigió una comprobación de diff por línea vacía final en documentación. El estado final y códigos de salida están en `auditoria_comprobaciones.json`.

## Dictamen y pendientes

La integración local está comprobada y preparada para revisión/publicación autorizada. No se puede afirmar todavía una entrega de 100 puntos: faltan publicación GitHub/artefactos/DockerHub y verificación desde sus URLs, contribuciones individuales acreditadas, cotejo Canvas y PDF con reflexión, enlaces y evidencias contextualizadas. El candidato sigue sin aprobación productiva; ampliar la validación requiere un nuevo plan temporal y periodos no examinados. No hubo commits, push, merge, publicaciones ni generación de PDF.
