# Matriz de rúbrica — entrega final MiBici

Referencia: [transcripción de Canvas aportada por el equipo](../../references/rubrica_entrega_final.md). **Cotejo textual contra Canvas pendiente**. Esta matriz identifica evidencia y brechas; no asigna puntos ganados ni promete una calificación.

| Criterio | Máximo | Evidencia concreta | Cumplimiento verificable actual | Para aspirar a Excelente / brecha |
|---|---:|---|---|---|
| Acercamiento inicial al problema | 15 | `references/planteamiento_del_problema.md` §§1–7; notebooks EDA; `references/hallazgos_de_datos.md`; figuras Fase 1 | Base técnica/documental disponible: conocimientos previos, contexto y exploración. El interés del equipo es una valoración humana | Integrar al PDF el razonamiento inicial y cómo cambió con los datos; reflexión real del equipo. No inventar experiencia como usuarios ni beneficios operativos medidos |
| Análisis del problema | 15 | ML Canvas, diccionario, README, `temporal.py`, esquema API | Objetivo definido: conteo de viajes por estación/hora; destinatario: operación de redistribución; horizonte: 24 h por emisión diaria. Fuente/etiquetas/límites explicitados | Redactar el vínculo negocio→ML→decisión; distinguir demanda observada de necesidad no satisfecha, servicio académico de operación real y disponibilidad mensual de ingestión diaria |
| Actividades/tareas por rol | 20 | [roles y evidencias](integracion/roles_evidencias.md); commits `20f954b`, `ad512b7`, `b69bf27`, `ebeaa77`, `7dbccea`, `34e4350`, `62793b2`, `7288a2b`, `87878d2` | Funciones y colaboración documentadas; contribuciones históricas comprobables. Fase 3 local todavía carece de autoría individual acreditada | **Prioridad P0:** equipo confirma asignaciones reales y aporta evidencia de participación/revisión de todos. Relacionar actividades con entregables e interacción, sin adjudicar tareas por conveniencia. Riesgo que afecta al criterio de 20 puntos |
| Métodos y técnicas | 20 | `pipeline.py`, `train.py`, tres cortes, auditoría de 438, snapshots, locks, `reproduce.py`, API/Docker/drift; JUnit | Pipeline causal Fase 3, parámetros fijos, preprocesamiento en train, baseline común, reentrenamiento limpio Windows/Linux y serving del mismo SHA comprobados. CV legado corregido por timestamps | **Prioridad P0:** argumentar 28 días como limitación, no óptimo; comparar supuestos Fase 2/3; justificar Poisson, perfiles/PCA, métricas, PSI/umbrales y elección cronológica del candidato. No afirmar que un corte favorable certifica robustez ni que octubre es un holdout virgen. Riesgo que afecta al criterio de 20 puntos |
| Resultados | 15 | `metricas_cortes.csv`, `auditoria_backtesting.md`, diez escenarios originales + diez del candidato; `repro_*/resultado.json`, `inferencia_final.json`, `pytest_final.xml` | **161 pruebas aprobadas**, sin fallos/omisiones. Cifras históricas preservadas, extremos visibles, entrenamiento reproducido y predicciones de serving idénticas | Seleccionar tablas/gráficos representativos para el PDF, con periodos, N, unidades y artefacto por resultado. No mezclar histórico/candidato o WAPE de mezclas distintas como comparación causal. Incorporar enlaces reales de publicación cuando existan |
| Conclusiones y reflexión final | 15 | Auditorías temporal/integración/drift; lista de limitaciones; checklist | Hay hallazgos técnicos demostrables y recomendaciones, no un cierre personal del equipo ni PDF | **Prioridad P1:** redactar reflexión auténtica: qué aprendieron, qué decisiones conservaron/corrigieron, qué no demuestra la evidencia y plan futuro. Incluir límites de disponibilidad, ventana corta, extremos, umbrales y recuperación/publicación |
| Total posible | **100** | Evaluación oficial del profesor | No se estima una nota con el número de pruebas | La calidad documental y la atribución de roles no se deducen automáticamente del éxito técnico |

## Requisitos obligatorios

| N.º | Requisito | Estado y evidencia / pendiente |
|---:|---|---|
| 1 | Preparación/EDA/preprocesamiento | Comprobable: cleaning/features/notebooks/figuras y relectura de agosto/septiembre en auditoría |
| 2 | Versionado de datos | Snapshot v1 con SHA-256 y paquete local recuperable. **Pendiente alojarlo como artefacto accesible** con URL/hash antes de declarar reproducible un clon externo |
| 3 | Construcción/ajuste/evaluación | Fase 2 documentada; Fase 3 tres pipelines causales. Reproducción limpia Fase 3 realizada. Búsqueda exacta histórica Fase 2 no reproducida por snapshot original faltante |
| 4 | Cookiecutter | Estructura organizada conservada; módulos nuevos siguen el paquete |
| 5 | MLflow/DVC | Se usa MLflow: 36 métricas originales contrastadas con SQLite en lectura; nuevo run/modelo candidato. DVC no es necesario para duplicar la función ya cubierta |
| 6 | Comparación/Model Registry | Tres cortes contra dos baselines; registro `mibici-fase3-candidato`, alias `candidate`, SHA del joblib recuperado. No se altera champion histórico |
| 7 | Refactorización/buenas prácticas | Ruff, módulos, contratos, CLI con salidas aisladas y recuperación con hashes. No se borra código/evidencia por resultar desfavorable |
| 8 | pytest | Suite completa ejecutada: 161, 0 fallos, 0 errores, 0 omitidas; cuatro advertencias de dependencias registradas |
| 9 | FastAPI/Swagger | HTTP real /health, /predict y /docs 200; inválidos 422, corte mínimo y tipos preservados |
| 10 | Otro entorno | **Entrenamiento** de tres cortes en Windows limpio/Linux: métricas idénticas; además portabilidad del mismo joblib comprobada. Se distinguen ambas verificaciones |
| 11 | Docker y DockerHub | Build/run/healthy reales e imagen local integrada. **Publicación DockerHub pendiente**, expresamente no autorizada en este bloque |
| 12 | Drift | Control, composición, faltantes y progresión 10/30/50; etiquetas íntegras, alertas y acciones. Original y candidato se conservan por separado |

## Comentarios del profesor

- **Fase 1, cortes sucesivos:** 7/14/21 septiembre de 2026, entrenamiento independiente con pasado, semana posterior, etiquetas maduras, cohorte fija por origen, mismos casos para los baselines. Se atendió con ejecución, no solo texto. Reproducción limpia confirma las cifras, incluido el fallo del primer corte.
- **Fase 1, objetivo/destinatario/horizonte:** ML Canvas y README especifican conteo estación/hora, planeación de redistribución y emisión diaria a 24 h. No se confunde con un único pronóstico semanal ni con mejora operativa demostrada.
- **Fase 2, justificar partición/variabilidad/sesgos:** la auditoría documenta cuatro ciclos semanales, ahorro de descarga/cómputo, pero **no demuestra** que 28 días sean suficientes frente a seis meses. Se estudian distribución, cobertura, estaciones con historia desigual, meses y concentración de error. No se mantuvo la partición sin reconocer esos límites ni se optimizó usando las ventanas evaluadas. Sigue pendiente una comparación prospectiva/nested temporal predeclarada para concluir suficiencia.
- **Fase 2, índice README:** incorporado con enlaces a instalación, arquitectura, datos, entrenamiento, pruebas, serving, Docker y monitoreo.

## Riesgos de pérdida de puntos y prioridades

**P0 — cerrar antes de entregar:** asignación/contribuciones verificadas de todos los integrantes (roles, 20); argumentación de decisiones y límites sin sobreafirmaciones (métodos, 20); publicación autorizada GitHub/DockerHub y artefactos recuperables con enlaces reales. La rúbrica no permite calcular una deducción exacta sin evaluación del profesor.

**P1 — construir el PDF:** introducción y arquitectura; integración por fases; tablas y gráficos contextualizados; discusión de extremos, WAPE/mezcla, PSI/calendario y NaN/API; reflexión real del equipo y trabajo futuro. Añadir evidencia Swagger del equipo si se dispone de `Soporte_Swagger.pdf`: su existencia fue informada, pero no se encontró una copia local en este repositorio durante la auditoría.

**P2 — presentación y cierre:** cotejar transcripción en Canvas, verificar enlaces del PDF, nombres/roles y nombre `Entrega_Final_Equipo07.pdf`; comprobar la entrega efectiva en Canvas. No se genera ni envía ese PDF aquí.

Lo comprobable ahora está en código, hashes, logs y resultados. Lo personal/reflexivo, la asignación de trabajo, la publicación y el documento final deben completarse posteriormente por el equipo; no se sustituyen por pruebas automatizadas.
