"""Figuras y reporte derivados exclusivamente de resultados ejecutados de drift."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fmt(value, digits: int = 4) -> str:
    return "NA" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def make_plots(tables: dict, reference: pd.DataFrame, frames: list[pd.DataFrame], out: Path):
    metrics = tables["metricas_escenarios"]
    selected = {
        f["escenario"].iloc[0]: f
        for f in frames
        if f["escenario"].iloc[0] in {"control", "composicion_50", "faltantes_50", "combinado_50"}
    }
    distributions = {"referencia histórica": reference, **selected}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for label, frame in distributions.items():
        hours = frame["hora"].value_counts(normalize=True).sort_index()
        axes[0, 0].plot(hours.index, hours, marker=".", label=label)
        for ax, name in zip(
            (axes[0, 1], axes[1, 0], axes[1, 1]), ("lag_24", "lag_168", "demanda")
        ):
            array = np.sort(frame[name].dropna().to_numpy())
            positions = np.unique(
                np.r_[np.linspace(0, len(array) - 1, 600).astype(int), len(array) - 1]
            )
            ax.plot(array[positions], (positions + 1) / len(array), label=label)
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel(name + " (valores observados; escala symlog)")
            ax.set_ylabel("Distribución acumulada")
    axes[0, 0].set(xlabel="Hora", ylabel="Proporción del lote", title="Composición temporal")
    axes[0, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Referencia y escenarios al 50%: etiquetas conservadas; NaN fuera de las CDF")
    fig.savefig(out / "distribuciones_comparativas.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    control = metrics.iloc[0]
    for ax, metric in zip(axes.flat, ("mae", "rmse", "wape", "r2")):
        factor = 100 if metric == "wape" else 1
        for family, rows in metrics[metrics["familia"] != "control"].groupby(
            "familia", sort=False
        ):
            ax.plot(
                [0, *(rows["intensidad"] * 100)],
                [control[metric] * factor, *(rows[metric] * factor)],
                marker="o",
                label=family,
            )
        ax.axhline(control[metric] * factor, color="black", ls="--", label="control")
        ax.set(xlabel="Intensidad (%)", ylabel="WAPE (%)" if metric == "wape" else metric.upper())
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Cambio de desempeño: mismo modelo, sin omitir extremos ni recortar predicciones")
    fig.savefig(out / "desempeno_intensidad.png", dpi=160)
    plt.close(fig)

    detectors = tables["detectores_drift"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    maximum = detectors["psi"].max()
    for ax, comparison in zip(axes, ("historica", "control")):
        pivot = detectors[detectors["comparacion"] == comparison].pivot(
            index="escenario", columns="variable", values="psi"
        )
        pivot = pivot.reindex(metrics["escenario"])
        im = ax.imshow(pivot, aspect="auto", cmap="YlOrRd", vmin=0, vmax=maximum)
        ax.set(
            xticks=range(len(pivot.columns)),
            xticklabels=pivot.columns,
            yticks=range(len(pivot)),
            yticklabels=pivot.index,
            title=f"PSI contra {comparison}",
        )
        ax.tick_params(axis="x", rotation=45)
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                ax.text(
                    j,
                    i,
                    _fmt(value, 2),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > maximum * 0.6 else "black",
                )
    fig.colorbar(im, ax=axes, label="PSI condicional a valores observados")
    fig.suptitle("PSI por variable: demanda es diagnóstico con etiqueta diferida, no entrada")
    fig.savefig(out / "detectores_psi.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    names = metrics["escenario"].tolist()
    axes[0].barh(
        names, metrics["faltantes_lag_24"] * 100, label="lag_24 y lag_168 (misma máscara)"
    )
    axes[0].axvline(5, ls="--", color="red", label="alerta 5%")
    axes[0].set(xlabel="Faltantes (%)", title="Calidad: ocultación conjunta de rezagos")
    axes[0].legend(fontsize=7)
    for name, frame in selected.items():
        values = np.sort(frame["prediccion"].to_numpy())
        positions = np.unique(
            np.r_[
                np.linspace(0, len(values) - 1, 600).astype(int),
                np.arange(len(values) - 40, len(values)),
            ]
        )
        axes[1].plot(values[positions], (len(values) - positions) / len(values), label=name)
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Predicción (symlog)",
        ylabel="Fracción de observaciones en la cola (log)",
        title="Predicciones originales, sin límites máximos",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.savefig(out / "calidad_y_predicciones.png", dpi=160)
    plt.close(fig)


def write_report(out: Path, tables: dict, result: dict):
    metrics = tables["metricas_escenarios"]
    detectors = tables["detectores_drift"]
    alerts = tables["alertas_acciones"]
    source, plan = result["procedencia"], result["plan"]
    p = plan["umbrales"]
    lines = [
        "# Fase 3: simulación ejecutada de Data Drift y desempeño",
        "",
        f"Estado: **{result['estado'].upper()}**. Fecha UTC: `{result['ejecutado_utc']}`. Rama `{result['rama']}`; Python {result['python']}. No se entrenó ni reemplazó ningún modelo.",
        "",
        "## Datos y modelo fijo",
        "",
        "Se eligió el **primer corte cronológico**, sin seleccionar la semana por sus resultados. El comportamiento anómalo de ese corte ya era conocido; este experimento no es una nueva validación ciega. No se usa `models/best_pipeline.joblib` porque su periodo de entrenamiento no acredita ausencia de información posterior.",
        "",
        f"Se carga `{source['modelo']}` del backtesting `{source['ejecucion_backtest']}` (run MLflow de origen `{source['mlflow_run_id_origen']}`). Su SHA-256 es `{source['modelo_sha256']}`. Se conserva el mismo objeto, pesos, perfil de estación, PCA, imputador, escalador y codificadores en todos los escenarios. El hash del estado antes/después coincide: `{result['estado_modelo_antes_despues']}`.",
        "",
        "| Conjunto | Periodo local semiabierto | Filas | Uso |",
        "|---|---|---:|---|",
        f"| Referencia histórica | [{source['train_inicio']}, {source['train_fin_exclusivo']}) | {source['referencia_filas']:,} | Distribuciones de entrenamiento; no se publican métricas in-sample |",
        f"| Control etiquetado | [{source['monitoreo_inicio']}, {source['monitoreo_fin_exclusivo']}) | {source['monitoreo_filas']:,} | Métricas y base íntegra de simulación |",
        "",
        f"Zona `America/Mexico_City`; cohorte fija de {source['estaciones']} estaciones y {len(source['horas_servicio'])} horas de servicio por día, determinada solo con el pasado del corte. El número de estaciones con historia puede variar dentro del entrenamiento, según la elegibilidad causal heredada; no se fuerza una referencia balanceada retrospectivamente. Las fechas son locales sin tz, conforme a los CSV originales.",
        "",
        "Las fuentes son los CSV abiertos de MiBici de agosto/septiembre de 2026, limpiados anteriormente con `TripCleaner`. Se reutiliza `data/processed/fase3/viajes_limpios.parquet`, verificando su hash de la ejecución original, y `build_fold` reconstruye la referencia/control; se exige coincidencia exacta con el parquet de backtesting antes de simular. La procedencia completa y hashes están en `drift/ejecucion.json`. No se vuelve a descargar ni limpiar ni se lee octubre. El parquet original tiene datos hasta el 28 de septiembre, pero la lectura para esta simulación filtra comienzos anteriores al 14; su hash completo solo verifica integridad.",
        "",
        "## Disponibilidad temporal y etiquetas",
        "",
        f"El modelo se ajustó antes del corte `{source['corte']}` con 28 días, fin exclusivo `{source['train_fin_exclusivo']}` y purga de 24 h. Último fin de viaje de las etiquetas de entrenamiento: `{source['max_fin_viaje_etiquetas_train']}`. Se reutiliza `verify_composition` y la reconstrucción independiente `verify_labels_and_lags` sobre referencia y monitoreo: **cero discrepancias**. No se vuelve a llamar a `fit`.",
        "",
        "Cada origen es la medianoche del día objetivo; `lag_24` y `lag_168` solo cuentan viajes terminados antes de ese origen y de horas cerradas. Calendario, cohorte y horario no se determinan con etiquetas futuras. No se usan `lag_1`, medias del día objetivo, metadatos actuales ni el campeón histórico. Cada escenario conserva los timestamps de origen/fuentes de sus registros. La selección de horas pico consulta solo `hora` y RNG, nunca demanda.",
        "",
        f"Las etiquetas finales se utilizan **solo después de observarlas** para medir desempeño offline. Se declara disponibilidad conservadora del lote completo desde `{source['evaluacion_etiquetada_disponible_desde']}` (24 h después del fin exclusivo); máximo fin de viaje leído `{source['max_fin_viaje_leido']}`. Los CSV públicos mensuales no constituyen un feed diario real: la disponibilidad operativa se simula, como en el backtesting. Octubre sigue sin consumirse y se conserva la salvedad previa sobre su trazabilidad histórica.",
        "",
        "Tipos originales de entrada: "
        + ", ".join(f"`{k}: {v}`" for k, v in source["dtypes_entrada"].items())
        + ". Se verifica que el muestreo y NaN no cambien esos tipos; en particular los rezagos permanecen float32 y las predicciones float64.",
        "",
        "## Plan fijado antes de puntuar",
        "",
        f"`references/plan_drift.json` fue guardado a `{plan['predeclarado_utc']}` antes de ejecutar esta simulación; SHA-256 `{result['plan_sha256']}`. Se copia a `drift/plan_predeclarado.json` antes de cargar datos. No se ajustaron umbrales ni intensidades tras observar resultados. El backtesting previo ya era conocido; esto no supone calibración prospectiva de alertas.",
        "",
        "- **A. Control:** todas las observaciones, sin modificación. Se recalculan las predicciones y se comparan con las anteriores.",
        "- **B. Composición:** se sustituye `floor(a*N_no_pico)` posiciones no pico por filas pico completas muestreadas con reemplazo. Horas pico heredadas: 7, 8, 9, 17, 18 y 19. La fracción pico resultante es `p + floor(a*N_no_pico)/N`; no se confunde la intensidad con el porcentaje final de horas pico.",
        "- **C. Faltantes:** se ocultan **ambos** rezagos en `floor(a*N)` posiciones elegidas sin consultar X o y (MCAR en la población de solicitudes). Simula una interrupción conjunta de disponibilidad del feature store; el fenómeno físico y su demanda no se alteran. Solo se sustituye el valor recibido por NaN.",
        "- **D. Progresivo:** B y C se ejecutan por separado al 10%, 30% y 50%; además se combina composición seguida de faltantes a esas tres intensidades. El combinado no identifica por sí solo qué mecanismo causa el cambio.",
        "",
        f"Semilla {plan['semilla']} para composición y {plan['semilla'] + 1} para faltantes, con `numpy.random.default_rng`; sustituciones/máscaras anidadas entre intensidades. Se mantiene el tamaño del lote. `fila_fuente` enlaza cada fila con el control original y `posicion_lote` identifica cada solicitud simulada. Se comprueba igualdad exacta de todas las columnas con su fila fuente, permitiendo únicamente NaN en rezagos en C/D. El muestreo tiene duplicados: no crea viajes nuevos ni nuevas observaciones independientes. Es una mezcla de solicitudes, no una nueva serie de viajes; no se recalculan rezagos a partir del lote remuestreado.",
        "",
        "## Detectores y umbrales",
        "",
        "PSI = Σ(qᵢ−pᵢ)·ln(qᵢ/pᵢ). Para rezagos/demanda se fijan cuantiles de la referencia (deciles, eliminando cortes repetidos), límites de soporte y colas. Para calendario/estación se fijan categorías históricas y una categoría desconocida. No se redefine el binning con monitoreo, ni siquiera al comparar un escenario con control. Se suma 1e-6 a cada probabilidad y se renormaliza para manejar bins vacíos. `bins_referencia.json` y `distribuciones_bins.csv` permiten auditar los cálculos.",
        "",
        "El PSI es **condicional a valores observados**: NaN se mide aparte para no llamar cambio de demanda a un fallo de captura. Se informa PSI contra entrenamiento y contra control para separar cambio natural de la semana y perturbación añadida. Se complementa con distancia TV discretizada y distancia empírica KS para numéricas; no se usan p-valores ni se afirma significación, porque hay conteos discretos, dependencia estación/tiempo y duplicados. Las hipótesis de la [prueba KS de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) no se cumplen como prueba IID en este diseño. La literatura primaria [Haas y Sibbald, 2024](https://journals.sagepub.com/doi/10.3233/DS-240059) también advierte que los puntos de corte de estabilidad deben depender del caso de uso.",
        "",
        "| Señal | Vigilancia / alerta predeclarada | Acción propuesta |",
        "|---|---|---|",
        f"| Distribución de entrada | PSI ≥{p['psi_vigilancia']:.2f} / ≥{p['psi_alerta']:.2f} | Revisar calendario y mezcla; segmentar y monitorizar; no inferir degradación por PSI |",
        f"| Calidad de rezagos | Faltantes ≥{p['faltantes_vigilancia']:.0%} / ≥{p['faltantes_alerta']:.0%} | Investigar captura/feature store; corregir causa y recuperar valores |",
        f"| Pérdida MAE | Aumento ≥{p['degradacion_relativa']:.0%} y ≥{p['mae_delta_min']:.2f} viajes | Revisar errores/segmentos y causas de datos |",
        f"| Pérdida RMSE | Aumento ≥{p['degradacion_relativa']:.0%} y ≥{p['rmse_delta_min']:.2f} viajes | Investigar errores extremos sin suprimirlos |",
        f"| Pérdida WAPE | Aumento ≥{p['degradacion_relativa']:.0%} y ≥{p['wape_delta_min']:.2f} ({100 * p['wape_delta_min']:.0f} puntos porcentuales) | Revisar precisión relativa al volumen |",
        f"| Pérdida R² | Descenso ≥{p['r2_caida_min']:.2f} unidades | Revisar junto con errores y varianza de y |",
        "",
        "Son reglas **ilustrativas de ingeniería**, sin calibración de falsos positivos ni aprobación operativa, no conclusiones estadísticas ni SLO. Los mínimos absolutos evitan alertar solo por cambios diminutos relativos. Si el control tiene error cero, se usa únicamente el mínimo absoluto (el cambio relativo no está definido). La demanda se compara como diagnóstico diferido; no activa una alerta de drift de entradas. Una alerta solo propone investigar/monitorizar/revisar el modelo y evaluar un eventual reentrenamiento con nueva validación temporal; aquí no se ejecuta ninguno ni se cambia el artefacto desplegado.",
        "",
        "## Resultados ejecutados",
        "",
        f"Diferencia máxima control frente al backtesting original: **{result['control_max_diferencia_backtest']:.12g}**, tolerancia 1e-12. Se mantienen predicciones extremas y todas las filas; no se recorta, elimina ni ajusta ninguna para favorecer resultados. MAE/RMSE en viajes/estación/hora, WAPE como proporción, R² puede ser negativo.",
        "",
        "| Escenario | N únicos fuente / N lote | Pico | Faltantes por lag | MAE | RMSE | WAPE | R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics.to_dict("records"):
        lines.append(
            f"| {r['escenario']} | {r['filas_fuente_unicas']:,} / {r['n']:,} | {r['fraccion_pico']:.2%} | {r['faltantes_lag_24']:.0%} | "
            + " | ".join(_fmt(r[k]) for k in ("mae", "rmse", "wape", "r2"))
            + " |"
        )
    lines += [
        "",
        "| Escenario | Δ MAE | Δ RMSE | Δ WAPE (pp) | Δ R² | Δ MAE relativo | Δ RMSE relativo |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics.to_dict("records"):
        lines.append(
            f"| {r['escenario']} | {r['delta_mae']:+.4f} | {r['delta_rmse']:+.4f} | {100 * r['delta_wape']:+.2f} | {r['delta_r2']:+.4f} | {r['delta_relativa_mae']:+.2%} | {r['delta_relativa_rmse']:+.2%} |"
        )
    lines += [
        "",
        "Δ = escenario − control. `metricas_escenarios.csv` conserva valores completos, diferencia con signo, diferencia absoluta (magnitud) y relativa de las cuatro métricas; el denominador relativo es |control|, y cero/indefinidos se dejan NA. El porcentaje de cambio de un R² negativo no es una mejora porcentual interpretable; la alerta usa caída en unidades. WAPE/R² se dejan NA si falta denominador/varianza.",
        "",
        "## Distribuciones, calidad y alertas observadas",
        "",
        "| Escenario | Drift vs historia | Drift añadido vs control | Calidad | Pérdida de desempeño | Métricas que activan pérdida |",
        "|---|---|---|---|---|---|",
    ]
    for r in metrics.to_dict("records"):
        perf = alerts[(alerts["escenario"] == r["escenario"]) & (alerts["tipo"] == "desempeno")][
            "variable"
        ].tolist()
        flags = [
            "sí" if r[c] else "no"
            for c in (
                "drift_historico",
                "drift_vs_control",
                "problema_calidad",
                "perdida_desempeno",
            )
        ]
        lines.append(
            f"| {r['escenario']} | "
            + " | ".join(flags)
            + " | "
            + (", ".join(perf) or "ninguna")
            + " |"
        )
    lines += [
        "",
        "Resumen PSI/KS de entradas y máximos (valores completos por variable/comparación en `detectores_drift.csv`):",
        "",
        "| Escenario | PSI máximo vs historia (entrada) | PSI máximo vs control (entrada) | KS lag_24 vs control | KS lag_168 vs control | Predicción máxima | SSE del 1% peor |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics.to_dict("records"):
        d = detectors[(detectors["escenario"] == r["escenario"]) & (detectors["rol"] == "entrada")]
        hist, ctrl = d[d["comparacion"] == "historica"], d[d["comparacion"] == "control"]
        ks = [
            ctrl.loc[ctrl["variable"] == c, "distancia_ks"].iloc[0] for c in ("lag_24", "lag_168")
        ]
        lines.append(
            f"| {r['escenario']} | {_fmt(hist['psi'].max())} | {_fmt(ctrl['psi'].max())} | {_fmt(ks[0])} | {_fmt(ks[1])} | {r['max_prediccion']:.4f} | {r['sse_top1_proporcion']:.2%} |"
        )
    lines += [
        "",
        "El control puede activar drift histórico, por ejemplo por el cambio calendario agosto→septiembre, aunque contra sí mismo tiene PSI cero y ninguna pérdida respecto a sí mismo. Esa alerta natural no significa automáticamente avería ni degradación del modelo. Las alertas de `alertas_acciones.csv` separan tipo, comparación, severidad, valor y acción.",
        "",
    ]
    for family in ("composicion", "faltantes", "combinado"):
        rows = metrics[metrics["familia"] == family]
        end = rows.iloc[-1]
        lines.append(
            f"- **{family}, 50%:** MAE cambia {end['delta_relativa_mae']:+.2%}, RMSE {end['delta_relativa_rmse']:+.2%}, WAPE {100 * end['delta_wape']:+.2f} pp y R² {end['delta_r2']:+.4f}. Pérdida según reglas: {'sí' if end['perdida_desempeno'] else 'no'}. No se presupone monotonicidad; se conservan todas las intensidades."
        )
    lines += [
        "",
        "La semana control ya tiene RMSE elevado y R² negativo por extremos previamente auditados (incluida estación 46). Si faltantes/remuestreo reducen un extremo, RMSE puede mejorar mientras MAE o WAPE empeoran; eso no convierte datos faltantes en una solución. `errores_mayores.csv` y `sse_top1_proporcion` permiten examinarlo sin eliminar filas. El cambio de composición también altera la distribución de y y su varianza, por lo que R²/WAPE entre mezclas tienen denominadores distintos. Son riesgos sobre cada mezcla de solicitudes, no estimaciones causales del cambio físico de demanda. El fallo de captura sí conserva el mismo y por posición y permite contraste pareado de errores.",
        "",
        "Para separar los mecanismos se reutiliza la predicción control de **cada fila fuente** en la mezcla simulada, antes de ocultar rezagos. Todas las filas sin fallo deben conservar esa predicción (tolerancia 1e-12). `descomposicion_errores.csv` descompone el cambio total en efecto mecánico de mezcla y efecto adicional de pérdida de entradas, en MAE y MSE; no se suman diferencias de RMSE porque la raíz no es aditiva. Es una descomposición del experimento, sin inferir efectos sobre demanda física.",
        "",
        "| Escenario | Δ MAE por mezcla | Δ MAE adicional por faltantes | Δ MSE por mezcla | Δ MSE adicional por faltantes |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in tables["descomposicion_errores"].to_dict("records"):
        lines.append(
            f"| {r['escenario']} | {r['delta_mae_por_composicion']:+.4f} | {r['delta_mae_por_faltantes']:+.4f} | {r['delta_mse_por_composicion']:+.4f} | {r['delta_mse_por_faltantes']:+.4f} |"
        )
    lines += [
        "",
        "![Distribuciones](drift/distribuciones_comparativas.png)",
        "",
        "![Desempeño según intensidad](drift/desempeno_intensidad.png)",
        "",
        "![Detectores PSI](drift/detectores_psi.png)",
        "",
        "![Calidad y colas de predicción](drift/calidad_y_predicciones.png)",
        "",
        "## Reproducción y pruebas",
        "",
        "Desde PowerShell en la raíz, rama fase3; usa exclusivamente el entorno existente:",
        "",
        "```powershell",
        r"& .\.venv\Scripts\python.exe -m mibici_demand.monitoring.drift --threads 4",
        r"& .\.venv\Scripts\python.exe -m pytest tests -q --junitxml=reports/fase3/drift/pytest_drift.xml",
        r"& .\.venv\Scripts\python.exe -m mibici_demand.monitoring.drift --actualizar-reporte",
        "```",
        "",
        "No se instalan dependencias nuevas: NumPy/Pandas/SciPy/Matplotlib/joblib ya están en `.venv`; sus versiones y hashes de código/plan constan en `ejecucion.json`. No requiere un servidor MLflow ni registra/modifica sus corridas anteriores. Para ejecutar también las diez pruebas Docker, mantener Docker Desktop Linux activo y la imagen `mibici-demand:fase3` construida previamente. Las salidas nuevas de este bloque se escriben solo en `reports/fase3/drift/` y este informe; README/documentación y código nuevo se gestionan aparte.",
        "",
    ]
    test = result.get("pytest")
    if test:
        lines += [
            f"Suite ejecutada: **{test['tests']} pruebas; {test['failures']} fallos, {test['errors']} errores y {test['skipped']} omitidas**; {test['time']} s. Evidencia JUnit: `drift/pytest_drift.xml`. Las 107 pruebas anteriores se conservan íntegramente. Las nuevas verifican reproducibilidad/anidación, integridad de etiquetas, disponibilidad temporal, tipos, PSI/KS/categorías, métricas/denominadores, calidad, alertas y uso sin fit.",
            "",
        ]
    else:
        lines += [
            "La simulación está ejecutada; la suite de este bloque todavía no se ha registrado. No se declara aprobada antes de ejecutarla.",
            "",
        ]
    lines += [
        f"Se verificaron hashes de {len(result['resultados_previos_sin_cambios'])} artefactos/resultados previos antes y después: todos sin cambios. El artefacto histórico desplegado tampoco se modificó. `predicciones_escenarios.parquet` conserva trazabilidad de cada solicitud simulada, sus etiquetas y predicciones originales.",
        "",
        "## Limitaciones y recomendaciones",
        "",
        "- Una semana, una semilla y perturbaciones controladas; no hay replicación que estime incertidumbre, potencia o falsas alarmas. La dependencia temporal y los duplicados impiden tratar N como muestras IID. Es una demostración offline reproducible, no validación productiva ni prueba de concept drift P(y|X).",
        "- La referencia de 28 días se hereda sin optimizarla; su suficiencia frente a seis meses sigue sin demostrar. Mes y cambios de cohorte/historia pueden dominar PSI; se deben monitorizar segmentos y calendarios comparables en una futura calibración anterior al periodo de aceptación.",
        "- Las métricas comparan etiquetas completas y no se omiten casos difíciles. En composición la mezcla objetivo cambia; en faltantes se mantiene la misma demanda. No se incrementan rezagos ni demandas con etiquetas incompatibles ni se modifica calendario sin su timestamp.",
        "- **Contrato API:** FastAPI actual rechaza rezagos faltantes/NaN con 422. Este experimento llama directamente al pipeline para estudiar la tolerancia del imputador; no demuestra que esas solicitudes sean aceptadas en producción. En operación deben vigilarse también rechazos, cobertura y latencia, sin abrir silenciosamente el esquema.",
        "- No hay ingestión en vivo, scheduler, alarmas externas, reentrenamiento automático ni sustitución del modelo. Calidad requiere corregir el pipeline de datos; drift sin pérdida requiere investigar/monitorizar; pérdida persistente con datos correctos justifica revisar el modelo y planear reentrenamiento con nueva validación temporal y holdout verdaderamente no examinado.",
        "- No se usan viajes de octubre, no se modifican backtesting/API/Docker y no se publica nada. No se prepara el PDF final. Las alertas y mejoras aparentes no rehabilitan el extremo previamente auditado ni acreditan fiabilidad operativa.",
        "",
    ]
    rendered = "\n".join(lines)
    if source["corte"][:10] != "2026-09-07":
        rendered = rendered.replace(
            "**primer corte cronológico**", "**último corte cronológico disponible**"
        )
        rendered = rendered.replace(
            "comienzos anteriores al 14", "comienzos anteriores al 28 de septiembre"
        )
        rendered = rendered.replace(
            "La semana control ya tiene RMSE elevado y R² negativo por extremos previamente auditados (incluida estación 46).",
            "El primer corte anterior mostró RMSE elevado y R² negativo por extremos (estación 46); el control de este candidato es una semana distinta y conserva sus propios resultados.",
        )
    if out.name != "drift":
        rendered = rendered.replace("(drift/", f"({out.name}/").replace("`drift/", f"`{out.name}/")
        rendered = rendered.replace(
            "mibici_demand.monitoring.drift --threads 4",
            "mibici_demand.monitoring.drift --threads 4 --plan references/plan_drift_final.json --modelo models/final/pipeline.joblib --salida reports/fase3/integracion/drift_final",
        )
    (out.parent / "validacion_drift.md").write_text(rendered, encoding="utf-8")
