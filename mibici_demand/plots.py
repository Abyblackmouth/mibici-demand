"""Gráficas del EDA. Cada función guarda una figura en reports/figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from mibici_demand import config

sns.set_theme(style="whitegrid")
DAYS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
THOUSANDS = FuncFormatter(lambda value, _: f"{value:,.0f}")
# Fechas que se marcan en la serie mensual: la pandemia y el inicio de la ventana que se compara en MLflow
EVENTS = [("2020-03-01", "pandemia (mar. 2020)"), ("2022-01-01", "ventana desde 2022")]


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def trips_per_month(summary: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Viajes limpios por mes y estaciones activas por mes: muestra los cambios de régimen (crecimiento de la red, pandemia)."""
    data = summary.sort_values("mes")
    months = pd.to_datetime(data["mes"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    top.plot(months, data["filas_finales"], color="#2C7FB8")
    top.set_ylabel("Viajes por mes")
    top.yaxis.set_major_formatter(THOUSANDS)
    bottom.plot(months, data["estaciones_activas"], color="#33A02C")
    bottom.set_ylabel("Estaciones activas")
    top.set_ylim(0, data["filas_finales"].max() * 1.12)
    for date, label in EVENTS:
        for ax in (top, bottom):
            ax.axvline(pd.Timestamp(date), color="#555555", linestyle="--", linewidth=1)
        top.text(
            pd.Timestamp(date),
            top.get_ylim()[1] * 0.02,
            f"{label} ",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="#333333",
        )
    top.set_title("Uso del sistema a lo largo del tiempo")
    return _save(fig, "viajes_y_estaciones_por_mes.png", out_dir)


def cleaning_by_rule(summary: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Filas que elimina cada regla de limpieza (rojo) y datos corregidos sin eliminar la fila (gris), en escala logarítmica."""
    labels = {
        "anclaje_fallido": "Anclaje fallido\n(< 60 s, mismo origen y destino)",
        "duracion_imposible": "Duración imposible\n(negativa o de más de 24 h)",
        "duplicados_exactos": "Filas duplicadas",
        "sin_inicio_u_origen": "Sin fecha de inicio u origen",
    }
    removed = summary[list(labels)].sum().sort_values()
    total = float(summary["filas_originales"].sum())
    items = [
        (
            "Año de nacimiento corregido\n(no elimina filas)",
            float(summary["anio_nacimiento_corregido"].sum()),
            "#7F7F7F",
        )
    ]
    items += [(labels[key], float(value), "#E31A1C") for key, value in removed.items()]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for y, (_, value, color) in enumerate(items):
        ax.barh(
            y, max(value, 1.0), color=color
        )  # en escala logarítmica el 0 no se puede dibujar; su etiqueta lo indica
        ax.text(
            max(value, 1.0) * 1.2,
            y,
            f"{int(value):,}  ({value / total:.3%})",
            va="center",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlim(1, max(v for _, v, _ in items) * 60)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([name for name, _, _ in items], fontsize=9)
    ax.set_xlabel("Filas afectadas (escala logarítmica)")
    ax.set_title(
        f"Limpieza: {int(removed.sum()):,} de {int(total):,} filas eliminadas ({removed.sum() / total:.2%})",
        fontsize=11,
    )
    return _save(fig, "limpieza_por_regla.png", out_dir)


def demand_heatmap(table: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Demanda media por hora del día y día de la semana. Las horas sin servicio se dejan en blanco para que el eje sea continuo."""
    pivot = table.groupby(["dia_semana", "hora"], observed=True)["demanda"].mean().unstack("hora")
    pivot = pivot.reindex(columns=range(24)).astype(float)
    pivot.index = [DAYS[i] for i in pivot.index]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Viajes por estación y hora"})
    closed = [h for h in range(24) if pivot[h].isna().all()]
    if closed:
        ax.text(
            np.mean(closed) + 0.5,
            len(pivot) / 2,
            "sin servicio",
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
        )
    ax.set_xlabel("Hora del día (promedio de todos los años con servicio a esa hora)")
    ax.tick_params(axis="y", rotation=0)
    ax.grid(False)
    ax.set_title("Demanda media por día de la semana y hora")
    return _save(fig, "demanda_hora_dia.png", out_dir)


def demand_distribution(table: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Distribución de la demanda por estación y hora. La última barra acumula los valores desde el percentil 99.9."""
    y = table["demanda"]
    q1, q3 = y.quantile([0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    share = float((y > fence).mean())
    cap = int(y.quantile(0.999))
    counts = y.clip(upper=cap).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(counts.index, counts.values, color="#2C7FB8")
    ax.set_yscale("log")
    ax.axvline(
        fence,
        color="#E31A1C",
        linestyle="--",
        label=f"Valla del IQR = {fence:.0f} viajes ({share:.1%} de las filas por encima)",
    )
    ticks = [t for t in range(0, cap - 2, 5)] + [cap]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks[:-1]] + [f"{cap} o más"])
    ax.annotate(
        f"{(y == 0).mean():.1%} de las filas\nno tienen viajes",
        xy=(0, counts.iloc[0]),
        xytext=(fence + 1.2, counts.iloc[0] * 0.45),
        fontsize=9,
    )
    ax.set_xlabel("Viajes iniciados en una estación en una hora")
    ax.set_ylabel("Frecuencia (escala log)")
    ax.set_title("Distribución de la demanda por estación y hora")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, "distribucion_demanda.png", out_dir)


def pca_stations(
    components: pd.DataFrame,
    explained: np.ndarray,
    out_dir: Path = config.FIGURES_DIR,
    table: pd.DataFrame | None = None,
) -> Path:
    """Estaciones en los dos primeros componentes del perfil semanal, coloreadas por su demanda media, y varianza explicada."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 4.7), gridspec_kw={"width_ratios": [1.7, 1]}
    )
    if table is not None:
        mean = (
            table.groupby("estacion")["demanda"].mean().reindex(components["estacion"]).to_numpy()
        )
        points = left.scatter(
            components["pc1"],
            components["pc2"],
            c=np.log10(np.maximum(mean, 1e-3)),
            cmap="viridis",
            s=14,
            alpha=0.85,
        )
        fig.colorbar(points, ax=left, label="Demanda media (log10)")
    else:
        left.scatter(components["pc1"], components["pc2"], s=14, alpha=0.7, color="#6A3D9A")
    left.set_xlabel(f"Componente 1 ({explained[0]:.0%} de la varianza)")
    left.set_ylabel(f"Componente 2 ({explained[1]:.0%} de la varianza)")
    left.set_title("Estaciones según su perfil semanal de demanda")
    shown = explained[:3] * 100
    right.bar([f"Comp. {i + 1}" for i in range(len(shown))], shown, color="#6A3D9A")
    for i, value in enumerate(shown):
        right.text(i, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
    right.set_ylim(0, 100)
    right.set_ylabel("Varianza explicada (%)")
    right.set_title(f"Con 3 componentes: {shown.sum():.1f}%")
    return _save(fig, "pca_estaciones.png", out_dir)


# ---------------------------------------------------------------------------------------------------------------
# Fase 2: visualización comparativa de modelos y perfil de predicciones vs. demanda real
# ---------------------------------------------------------------------------------------------------------------
def _pick_metric_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Ninguna de las columnas esperadas {candidates} existe en {list(df.columns)}")


def model_comparison_plot(results_df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Compara MAE, RMSE, WAPE global y WAPE en horas pico entre baselines y modelos ajustados.

    Genera `reports/figures/comparacion_modelos_fase2.png` diferenciando por color las
    referencias de la Fase 1 (`estacional_168h`, `ayer_24h`) frente a los algoritmos de ML
    (`ridge`, `poisson`, `hist_gbm_poisson`, `random_forest`).
    """
    model_col = _pick_metric_col(results_df, ("modelo", "model_name", "model"))
    mae_col = _pick_metric_col(results_df, ("test_mae", "mae", "val_mae"))
    rmse_col = _pick_metric_col(results_df, ("test_rmse", "rmse", "val_rmse"))
    wape_col = _pick_metric_col(results_df, ("test_wape", "wape", "val_wape"))
    wape_peak_col = _pick_metric_col(
        results_df, ("test_wape_horas_pico", "wape_horas_pico", "val_wape_horas_pico", wape_col)
    )

    data = results_df.sort_values(mae_col, ascending=True).reset_index(drop=True)
    names = data[model_col].astype(str).tolist()
    is_baseline = [
        ("baseline" in name.lower()) or ("168h" in name.lower()) or ("24h" in name.lower())
        for name in names
    ]
    colors = ["#7F7F7F" if b else "#2C7FB8" for b in is_baseline]
    if colors:
        # Destaca en verde el modelo con menor MAE
        colors[0] = "#33A02C" if not is_baseline[0] else colors[0]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), sharey=True)
    panels = [
        (axes[0, 0], mae_col, "MAE global (viajes / estación-hora)", False),
        (axes[0, 1], rmse_col, "RMSE global (viajes / estación-hora)", False),
        (axes[1, 0], wape_col, "WAPE global (%)", True),
        (axes[1, 1], wape_peak_col, "WAPE en horas pico 7-9h y 17-19h (%)", True),
    ]

    y_pos = np.arange(len(names))
    for ax, col, title, as_pct in panels:
        values = data[col].to_numpy(dtype=np.float64) * (100.0 if as_pct else 1.0)
        ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.65)
        max_val = float(np.nanmax(values)) if len(values) > 0 else 1.0
        ax.set_xlim(0, max(max_val * 1.22, 0.1))
        for idx, val in enumerate(values):
            label = f"{val:.1f}%" if as_pct else f"{val:.3f}"
            ax.text(
                val + max_val * 0.02,
                idx,
                label,
                va="center",
                fontsize=8.5,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5)

    fig.suptitle(
        "Fase 2: Comparación de modelos de Machine Learning frente a baselines estacionales",
        fontsize=12,
        y=1.01,
    )
    return _save(fig, "comparacion_modelos_fase2.png", out_dir)


def predictions_profile_plot(
    test_df: pd.DataFrame,
    y_pred: np.ndarray | pd.Series,
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    """Grafica la demanda agregada real vs. predicha y el MAE por hora del día en el periodo de prueba.

    Genera `reports/figures/prediccion_vs_real_fase2.png` con dos paneles:
    - Panel superior: serie temporal de viajes totales por hora en la red (real vs. predicho)
      durante los 7 días de prueba.
    - Panel inferior: error absoluto medio (MAE) por hora del día (`0..23`) comparando el mejor
      modelo contra el baseline estacional de 168 horas (`lag_168`), sombreando `config.PEAK_HOURS`.
    """
    work = test_df.copy()
    work["y_true"] = work["demanda"].to_numpy(dtype=np.float64)
    work["y_pred"] = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    work["err_modelo"] = np.abs(work["y_true"] - work["y_pred"])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11.5, 7.0))

    if "hora_inicio" in work.columns:
        hourly_series = (
            work.groupby("hora_inicio", observed=True)[["y_true", "y_pred"]].sum().sort_index()
        )
        x_time = hourly_series.index
    else:
        hourly_series = work[["y_true", "y_pred"]].reset_index(drop=True)
        x_time = hourly_series.index

    top.plot(
        x_time,
        hourly_series["y_true"],
        label="Demanda real observada",
        color="#1F78B4",
        linewidth=1.5,
    )
    top.plot(
        x_time,
        hourly_series["y_pred"],
        label="Demanda predicha (mejor modelo)",
        color="#E31A1C",
        linestyle="--",
        linewidth=1.4,
    )
    top.set_ylabel("Viajes totales por hora en la red")
    top.yaxis.set_major_formatter(THOUSANDS)
    top.set_title("Serie temporal de demanda horaria total: real vs. predicha (ventana de prueba)")
    top.legend(loc="upper right", fontsize=9)

    by_hour_model = work.groupby("hora", observed=True)["err_modelo"].mean()
    hours = by_hour_model.index.to_numpy(dtype=int)
    bottom.plot(
        hours,
        by_hour_model.to_numpy(),
        marker="o",
        color="#E31A1C",
        linewidth=1.8,
        label="Mejor modelo ML",
    )

    if "lag_168" in work.columns:
        work["err_baseline_168"] = np.abs(
            work["y_true"] - work["lag_168"].to_numpy(dtype=np.float64)
        )
        by_hour_base = work.groupby("hora", observed=True)["err_baseline_168"].mean()
        bottom.plot(
            by_hour_base.index.to_numpy(dtype=int),
            by_hour_base.to_numpy(),
            marker="s",
            color="#7F7F7F",
            linestyle=":",
            linewidth=1.6,
            label="Baseline estacional (168h)",
        )

    for peak_range in ((6.5, 9.5), (16.5, 19.5)):
        bottom.axvspan(
            peak_range[0],
            peak_range[1],
            color="#FDBF6F",
            alpha=0.22,
            label="Horas pico" if peak_range[0] < 10 else None,
        )

    bottom.set_xticks(range(24))
    bottom.set_xlabel("Hora del día (0-23 h)")
    bottom.set_ylabel("MAE (viajes / estación-hora)")
    bottom.set_title(
        "Error absoluto medio por hora del día: mejor modelo vs. baseline estacional (168h)"
    )
    bottom.legend(loc="upper left", fontsize=9)

    return _save(fig, "prediccion_vs_real_fase2.png", out_dir)
