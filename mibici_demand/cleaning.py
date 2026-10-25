"""Limpieza de los viajes de un mes. Cada regla cuenta cuántas filas toca, para documentar todas las modificaciones."""

from __future__ import annotations

import pandas as pd

from mibici_demand import config


class TripCleaner:
    """Aplica las reglas de limpieza a los viajes de un mes y devuelve también un reporte con los conteos.

    perfil="limpio": aplica todas las reglas.
    perfil="bruto":  solo exige fecha de inicio y origen válidos; sirve de línea de comparación en MLflow.
    """

    def __init__(
        self,
        perfil: str = "limpio",
        min_seconds: int = config.MIN_TRIP_SECONDS,
        max_seconds: int = config.MAX_TRIP_SECONDS,
    ) -> None:
        if perfil not in ("limpio", "bruto"):
            raise ValueError("perfil debe ser 'limpio' o 'bruto'")
        self.perfil = perfil
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    def clean(self, trips: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        report = {"filas_originales": len(trips)}
        df = trips

        # 1) Sin fecha de inicio o sin estación de origen no se puede asignar el viaje a una estación y hora
        valid = df["Inicio_del_viaje"].notna() & df["Origen_Id"].notna()
        report["sin_inicio_u_origen"] = int((~valid).sum())
        df = df[valid]

        if self.perfil == "limpio":
            # 2) Filas repetidas: el mismo viaje aparece más de una vez y duplica la demanda
            before = len(df)
            df = df.drop_duplicates()
            report["duplicados_exactos"] = before - len(df)

            # 3) Duraciones imposibles: el fin antes del inicio, o un viaje de más de un día
            seconds = (df["Fin_del_viaje"] - df["Inicio_del_viaje"]).dt.total_seconds()
            impossible = seconds.notna() & ((seconds < 0) | (seconds > self.max_seconds))
            report["duracion_imposible"] = int(impossible.sum())
            df, seconds = df[~impossible], seconds[~impossible]

            # 4) Anclajes fallidos: viajes de segundos que salen y llegan a la misma estación
            failed = (seconds < self.min_seconds) & (df["Origen_Id"] == df["Destino_Id"])
            report["anclaje_fallido"] = int(failed.sum())
            df = df[~failed]

            # 5) Año de nacimiento fuera de rango: no afecta la demanda, pero se deja el dato como nulo
            df = df.copy()
            latest_ok = df["Inicio_del_viaje"].dt.year - config.MIN_AGE
            bad_year = df["Año_de_nacimiento"].notna() & (
                (df["Año_de_nacimiento"] < config.BIRTH_YEAR_MIN)
                | (df["Año_de_nacimiento"] > latest_ok)
            )
            report["anio_nacimiento_corregido"] = int(bad_year.sum())
            df.loc[bad_year, "Año_de_nacimiento"] = pd.NA
        else:
            report.update(
                {
                    "duplicados_exactos": 0,
                    "duracion_imposible": 0,
                    "anclaje_fallido": 0,
                    "anio_nacimiento_corregido": 0,
                }
            )

        report["filas_finales"] = len(df)
        return df.reset_index(drop=True), report
