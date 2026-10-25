"""Obtención de los datos: descubrir los archivos publicados, descargarlos sin modificarlos y leerlos."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import unicodedata
from urllib.parse import urljoin

import pandas as pd

from mibici_demand import config

# Acepta enlaces completos o relativos, en el HTML de la página o en su versión en markdown
_LINK = r"[^\s\"'<>()\[\]]*"
_FILE_RE = re.compile(_LINK + r"datos_abiertos_(\d{4})_(\d{2})" + _LINK + r"\.csv")
_CATALOG_RE = re.compile(_LINK + r"nomenclatura_\d{4}_\d{2}" + _LINK + r"\.csv")


# El sitio rechaza el identificador por defecto de la librería requests (error 406); se identifica el proyecto con uno propio
HEADERS = {"User-Agent": "mibici-demand-fase1"}


@dataclass(frozen=True)
class RemoteFile:
    year: int
    month: int
    url: str

    @property
    def period(self) -> str:
        return f"{self.year}-{self.month:02d}"

    @property
    def filename(self) -> str:
        return f"datos_abiertos_{self.year}_{self.month:02d}.csv"


def parse_index(
    page_text: str, base_url: str = config.OPEN_DATA_URL
) -> tuple[list[RemoteFile], str | None]:
    """Extrae de la página los archivos mensuales (uno por año y mes) y el catálogo de estaciones."""
    found: dict[tuple[int, int], RemoteFile] = {}
    for match in _FILE_RE.finditer(page_text):
        year, month = int(match.group(1)), int(match.group(2))
        found.setdefault((year, month), RemoteFile(year, month, urljoin(base_url, match.group(0))))
    catalog = _CATALOG_RE.search(page_text)
    return [found[k] for k in sorted(found)], (
        urljoin(base_url, catalog.group(0)) if catalog else None
    )


def in_window(period: str, desde: str | None, hasta: str | None) -> bool:
    return (desde is None or period >= desde) and (hasta is None or period <= hasta)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    desde: str | None = None, hasta: str | None = None, raw_dir: Path = config.RAW_DIR
) -> pd.DataFrame:
    """Descarga los archivos que falten y escribe el manifiesto (nombre, url, tamaño, hash).
    Un archivo que ya existe no se vuelve a descargar ni se sobrescribe: data/raw es inmutable."""
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    page = requests.get(config.OPEN_DATA_URL, headers=HEADERS, timeout=60)
    page.raise_for_status()
    files, catalog_url = parse_index(page.text)
    if not files:
        raise RuntimeError(
            "no se encontraron enlaces a archivos en la página; puede haber cambiado de formato"
        )
    selected = [f for f in files if in_window(f.period, desde, hasta)]
    targets = [(f.filename, f.url) for f in selected]
    if catalog_url:
        targets.append((Path(catalog_url).name, catalog_url))

    for name, url in targets:
        destination = raw_dir / name
        if destination.exists():
            continue
        partial = destination.with_suffix(".part")
        with requests.get(url, headers=HEADERS, stream=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for block in response.iter_content(1 << 20):
                    handle.write(block)
        partial.rename(destination)
        print(f"descargado: {name}")
    return write_manifest(
        raw_dir,
        {f.filename: f.url for f in selected}
        | ({Path(catalog_url).name: catalog_url} if catalog_url else {}),
    )


def write_manifest(
    raw_dir: Path = config.RAW_DIR, urls: dict[str, str] | None = None
) -> pd.DataFrame:
    """Registra qué archivos hay en data/raw con su tamaño y hash, para poder comprobar que no cambiaron."""
    urls = urls or {}
    old = (
        pd.read_csv(raw_dir / "manifest.csv")
        if (raw_dir / "manifest.csv").exists()
        else pd.DataFrame()
    )
    known = dict(zip(old.get("archivo", []), old.get("url", [])))
    rows = []
    for path in sorted(raw_dir.glob("*.csv")):
        if path.name == "manifest.csv":
            continue
        rows.append(
            {
                "archivo": path.name,
                "url": urls.get(path.name, known.get(path.name, "")),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(rows, columns=["archivo", "url", "bytes", "sha256"])
    manifest.to_csv(raw_dir / "manifest.csv", index=False)
    return manifest


def period_of(path: Path) -> str:
    match = re.search(r"datos_abiertos_(\d{4})_(\d{2})", path.name)
    if not match:
        raise ValueError(f"nombre de archivo inesperado: {path.name}")
    return f"{match.group(1)}-{match.group(2)}"


def list_raw_files(
    desde: str | None = None, hasta: str | None = None, raw_dir: Path = config.RAW_DIR
) -> list[Path]:
    files = [p for p in raw_dir.glob("datos_abiertos_*.csv")]
    files = [p for p in files if in_window(period_of(p), desde, hasta)]
    return sorted(files, key=period_of)


_REQUIRED = ["Viaje_Id", "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]


def _normalize(name: str) -> str:
    """Minúsculas, sin acentos y con guiones bajos: 'Año de nacimiento' y 'Ano_de_nacimiento' quedan iguales."""
    text = unicodedata.normalize("NFKD", name.strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


_CANONICAL = {_normalize(c): c for c in config.TRIP_COLUMNS}


def _canonical_name(column: str) -> str:
    key = _normalize(column)
    if key in _CANONICAL:
        return _CANONICAL[key]
    if "nacimiento" in key:  # variantes con acento, sin acento o con la codificación dañada
        return "Año_de_nacimiento"
    return column


_DATE_FORMATS = [
    "%d/%m/%Y %H:%M",  # DD/MM/AAAA h:mm, como se publicó noviembre de 2023
    "%d/%m/%Y %H:%M:%S",
    "%Y-%d-%m %H:%M:%S",  # día y mes invertidos
]


def _parse_dates(
    starts: pd.Series, ends: pd.Series, period: str
) -> tuple[pd.Series, pd.Series, str]:
    """Interpreta las fechas eligiendo el formato con el que los viajes caen dentro del mes del archivo.
    Primero la lectura automática (AAAA-MM-DD hh:mm:ss); si menos del 90% cae en su mes, prueba los formatos
    alternativos con los que se publicaron algunos meses y usa el que mejor encaje."""

    def in_month(parsed: pd.Series) -> float:
        return float((parsed.dt.strftime("%Y-%m") == period).mean())

    start = pd.to_datetime(starts, errors="coerce")
    best_score, best_format, best_start = in_month(start), None, start
    if best_score < 0.9:
        for fmt in _DATE_FORMATS:
            candidate = pd.to_datetime(starts, format=fmt, errors="coerce")
            score = in_month(candidate)
            if score > best_score:
                best_score, best_format, best_start = score, fmt, candidate
    if best_format is None:
        return start, pd.to_datetime(ends, errors="coerce"), "AAAA-MM-DD"
    return best_start, pd.to_datetime(ends, format=best_format, errors="coerce"), best_format


def read_trips(path: Path) -> pd.DataFrame:
    """Lee un mes como texto y convierte los tipos de forma explícita: lo que no se pueda convertir queda como nulo
    y se contará en el reporte de limpieza, en vez de ocultarse."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
    df = df.rename(columns=_canonical_name)
    df = df.loc[:, ~df.columns.duplicated()]
    missing_required = [c for c in _REQUIRED if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"{path.name}: faltan columnas obligatorias {missing_required}; columnas encontradas: {list(df.columns)}"
        )
    # Las columnas que no afectan la demanda (usuario, género, año de nacimiento) pueden faltar: quedan como nulas
    missing_optional = [c for c in config.TRIP_COLUMNS if c not in df.columns]
    for column in missing_optional:
        df[column] = ""
    df = df[config.TRIP_COLUMNS].apply(lambda s: s.str.strip())
    df = df.replace("", pd.NA)
    start, end, date_format = _parse_dates(
        df["Inicio_del_viaje"], df["Fin_del_viaje"], period_of(path)
    )
    out = pd.DataFrame(
        {
            "Viaje_Id": pd.to_numeric(df["Viaje_Id"], errors="coerce"),
            "Usuario_Id": pd.to_numeric(df["Usuario_Id"], errors="coerce"),
            "Genero": df["Genero"],
            "Año_de_nacimiento": pd.to_numeric(df["Año_de_nacimiento"], errors="coerce"),
            "Inicio_del_viaje": start,
            "Fin_del_viaje": end,
            "Origen_Id": pd.to_numeric(df["Origen_Id"], errors="coerce"),
            "Destino_Id": pd.to_numeric(df["Destino_Id"], errors="coerce"),
        }
    )
    out.attrs["columnas_faltantes"] = missing_optional
    out.attrs["formato_fecha"] = date_format
    return out


def read_catalog(path: Path | None = None) -> pd.DataFrame | None:
    """Catálogo de estaciones (id, name, latitude, longitude, dpcapacity), si ya se descargó."""
    if path is None:
        candidates = sorted(config.RAW_DIR.glob("nomenclatura_*.csv"))
        if not candidates:
            return None
        path = candidates[-1]
    catalog = pd.read_csv(path)
    catalog["activa_en_catalogo"] = ~catalog["name"].astype(str).str.lower().str.startswith("ex-")
    return catalog
