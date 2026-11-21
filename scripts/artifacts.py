"""Empaqueta/restaura artefactos locales con SHA-256; no publica ni descarga."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from mibici_demand import config


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def pack() -> dict:
    root = config.ROOT
    execution = json.loads((root / "reports/fase3/ejecucion.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / "references/data_snapshot.json").read_text(encoding="utf-8"))
    names = {
        "models/best_pipeline.joblib",
        "models/final/pipeline.joblib",
        "data/processed/fase3/viajes_limpios.parquet",
        "data/processed/fase3/predicciones_backtest.parquet",
        "data/processed/fase3/demanda_horaria.parquet",
    }
    names.update(c["modelo_local"] for c in execution["cortes"])
    names.update(f["ruta"] for c in snapshot["cortes"] for f in c["archivos"])
    destination = root / "artifacts/mibici-fase3-v1.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            path = root / name
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
            entries.append({"path": name, "sha256": file_hash(path), "bytes": path.stat().st_size})
    manifest = {
        "version": "mibici-fase3-v1",
        "archive": destination.name,
        "sha256": file_hash(destination),
        "bytes": destination.stat().st_size,
        "download_url": None,
        "publication_status": "preparado_local_no_publicado",
        "files": entries,
        "raw_sources": "data/raw/manifest.csv; solo agosto y septiembre necesarios para reconstruir desde crudo",
    }
    (root / "references/artifact_bundle.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def restore(archive_path: Path, destination: Path = config.ROOT) -> int:
    manifest = json.loads(
        (config.ROOT / "references/artifact_bundle.json").read_text(encoding="utf-8")
    )
    if file_hash(archive_path) != manifest["sha256"]:
        raise ValueError("SHA-256 del ZIP incorrecto; no se deserializa ni restaura")
    allowed = {entry["path"]: entry for entry in manifest["files"]}
    destination = destination.resolve()
    if not destination.is_relative_to(config.ROOT.resolve()):
        raise ValueError("Restaurar solo dentro del repositorio")
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(allowed) or set(archive.namelist()) != set(allowed):
            raise ValueError("Contenido del ZIP distinto del manifiesto")
        payloads = []
        for name, entry in allowed.items():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Ruta de extracción fuera del destino")
            payload = archive.read(name)
            if len(payload) != entry["bytes"] or sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError("Artefacto interno adulterado")
            if target.exists() and file_hash(target) != entry["sha256"]:
                raise ValueError("No sobrescribir un artefacto diferente: " + name)
            payloads.append((target, payload))
        for target, payload in payloads:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
    return len(payloads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pack")
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        manifest = pack()
        print(json.dumps({k: v for k, v in manifest.items() if k != "files"}))
    else:
        print("Artefactos verificados/restaurados:", restore(args.archive))


if __name__ == "__main__":
    main()
