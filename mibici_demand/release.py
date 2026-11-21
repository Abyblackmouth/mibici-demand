"""Contrato del candidato técnico común, sin dependencias de MLflow al servir."""

import json
from pathlib import Path

from mibici_demand import config

RELEASE_MANIFEST = config.ROOT / "references/model_release.json"


def read_release() -> dict:
    return json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))


def default_model_path() -> Path:
    return config.ROOT / read_release()["model_path"]


def artifact_contract(path: Path, digest: str) -> dict | None:
    if not RELEASE_MANIFEST.is_file():
        return None
    release = read_release()
    if digest == release["sha256"]:
        return release
    if path.resolve() == default_model_path().resolve():
        raise ValueError("El SHA-256 del candidato final no coincide con su manifiesto.")
    return None
