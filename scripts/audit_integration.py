"""Verifica evidencias sin reescribir resultados históricos ni publicar archivos."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from mibici_demand import config, dataset
from mibici_demand.modeling.audit_backtest import (
    read_sources_independently,
    verify_labels_and_lags,
)
from mibici_demand.reproduce import metrics


def main():
    root = config.ROOT
    out = root / "reports/fase3/integracion"
    baseline_path = out / "baseline_evidencias.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = json.loads(
            (root / ".venv/integration-baseline.json").read_text(encoding="utf-8")
        )
        baseline = {
            p: h
            for p, h in baseline.items()
            if p.startswith(("reports/", "models/", "examples/", "data/"))
        }
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    previous_evidence = {
        p: dataset.sha256_file(root / p) == h
        for p, h in baseline.items()
        if p.startswith(("reports/", "models/", "examples/", "data/"))
    }
    assert all(previous_evidence.values()), "Una evidencia anterior cambió"
    old_test = subprocess.check_output(
        ["git", "show", "HEAD:tests/test_pipeline.py"], cwd=root, text=True, encoding="utf-8"
    )
    assert ast.dump(ast.parse(old_test)) == ast.dump(
        ast.parse((root / "tests/test_pipeline.py").read_text(encoding="utf-8"))
    )
    original = pd.read_parquet(root / "data/processed/fase3/predicciones_backtest.parquet")
    provenance = json.loads(
        (root / "reports/fase3/procedencia_datos.json").read_text(encoding="utf-8")
    )
    valid, _, source_integrity = read_sources_independently(root, provenance)
    raw_temporal_check = verify_labels_and_lags(valid, original)
    table = pd.read_csv(root / "reports/fase3/metricas_cortes.csv")
    metric_checks = []
    for row in table.to_dict("records"):
        group = original[original.corte.eq(row["corte"])]
        actual = metrics(group.demanda, group[row["modelo"]])
        for key, value in actual.items():
            np.testing.assert_allclose(value, row[key], rtol=1e-12, atol=1e-12)
        assert len(group) == row["filas"]
        metric_checks.append(
            {
                "corte": row["corte"],
                "modelo": row["modelo"],
                "filas": len(group),
                "metricas": actual,
            }
        )
    drift_checks = []
    for directory in (root / "reports/fase3/drift", out / "drift_final"):
        frame = pd.read_parquet(directory / "predicciones_escenarios.parquet")
        expected = pd.read_csv(directory / "metricas_escenarios.csv")
        control = frame[frame.escenario.eq("control")].reset_index(drop=True)
        for scenario, group in frame.groupby("escenario", sort=False):
            group = group.reset_index(drop=True)
            source = control.iloc[group.fila_fuente.to_numpy()].reset_index(drop=True)
            for column in (
                "demanda",
                "estacion",
                "hora_inicio",
                "origen_pronostico",
                "hora",
                "dia_semana",
                "mes",
            ):
                np.testing.assert_array_equal(group[column], source[column])
            for lag in (24, 168):
                observed = group[f"lag_{lag}"].notna()
                np.testing.assert_array_equal(
                    group.loc[observed, f"lag_{lag}"], source.loc[observed, f"lag_{lag}"]
                )
                assert group[f"lag_{lag}"].dtype == np.dtype("float32")
                assert (
                    group[f"fuente_lag_{lag}"] + pd.Timedelta(hours=1) <= group.origen_pronostico
                ).all()
            actual = metrics(group.demanda, group.prediccion)
            row = expected[expected.escenario.eq(scenario)].iloc[0]
            for key, value in actual.items():
                np.testing.assert_allclose(value, row[key], rtol=1e-12, atol=1e-12)
            drift_checks.append(
                {
                    "experimento": directory.relative_to(root).as_posix(),
                    "escenario": scenario,
                    "filas": len(group),
                    "metricas_verificadas": True,
                    "etiquetas_y_fuentes_integras": True,
                }
            )
    with sqlite3.connect((root / "mlflow_fase3.db").as_uri() + "?mode=ro", uri=True) as connection:
        for row in table.to_dict("records"):
            for metric in ("mae", "rmse", "wape", "r2"):
                value = connection.execute(
                    "SELECT value FROM metrics WHERE run_uuid=? AND key=? ORDER BY timestamp DESC LIMIT 1",
                    (row["mlflow_run_id"], row["modelo"] + "." + metric),
                ).fetchone()[0]
                np.testing.assert_allclose(value, row[metric], rtol=1e-12, atol=1e-12)
    commands = {}
    # Recarga real desde Model Registry, además de contrastar su joblib exacto.
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    registry = json.loads((out / "mlflow_release.json").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(registry["tracking_uri"])
    client = MlflowClient(tracking_uri=registry["tracking_uri"])
    version = client.get_model_version_by_alias(registry["registered_model"], registry["alias"])
    assert str(version.version) == str(registry["version"]) and version.status == "READY"
    assert client.get_run(registry["run_id"]).info.status == "FINISHED"
    model = mlflow.sklearn.load_model(registry["model_uri"])
    snapshot = json.loads((root / "references/data_snapshot.json").read_text(encoding="utf-8"))
    last = snapshot["cortes"][-1]
    evaluation = pd.read_parquet(root / last["archivos"][1]["ruta"])
    with threadpool_limits(limits=4):
        predicted = model.predict(evaluation[last["columnas"]])
    np.testing.assert_allclose(predicted, evaluation.prediccion_referencia, rtol=1e-12, atol=1e-12)
    registry_diff = float(np.max(np.abs(predicted - evaluation.prediccion_referencia)))
    sources = sorted(
        [
            *root.joinpath("mibici_demand").rglob("*.py"),
            *root.joinpath("scripts").glob("*.py"),
            *root.joinpath("tests").glob("*.py"),
        ]
    )
    for name, args in {
        "ruff_check": [
            ".venv/Scripts/python.exe",
            "-m",
            "ruff",
            "check",
            *[str(p) for p in sources],
        ],
        "ruff_format": [
            ".venv/Scripts/python.exe",
            "-m",
            "ruff",
            "format",
            "--check",
            *[str(p) for p in sources],
        ],
        "pip_check": [".venv/Scripts/python.exe", "-m", "pip", "check"],
        "pip_check_clean": [".venv/repro-clean/Scripts/python.exe", "-m", "pip", "check"],
        "diff_check": ["git", "diff", "--check"],
    }.items():
        result = subprocess.run(args, cwd=root, text=True, capture_output=True, check=False)
        commands[name] = {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        assert result.returncode == 0, commands[name]
    suite = ET.parse(out / "pytest_final.xml").getroot().find("testsuite").attrib
    assert all(suite[k] == "0" for k in ("failures", "errors", "skipped"))
    secrets = []
    eligible = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True
    ).splitlines()
    eligible = sorted(set(eligible))
    for name in (
        "reports/fase3/integracion/archivos_github.txt",
        "reports/fase3/integracion/auditoria_comprobaciones.json",
        "reports/fase3/integracion/cambios_este_bloque.json",
        "reports/fase3/auditoria_integracion_final.md",
    ):
        if name not in eligible:
            eligible.append(name)
    eligible = sorted(eligible)
    # Nunca se imprimen posibles valores sensibles, solo ubicación y detector.
    for name in eligible:
        path = root / name
        if not path.is_file() or path.suffix.lower() not in {
            ".py",
            ".md",
            ".toml",
            ".json",
            ".ipynb",
            ".txt",
            ".yml",
        }:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in {
            "github_pat": r"gh[pousr]_[A-Za-z0-9]{30,}",
            "aws_key": r"AKIA[A-Z0-9]{16}",
            "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        }.items():
            if re.search(pattern, content):
                secrets.append({"archivo": name, "detector": label})
    assert not secrets, "Revisar hallazgos de credenciales sin imprimir sus valores"
    for name in eligible:
        assert not name.startswith((".venv/", "mlruns/", "mlartifacts/", "artifacts/", "cache/"))
        assert Path(name).suffix.lower() not in {
            ".joblib",
            ".parquet",
            ".db",
            ".zip",
            ".pkl",
            ".pickle",
        }
    (out / "archivos_github.txt").write_text("\n".join(eligible) + "\n", encoding="utf-8")
    write = {
        "estado": "verificado",
        "pytest": suite,
        "archivos_evidencia_previos_preservados": len(previous_evidence),
        "evidencias_previas_sha256_sin_cambios": previous_evidence,
        "metricas_tres_cortes": metric_checks,
        "escenarios_drift_verificados": drift_checks,
        "metricas_mlflow_originales_verificadas": 36,
        "mlflow_candidato_recargado_filas": len(evaluation),
        "mlflow_candidato_diferencia_maxima": registry_diff,
        "tests_historicos_ast_sin_cambios": True,
        "relectura_fuentes_crudas": source_integrity,
        "etiquetas_y_rezagos_desde_crudo": raw_temporal_check,
        "checks": commands,
        "credenciales_detectadas": secrets,
        "alcance_credenciales": "Patrones de tokens GitHub/AWS y claves privadas; no es una auditoria de seguridad exhaustiva",
        "archivos_github": len(eligible),
        "archivos_github_sha256": sha256((out / "archivos_github.txt").read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "rama": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip(),
        "codigo_sha256": {p.relative_to(root).as_posix(): dataset.sha256_file(p) for p in sources},
    }
    (out / "auditoria_comprobaciones.json").write_text(
        json.dumps(write, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    changed = {
        "modificados_desde_baseline": [
            p
            for p, h in baseline.items()
            if (root / p).is_file() and dataset.sha256_file(root / p) != h
        ],
        "configuracion_actualizada": [
            ".gitignore",
            ".dockerignore",
            "Dockerfile",
            "Makefile",
            "README.md",
            "pyproject.toml",
        ],
        "nota": "La lista de fuentes/evidencias elegibles para GitHub incluye tambien el trabajo local de bloques anteriores; no implica staging ni commit",
    }
    # La baseline pública anterior solo cubre evidencia inmutable; el inventario de
    # cambios de código de la sesión se conserva cuando existe su baseline local.
    local = root / ".venv/integration-baseline.json"
    if local.is_file():
        before = json.loads(local.read_text(encoding="utf-8"))
        changed["modificados_desde_baseline"] = [
            p for p, h in before.items() if dataset.sha256_file(root / p) != h
        ]
        tracked = set(
            subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
        )
        root_files_before = {
            "Dockerfile",
            ".dockerignore",
            "requirements-linux-py313.txt",
            "requirements-win-py313.txt",
        }
        changed["nuevos_elegibles"] = sorted(
            p
            for p in eligible
            if p not in before and p not in tracked and p not in root_files_before
        )
    (out / "cambios_este_bloque.json").write_text(
        json.dumps(changed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pytest": suite,
                "cortes": len(metric_checks),
                "escenarios": len(drift_checks),
                "archivos_github": len(eligible),
                "preservados": len(previous_evidence),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
