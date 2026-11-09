# Resumen Auditable de MLflow Tracking y Model Registry (Fase 2)

- **Tracking URI**: `sqlite:///mlflow.db` (base local del proyecto; no se versiona)
- **Experimento**: `mibici-fase2-modelado` (ID `1`)
- **Modelo en Registry**: `mibici-demanda-estacion-hora`
- **Runs principales registrados**: 18
- **Runs hijos anidados (`nested=True`) de ajuste de hiperparámetros**: 40

## 1. Versiones registradas en MLflow Model Registry

| Versión | Alias | Algoritmo | Ventana | Horizonte | MAE | RMSE | WAPE | WAPE Horas Pico | Hiperparámetros óptimos | Run ID |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| v1 | - | `hist_gbm_poisson` | `completa` | `1h` | 1.1129 | 1.7301 | 0.5149 | 0.4507 | `{"min_samples_leaf": 20, "max_leaf_nodes": 127, "max_iter": 350, "learning_rate": 0.03, "l2_regularization": 0.0}` | `28c14503ad` |
| v2 | - | `hist_gbm_poisson` | `completa` | `24h` | 1.1027 | 1.7510 | 0.5102 | 0.4466 | `{"min_samples_leaf": 100, "max_leaf_nodes": 127, "max_iter": 250, "learning_rate": 0.1, "l2_regularization": 0.1}` | `f1abc4e2e5` |
| v3 | `@champion` | `hist_gbm_poisson` | `desde2022` | `24h` | 1.1027 | 1.7510 | 0.5102 | 0.4466 | `{"min_samples_leaf": 100, "max_leaf_nodes": 127, "max_iter": 250, "learning_rate": 0.1, "l2_regularization": 0.1}` | `476516c68a` |

## 2. Comparativa de Runs Principales en MLflow Tracking (`mibici-fase2-modelado`)

| Run Name | Algoritmo | Ventana | Horizonte | MAE | RMSE | WAPE | R2 | MAE Pico | WAPE Pico | Mejora WAPE vs Baseline (%) | Hiperparámetros |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `hist_gbm_poisson-24h-desde2022` | `hist_gbm_poisson` | `desde2022` | `24h` | 1.1027 | 1.7510 | 0.5102 | 0.7272 | 1.3550 | 0.4466 | +19.33% | `{"min_samples_leaf": "100", "max_leaf_nodes": "127", "max_iter": "250", "learning_rate": "0.1", "l2_regularization": "0.1"}` |
| `hist_gbm_poisson-24h-completa` | `hist_gbm_poisson` | `completa` | `24h` | 1.1027 | 1.7510 | 0.5102 | 0.7272 | 1.3550 | 0.4466 | +19.33% | `{"min_samples_leaf": "100", "max_leaf_nodes": "127", "max_iter": "250", "learning_rate": "0.1", "l2_regularization": "0.1"}` |
| `hist_gbm_poisson-1h-completa` | `hist_gbm_poisson` | `completa` | `1h` | 1.1129 | 1.7301 | 0.5149 | 0.7337 | 1.3675 | 0.4507 | +18.59% | `{"min_samples_leaf": "20", "max_leaf_nodes": "127", "max_iter": "350", "learning_rate": "0.03", "l2_regularization": "0.0"}` |
| `random_forest-1h-completa` | `random_forest` | `completa` | `1h` | 1.1366 | 1.7738 | 0.5259 | 0.7200 | 1.3870 | 0.4571 | +16.86% | `{"n_estimators": "100", "min_samples_leaf": "10", "max_features": "0.7", "max_depth": "20"}` |
| `random_forest-24h-desde2022` | `random_forest` | `desde2022` | `24h` | 1.1372 | 1.8062 | 0.5262 | 0.7097 | 1.3920 | 0.4588 | +16.81% | `{"n_estimators": "100", "min_samples_leaf": "10", "max_features": "0.7", "max_depth": "20"}` |
| `random_forest-24h-completa` | `random_forest` | `completa` | `24h` | 1.1372 | 1.8062 | 0.5262 | 0.7097 | 1.3920 | 0.4588 | +16.81% | `{"n_estimators": "100", "min_samples_leaf": "10", "max_features": "0.7", "max_depth": "20"}` |
| `ridge-1h-completa` | `ridge` | `completa` | `1h` | 1.2020 | 1.8613 | 0.5561 | 0.6918 | 1.4848 | 0.4894 | +12.07% | `{"alpha": "100.0"}` |
| `ridge-24h-desde2022` | `ridge` | `desde2022` | `24h` | 1.2332 | 1.9352 | 0.5706 | 0.6668 | 1.5242 | 0.5023 | +9.79% | `{"alpha": "100.0"}` |
| `ridge-24h-completa` | `ridge` | `completa` | `24h` | 1.2332 | 1.9352 | 0.5706 | 0.6668 | 1.5242 | 0.5023 | +9.79% | `{"alpha": "100.0"}` |
| `baseline_estacional_168h-24h-desde2022` | `baseline_estacional_168h` | `desde2022` | `24h` | 1.3670 | 2.2727 | 0.6325 | 0.5405 | 1.6916 | 0.5575 | +0.00% | `{"columna_referencia": "lag_168"}` |
| `baseline_estacional_168h-24h-completa` | `baseline_estacional_168h` | `completa` | `24h` | 1.3670 | 2.2727 | 0.6325 | 0.5405 | 1.6916 | 0.5575 | +0.00% | `{"columna_referencia": "lag_168"}` |
| `baseline_estacional_168h-1h-completa` | `baseline_estacional_168h` | `completa` | `1h` | 1.3670 | 2.2727 | 0.6325 | 0.5405 | 1.6916 | 0.5575 | +0.00% | `{"columna_referencia": "lag_168"}` |
| `poisson-24h-desde2022` | `poisson` | `desde2022` | `24h` | 1.4978 | 4.3462 | 0.6930 | -0.6807 | 2.0173 | 0.6649 | -9.57% | `{"max_iter": "200", "alpha": "0.0001"}` |
| `poisson-24h-completa` | `poisson` | `completa` | `24h` | 1.4978 | 4.3462 | 0.6930 | -0.6807 | 2.0173 | 0.6649 | -9.57% | `{"max_iter": "200", "alpha": "0.0001"}` |
| `baseline_ayer_24h-24h-desde2022` | `baseline_ayer_24h` | `desde2022` | `24h` | 1.5180 | 2.6245 | 0.7024 | 0.3872 | 1.9849 | 0.6542 | -11.05% | `{"columna_referencia": "lag_24"}` |
| `baseline_ayer_24h-24h-completa` | `baseline_ayer_24h` | `completa` | `24h` | 1.5180 | 2.6245 | 0.7024 | 0.3872 | 1.9849 | 0.6542 | -11.05% | `{"columna_referencia": "lag_24"}` |
| `baseline_ayer_24h-1h-completa` | `baseline_ayer_24h` | `completa` | `1h` | 1.5180 | 2.6245 | 0.7024 | 0.3872 | 1.9849 | 0.6542 | -11.05% | `{"columna_referencia": "lag_24"}` |
| `poisson-1h-completa` | `poisson` | `completa` | `1h` | 1.5269 | 5.2715 | 0.7065 | -1.4725 | 2.0751 | 0.6839 | -11.70% | `{"max_iter": "200", "alpha": "0.0001"}` |
