"""Nodos puros para la entrega intermedia de julio de 2026."""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from credit_risk_frontier import utils


TU_CODES = list(utils.TU_VARS)


ALLOWED = {
    ("logreg_tu_form", "tu_form", "trained"),
    ("xgb_tu_form", "tu_form", "trained"),
    ("qwen3:8b", "tu_form", "zero"),
    ("qwen3:8b", "tu_form_description", "zero"),
    ("qwen3:8b", "tu_form", "few"),
    ("qwen3:8b", "tu_form_description", "few"),
}


_XGB_SEARCH = {
    "n_estimators": (200, 800),
    "max_depth": (3, 7),
    "learning_rate": (0.01, 0.2),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "reg_alpha": (1e-4, 10.0),
    "reg_lambda": (1e-4, 10.0),
}


def _fit_intermediate_logreg(X_train, y_train, X_val, y_val, params: dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    best_model = None
    best_auc = -np.inf
    for c_value in params.get("logreg_c_grid", [0.01, 0.1, 1.0, 10.0]):
        model = LogisticRegression(
            C=float(c_value), max_iter=3000, solver="liblinear",
            random_state=int(params.get("seed", utils.SEED)),
        )
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc, best_model = auc, model
    return best_model, float(best_auc)


def _fit_intermediate_xgb(X_train, y_train, X_val, y_val, params: dict):
    """Busca hiperparámetros solo con validación y ajusta el XGBoost elegido."""
    import optuna
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    seed = int(params.get("seed", utils.SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        options = {
            "n_estimators": trial.suggest_int("n_estimators", *_XGB_SEARCH["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *_XGB_SEARCH["max_depth"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", *_XGB_SEARCH["learning_rate"], log=True
            ),
            "subsample": trial.suggest_float("subsample", *_XGB_SEARCH["subsample"]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *_XGB_SEARCH["colsample_bytree"]
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *_XGB_SEARCH["min_child_weight"]
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", *_XGB_SEARCH["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *_XGB_SEARCH["reg_lambda"], log=True
            ),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": seed,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**options, early_stopping_rounds=30, verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        objective,
        n_trials=int(params.get("xgb_n_trials", 20)),
        show_progress_bar=False,
    )
    best = dict(study.best_params)
    best.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": seed,
        "n_jobs": -1,
    })
    model = xgb.XGBClassifier(**best, early_stopping_rounds=30, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def build_intermediate_results(metrics: pd.DataFrame) -> pd.DataFrame:
    """Selecciona solo las seis configuraciones del diseño comparativo vigente."""
    keys = list(zip(metrics["model"], metrics["feature_profile"], metrics["mode"]))
    out = metrics.loc[metrics["segment"].eq("total") & pd.Series(
        [k in ALLOWED for k in keys], index=metrics.index
    )].copy()
    labels = {
        "xgb_tu_form": "XGBoost",
        "logreg_tu_form": "Regresión logística",
        "qwen3:8b": "Qwen3-8B",
    }
    out["modelo_presentado"] = out["model"].map(labels)
    out["informacion_presentada"] = out["feature_profile"].map({
        "tu": "20 variables de TransUnion",
        "tu_form": "TransUnion más 9 variables directas del formulario",
        "tu_form_description": (
            "TransUnion más 9 variables directas y descripción del negocio"
        ),
    })
    out["modalidad_presentada"] = out["mode"].map({
        "trained": "Entrenado",
        "zero": "Sin ejemplos",
        "few": "Ocho ejemplos",
    })
    cols = ["modelo_presentado", "informacion_presentada", "modalidad_presentada",
            "n", "n_total", "n_valid", "n_invalid", "AUC", "AUC_ci_low",
            "AUC_ci_high", "Gini", "KS", "Brier", "PR_AUC", "ECE",
            "model", "feature_profile", "mode"]
    order = {
        ("logreg_tu_form", "tu_form", "trained"): 0,
        ("xgb_tu_form", "tu_form", "trained"): 1,
        ("qwen3:8b", "tu_form", "zero"): 2,
        ("qwen3:8b", "tu_form_description", "zero"): 3,
        ("qwen3:8b", "tu_form", "few"): 4,
        ("qwen3:8b", "tu_form_description", "few"): 5,
    }
    out["_order"] = [order[key] for key in zip(
        out["model"], out["feature_profile"], out["mode"]
    )]
    return out.sort_values("_order")[cols].reset_index(drop=True)


def _prepare_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convierte a número y trata los códigos negativos del buró como ausencia."""
    result = frame[columns].apply(pd.to_numeric, errors="coerce").copy()
    tu_columns = [column for column in columns if column in utils.TU_VARS]
    result.loc[:, tu_columns] = result[tu_columns].mask(result[tu_columns] < 0)
    return result


def run_intermediate_classic_experiments(
    model_input_table: pd.DataFrame,
    params: dict,
    delivery_params: dict,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Entrena los dos brazos clásicos del diseño aprobado.

    La regresión logística, XGBoost y Qwen comparten las mismas veintinueve
    variables estructuradas: veinte de TransUnion y nueve declaraciones directas
    del formulario. Ningún modelo clásico recibe texto libre.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    frame = model_input_table.sort_values(
        ["fecha_desembolso", "credito_id_anon"], kind="stable"
    ).copy()
    split_frames = {
        split: frame[frame["set"].eq(split)].copy()
        for split in ("train", "val", "test")
    }
    if any(part.empty for part in split_frames.values()):
        raise ValueError("Los splits train/val/test deben tener observaciones")

    feature_contract = delivery_params["feature_contract"]
    arms = {
        "tu_form": utils.intermediate_feature_columns(
            frame, "tu_form", feature_contract
        ),
    }
    matrices = {
        profile: {
            split: _prepare_numeric(part, columns)
            for split, part in split_frames.items()
        }
        for profile, columns in arms.items()
    }
    y = {
        split: part["target"].astype(int).to_numpy()
        for split, part in split_frames.items()
    }

    predictions: list[dict] = []
    metrics: dict[str, dict] = {}
    logreg_coefficients: list[dict] = []
    xgb_shap_summary: list[dict] = []
    specifications = [
        ("logreg_tu_form", "logreg", "tu_form"),
        ("xgb_tu_form", "xgb", "tu_form"),
    ]
    for model_name, estimator_type, profile in specifications:
        values = matrices[profile]
        if estimator_type == "logreg":
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_train = scaler.fit_transform(imputer.fit_transform(values["train"]))
            x_val = scaler.transform(imputer.transform(values["val"]))
            x_test = scaler.transform(imputer.transform(values["test"]))
            estimator, validation_auc = _fit_intermediate_logreg(
                x_train, y["train"], x_val, y["val"], params
            )
            logreg_coefficients = [
                {
                    "codigo": feature,
                    "coeficiente_estandarizado": float(coefficient),
                    "razon_de_odds_por_desvio": float(np.exp(coefficient)),
                    "valor_absoluto": float(abs(coefficient)),
                    "C_seleccionado": float(estimator.get_params()["C"]),
                }
                for feature, coefficient in zip(
                    arms[profile], estimator.coef_[0], strict=True
                )
            ]
        else:
            x_train, x_val, x_test = (
                values["train"], values["val"], values["test"]
            )
            estimator = _fit_intermediate_xgb(
                x_train, y["train"], x_val, y["val"], params
            )
            validation_auc = utils.credit_metrics(
                y["val"], estimator.predict_proba(x_val)[:, 1]
            )["AUC"]
            # XGBoost devuelve una contribución aditiva por variable y una base.
            # Se calcula sobre prueba solo para describir el modelo ya cerrado;
            # no interviene en selección de variables ni hiperparámetros.
            import xgboost as xgb

            dtest = xgb.DMatrix(
                x_test,
                feature_names=arms[profile],
                missing=np.nan,
            )
            contribution_options = {"pred_contribs": True}
            best_iteration = getattr(estimator, "best_iteration", None)
            if best_iteration is not None:
                contribution_options["iteration_range"] = (
                    0,
                    int(best_iteration) + 1,
                )
            booster = estimator.get_booster()
            contributions = booster.predict(dtest, **contribution_options)
            expected_shape = (len(x_test), len(arms[profile]) + 1)
            if contributions.shape != expected_shape:
                raise ValueError(
                    "Las contribuciones SHAP de XGBoost no coinciden con las "
                    f"variables esperadas: {contributions.shape} != {expected_shape}"
                )
            margin_options = {"output_margin": True}
            if best_iteration is not None:
                margin_options["iteration_range"] = (
                    0,
                    int(best_iteration) + 1,
                )
            raw_margin = booster.predict(dtest, **margin_options)
            if not np.allclose(
                contributions.sum(axis=1), raw_margin, rtol=1e-5, atol=1e-5
            ):
                raise ValueError(
                    "Las contribuciones SHAP no reconstruyen la salida de XGBoost"
                )
            xgb_shap_summary = [
                {
                    "codigo": feature,
                    "shap_medio_absoluto": float(
                        np.mean(np.abs(contributions[:, index]))
                    ),
                    "shap_medio": float(np.mean(contributions[:, index])),
                    "unidad_salida": "logaritmo_de_chances",
                    "n_prueba": int(len(x_test)),
                }
                for index, feature in enumerate(arms[profile])
            ]

        probability = estimator.predict_proba(x_test)[:, 1]
        test_metrics = utils.credit_metrics(y["test"], probability)
        if estimator_type == "logreg":
            selected_parameters = {
                "C": float(estimator.get_params()["C"]),
                "solver": estimator.get_params()["solver"],
                "missing_values": "training_median",
                "standardization": "training_mean_and_standard_deviation",
            }
        else:
            selected_parameters = {
                key: estimator.get_params()[key]
                for key in _XGB_SEARCH
            }
            selected_parameters["best_iteration"] = int(estimator.best_iteration)
            selected_parameters["missing_values"] = "native_missing_branch"
        metrics[model_name] = utils.json_safe({
            "model": model_name,
            "estimator": estimator_type,
            "feature_profile": profile,
            "features": arms[profile],
            "feature_count": len(arms[profile]),
            "selected_parameters": selected_parameters,
            "validation_auc": float(validation_auc),
            "test": {**test_metrics, "n": len(y["test"])},
        })
        test_frame = split_frames["test"]
        for credit_id, segment, y_true, score in zip(
            test_frame["credito_id_anon"],
            test_frame["segmento"],
            y["test"],
            probability,
        ):
            predictions.append({
                "evaluation_id": str(credit_id),
                "model": model_name,
                "feature_profile": profile,
                "mode": "trained",
                "split": "test",
                "segment": str(segment),
                "y_true": int(y_true),
                "probability": float(score),
            })
    return (
        pd.DataFrame(predictions),
        metrics,
        pd.DataFrame(logreg_coefficients).sort_values(
            "coeficiente_estandarizado"
        ).reset_index(drop=True),
        pd.DataFrame(xgb_shap_summary).sort_values(
            "shap_medio_absoluto", ascending=False
        ).reset_index(drop=True),
    )


def combine_intermediate_predictions(
    classic: pd.DataFrame,
    qwen_tu_form_zero: pd.DataFrame,
    qwen_tu_form_few8: pd.DataFrame,
    qwen_tu_form_description_zero: pd.DataFrame,
    qwen_tu_form_description_few8: pd.DataFrame,
) -> pd.DataFrame:
    """Une solo predicciones compatibles con el nuevo contrato experimental."""
    required = [
        "evaluation_id", "model", "feature_profile", "mode", "split",
        "segment", "y_true", "probability",
    ]
    parts = [classic, qwen_tu_form_zero, qwen_tu_form_few8,
             qwen_tu_form_description_zero, qwen_tu_form_description_few8]
    normalized = []
    for part in parts:
        missing = [column for column in required if column not in part.columns]
        if missing:
            raise ValueError(f"Predicciones sin contrato intermedio: {missing}")
        normalized.append(part[required].copy())
    result = pd.concat(normalized, ignore_index=True)
    result["evaluation_id"] = result["evaluation_id"].astype(str)
    result["probability"] = pd.to_numeric(result["probability"], errors="coerce")
    observed = set(zip(result["model"], result["feature_profile"], result["mode"]))
    if observed != ALLOWED:
        raise ValueError(
            f"Configuraciones incompatibles con la entrega: {sorted(observed ^ ALLOWED)}"
        )
    key = ["evaluation_id", "model", "feature_profile", "mode", "split"]
    if result.duplicated(key).any():
        raise ValueError("Hay predicciones duplicadas en el contrato intermedio")

    configurations = result[["model", "feature_profile", "mode"]].apply(tuple, axis=1)
    reference_key = ("logreg_tu_form", "tu_form", "trained")
    reference = result[configurations.map(lambda value: value == reference_key)].set_index("evaluation_id")[
        ["segment", "y_true"]
    ].sort_index()
    if reference.empty:
        raise ValueError("Falta el conjunto de referencia para validar las predicciones")
    for configuration in ALLOWED:
        current = result[configurations.map(lambda value: value == configuration)].set_index("evaluation_id")[
            ["segment", "y_true"]
        ].sort_index()
        if not current.equals(reference):
            raise ValueError(
                "Las configuraciones no contienen exactamente los mismos créditos, "
                f"segmentos y etiquetas: {configuration}"
            )
    return result.sort_values(key).reset_index(drop=True)


def _stratified_bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    parts = []
    for target in np.unique(y_true):
        indices = np.flatnonzero(y_true == target)
        parts.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(parts)


def _bootstrap_rng(seed: int, *labels: str) -> np.random.Generator:
    """Generador estable por comparación, independiente del orden de las filas."""
    payload = "|".join([str(seed), *map(str, labels)]).encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(derived)


def build_intermediate_metrics_table(
    predictions: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Métricas e intervalo de AUC para las seis configuraciones autorizadas."""
    from sklearn.metrics import roc_auc_score

    seed = int(params.get("seed", 42))
    n_boot = int(params.get("bootstrap_samples", 2000))
    test = predictions[predictions["split"].eq("test")]
    rows = []
    for keys, group in test.groupby(["model", "feature_profile", "mode"], dropna=False):
        if keys not in ALLOWED:
            raise ValueError(f"Configuración no autorizada en métricas: {keys}")
        for segment in ("total", "esparso", "denso"):
            all_rows = group if segment == "total" else group[group["segment"].eq(segment)]
            subset = all_rows[all_rows["probability"].notna()]
            if len(subset) < 2 or subset["y_true"].nunique() < 2:
                continue
            y_true = subset["y_true"].to_numpy(dtype=int)
            probability = subset["probability"].to_numpy(dtype=float)
            rng = _bootstrap_rng(seed, *keys, segment, "auc")
            auc_samples = []
            for _ in range(n_boot):
                indices = _stratified_bootstrap_indices(y_true, rng)
                auc_samples.append(roc_auc_score(
                    y_true[indices], probability[indices]
                ))
            rows.append({
                "model": keys[0],
                "feature_profile": keys[1],
                "mode": keys[2],
                "segment": segment,
                "n": len(subset),
                "n_total": len(all_rows),
                "n_valid": len(subset),
                "n_invalid": len(all_rows) - len(subset),
                **utils.credit_metrics(y_true, probability),
                "AUC_ci_low": float(np.quantile(auc_samples, .025)),
                "AUC_ci_high": float(np.quantile(auc_samples, .975)),
            })
    result = pd.DataFrame(rows)
    observed_total = set(zip(
        result.loc[result["segment"].eq("total"), "model"],
        result.loc[result["segment"].eq("total"), "feature_profile"],
        result.loc[result["segment"].eq("total"), "mode"],
    ))
    if observed_total != ALLOWED:
        raise ValueError("La tabla de métricas no contiene las seis configuraciones")
    return result


def build_qwen_description_comparisons(
    predictions: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Cambio pareado de AUC al sumar la descripción libre a Qwen."""
    from sklearn.metrics import roc_auc_score

    seed = int(params.get("seed", 42))
    n_boot = int(params.get("bootstrap_samples", 2000))
    rows = []
    qwen_all = predictions[
        predictions["model"].eq("qwen3:8b")
        & predictions["split"].eq("test")
    ]
    for mode in ("zero", "few"):
        mode_all = qwen_all[qwen_all["mode"].eq(mode)]
        group = mode_all[mode_all["probability"].notna()]
        left = group[group["feature_profile"].eq("tu_form")][
            ["evaluation_id", "segment", "y_true", "probability"]
        ].rename(columns={"probability": "probability_base"})
        right = group[group["feature_profile"].eq("tu_form_description")][
            ["evaluation_id", "y_true", "probability"]
        ].rename(columns={"y_true": "y_true_text", "probability": "probability_text"})
        paired = left.merge(right, on="evaluation_id", validate="one_to_one")
        if paired.empty or not paired["y_true"].equals(paired["y_true_text"]):
            raise ValueError(f"Predicciones de Qwen desalineadas en modalidad {mode}")
        for segment in ("total", "esparso", "denso"):
            subset = paired if segment == "total" else paired[paired["segment"].eq(segment)]
            expected = mode_all[mode_all["feature_profile"].eq("tu_form")]
            if segment != "total":
                expected = expected[expected["segment"].eq(segment)]
            if len(subset) < 2 or subset["y_true"].nunique() < 2:
                continue
            y_true = subset["y_true"].to_numpy(dtype=int)
            base = subset["probability_base"].to_numpy(dtype=float)
            text = subset["probability_text"].to_numpy(dtype=float)
            rng = _bootstrap_rng(seed, "qwen3:8b", mode, segment, "paired_delta")
            deltas = []
            for _ in range(n_boot):
                indices = _stratified_bootstrap_indices(y_true, rng)
                deltas.append(
                    roc_auc_score(y_true[indices], text[indices])
                    - roc_auc_score(y_true[indices], base[indices])
                )
            auc_base = roc_auc_score(y_true, base)
            auc_text = roc_auc_score(y_true, text)
            rows.append({
                "comparison": f"qwen3:8b_{mode}",
                "segment": segment,
                "n": len(subset),
                "n_total": len(expected),
                "n_paired_valid": len(subset),
                "n_excluded_invalid": len(expected) - len(subset),
                "auc_without_description": auc_base,
                "auc_with_description": auc_text,
                "delta_auc": auc_text - auc_base,
                "ci_low": float(np.quantile(deltas, .025)),
                "ci_high": float(np.quantile(deltas, .975)),
                "bootstrap_samples": n_boot,
            })
    return pd.DataFrame(rows)


def summarize_text_fields(model_input: pd.DataFrame) -> pd.DataFrame:
    """Describe presencia y longitud sin exponer contenido privado."""
    names = {
        "descripcion_negocio": "Descripción del negocio",
        "subcategoria_texto": "Subcategoría",
        "otra_categoria_negocio": "Rubro declarado",
        "tipo_credito": "Tipo de crédito",
    }
    rows = []
    for col, label in names.items():
        values = model_input[col].fillna("").astype(str).str.strip()
        present = values.ne("")
        rows.append({
            "campo": label,
            "n_total": int(len(values)),
            "n_con_dato": int(present.sum()),
            "cobertura": float(present.mean()),
            "mediana_caracteres": float(values[present].str.len().median()),
            "valores_distintos": int(values[present].nunique()),
        })
    return pd.DataFrame(rows)


def build_intermediate_feature_contract(
    model_input: pd.DataFrame,
    delivery_params: dict,
) -> dict:
    """Materializa y valida el universo exacto de variables del experimento."""
    configured = delivery_params["feature_contract"]
    transunion = utils.intermediate_feature_columns(model_input, "tu", configured)
    structured = utils.intermediate_feature_columns(model_input, "tu_form", configured)
    form_direct = structured[len(transunion):]
    free_text = str(configured.get("free_text", ""))
    if free_text != "descripcion_negocio":
        raise ValueError("El único texto libre autorizado es descripcion_negocio")
    if len(transunion) != 20 or len(form_direct) != 9 or len(structured) != 29:
        raise ValueError("El contrato debe ser 20 TransUnion + 9 formulario = 29 variables")
    banned = set(
        utils.META + utils.TEXT + utils.DERIVED + utils.LEAK_COLS
        + utils.SCORES_INTERNOS + utils.TEMPORAL_PROXY_COLS
    )
    if set(structured) & banned:
        raise ValueError("El bloque estructurado contiene columnas prohibidas")
    return {
        "contract_name": "intermediate_tu20_form9_description1",
        "target": "mora mayor de 60 días dentro de 150 días",
        "target_positive_value": 1,
        "transunion_count": 20,
        "transunion": transunion,
        "form_direct_count": 9,
        "form_direct": form_direct,
        "structured_count": 29,
        "structured": structured,
        "free_text_count": 1,
        "free_text": free_text,
        "profiles": {
            "logreg_tu_form": structured,
            "xgb_tu_form": structured,
            "qwen_tu_form": structured,
            "qwen_tu_form_description": structured + [free_text],
        },
    }


def build_transunion_dictionary(dictionary: pd.DataFrame) -> pd.DataFrame:
    """Conserva los veinte atributos usados y su descripción oficial completa."""
    source = dictionary.copy()
    source["codigo"] = source["codigo"].astype(str).str.strip().str.lower()
    if source["codigo"].duplicated().any():
        duplicates = source.loc[source["codigo"].duplicated(), "codigo"].tolist()
        raise ValueError(f"Códigos duplicados en el diccionario TransUnion: {duplicates}")
    missing = sorted(set(TU_CODES) - set(source["codigo"]))
    if missing:
        raise ValueError(f"Faltan atributos TransUnion en el diccionario: {missing}")
    columns = [
        "codigo", "definicion_oficial_CreditVision", "uso",
        "ventana_tiempo", "tipo_valor",
    ]
    out = source.set_index("codigo").loc[TU_CODES].reset_index()[columns]
    if out[["definicion_oficial_CreditVision", "uso", "tipo_valor"]].isna().any().any():
        raise ValueError("El diccionario TransUnion contiene definiciones incompletas")
    return out


def build_real_serialization_example(
    model_input: pd.DataFrame,
    delivery_params: dict,
) -> dict:
    """Genera un ejemplo real y reproducible sin exponer la etiqueta en el reporte.

    El registro se identifica mediante una clave anonimizada fijada en parámetros. La
    serialización de Qwen se construye con la misma función utilizada por el experimento.
    Se incluyen exactamente las 29 variables del perfil estructurado de Qwen y, en
    el segundo perfil, la descripción libre del negocio.
    """
    record_id = str(delivery_params["serialization_example_id"])
    matches = model_input.loc[model_input["credito_id_anon"].astype(str).eq(record_id)]
    if len(matches) != 1:
        raise ValueError(
            "El ejemplo de serialización debe identificar exactamente un registro; "
            f"se encontraron {len(matches)} para {record_id!r}"
        )

    frame = model_input.sort_values(["fecha_desembolso", "credito_id_anon"]).copy()
    row = matches.iloc[0]
    if str(row["set"]) != "train":
        raise ValueError("El ejemplo real debe provenir del conjunto de entrenamiento")

    description = row.get("descripcion_negocio")
    if description is None or pd.isna(description) or not str(description).strip():
        raise ValueError("El ejemplo real debe tener una descripción del negocio")

    feature_names = utils.intermediate_feature_columns(
        frame, "tu_form", delivery_params["feature_contract"]
    )
    structured = utils.serialize_intermediate_profile(row, feature_names, "tu_form")
    structured_with_description = utils.serialize_intermediate_profile(
        row, feature_names, "tu_form_description"
    )
    messages = utils.build_messages_intermediate(
        row,
        feature_names,
        "tu_form_description",
        examples=None,
        prompt_variant="minimum",
    )

    return utils.json_safe({
        "source_dataset": "model_input_table",
        "record_is_real_and_anonymized": True,
        "record_key": record_id,
        "source_split": str(row["set"]),
        "target_excluded_from_report_and_prompt": True,
        "structured_features_in_order": feature_names,
        "structured_values": {
            column: None if pd.isna(row.get(column)) else row.get(column)
            for column in feature_names
        },
        "free_text_field": "descripcion_negocio",
        "free_text_value": str(description).strip(),
        "structured_feature_count": len(feature_names),
        "qwen_structured_serialization": structured,
        "qwen_structured_description_serialization": structured_with_description,
        "qwen_structured_description_character_count": len(structured_with_description),
        "qwen_system_message": messages[0]["content"],
        "qwen_user_message": messages[-1]["content"],
    })
