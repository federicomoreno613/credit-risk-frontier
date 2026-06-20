# %% [markdown]
# # LLM Prompting v2 — Qwen3-8B TabLLM Serialization (via Ollama)
# ## Tesis: LLMs vs. ML Clásico — UBA 2026
#
# Versión mejorada respecto a 06_llm_inference.py:
# 1. **Serialización TabLLM** de las top features (no solo 10)
# 2. **Decodificación de one-hot** → texto categórico antes de serializar
# 3. **Dos modos**: non_thinking y thinking (via Ollama con qwen3:8b local)
# 4. **Score continuo** via probabilidad 0-100 en lugar de log-probs
#
# **Backend:** Ollama local (qwen3:8b ya descargado), inferencia via HTTP a
# `http://localhost:11434`. No requiere torch ni HuggingFace.
#
# **Por qué v1 dio AUC=0.43:** prompt narrativo con ~10 features, sin chat template,
# y el LLM aplica la heurística global "mora alta = riesgo" — invertida por el sesgo
# de selección del portfolio aprobado (r(wd81, target) = -0.38).
#
# **Validación honesta:** las métricas se evalúan sobre el **conjunto de test**
# (columna `set`), por segmento de bureau, igual que el resto de los modelos.
# No se usa k-fold barajado (que mezclaría temporalidades).
#
# **Referencias:**
# - Hegselmann et al. (2023) — TabLLM: Few-shot classification of tabular data
# - Ollama qwen3:8b — thinking mode via parámetro `think`

# %% [markdown]
# ## Setup

# %%
import json
import re
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

META = ["credito_id_anon", "fecha_desembolso", "target", "set"]
TEXT = ["subcategoria_texto", "descripcion_negocio", "otra_categoria_negocio", "tipo_credito"]

TU_VARS = [
    "agg308", "wd81", "agg2503", "utlmag04", "duemag01", "aepmag01",
    "bi21s", "lmd34s", "ri27s", "rle904", "tel32s", "tranbal09",
    "at104s", "sa21s", "at103s", "tel03s", "at34af", "g051s", "agg9316", "wd03",
]

# Ollama config
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"

CORTE_ESPARSO = 6
MIN_SEG_CASOS = 20
SAVE_EVERY    = 50    # guardar cache cada N requests
TIMEOUT_SEC   = 45    # timeout por request Ollama

# Top features para serialización — solo las 20 más predictivas (EDA point-biserial)
TOP_FEATURES_LLM = [
    # TransUnion (top predictores del buró)
    "g051s", "wd81", "wd03", "aepmag01", "bi21s", "ri27s",
    "rle904", "at104s", "agg308", "utlmag04",
    # Formulario (top predictores del negocio y solicitante)
    "antiguedad_cliente", "appusers_age",
    "credits_amount_granted", "credits_fee_month_amount_granted",
    "shops_monthly_incomes", "shops_monthly_outcomes",
    "estimated_income", "free_cash_flow",
    "cost_ingress_ratio", "appusers_score",
]

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


# %% [markdown]
# ## Diccionario de descripciones (TabLLM)

# %%
# Variables numéricas / continuas
NUMERIC_DESC = {
    "appusers_age":                          "la edad del solicitante en años",
    "credits_amount_granted":                "el monto del crédito otorgado en pesos colombianos",
    "credits_fee_month_amount_granted":      "la cuota mensual del crédito en pesos colombianos",
    "credits_interest_amount":               "el monto de intereses del crédito en pesos",
    "credits_surety_bond_amount":            "el monto del seguro/fianza en pesos",
    "credits_digital_instrumentation_amount":"el costo de instrumentación digital en pesos",
    "credits_dependants_amount":             "el número de dependientes del solicitante",
    "credits_family_expenses":               "los gastos familiares mensuales en pesos",
    "shops_monthly_incomes":                 "los ingresos mensuales del negocio en pesos",
    "shops_monthly_outcomes":                "los egresos mensuales del negocio en pesos",
    "shops_daily_incomes":                   "los ingresos diarios del negocio en pesos",
    "shops_initial_capital":                 "el capital inicial del negocio en pesos",
    "shops_rent_amount":                     "el arriendo mensual del negocio en pesos",
    "shops_shop_age":                        "la antigüedad del negocio en años",
    "antiguedad_cliente":                    "la antigüedad del solicitante como cliente en días",
    "estimated_income":                      "el ingreso mensual estimado del solicitante en pesos",
    "free_cash_flow":                        "el flujo de caja libre estimado en pesos",
    "cost_ingress_ratio":                    "la relación costo/ingreso del negocio (0=eficiente, 1=sin margen)",
    "debts_savings":                         "la relación deudas/ahorros del solicitante",
    "score_debets":                          "el puntaje de deudas del solicitante",
    "relacion_edad_deuda":                   "la relación entre edad del solicitante y sus deudas actuales",
    "appusers_score":                        "el puntaje interno del solicitante en el sistema",
}

# Variables TransUnion — -1 significa "sin información en el buró"
TU_DESC = {
    "agg308":    "el saldo total de deudas vigentes en el buró (últimos 8 meses) en pesos",
    "wd81":      "la peor calificación de mora histórica — máximo días en atraso (últimos 81 meses)",
    "agg2503":   "el monto de cartera castigada acumulada (deudas irrecuperables) en pesos",
    "utlmag04":  "la utilización promedio del límite de crédito en porcentaje (últimos 4 meses)",
    "duemag01":  "el monto vencido promedio — saldo atrasado (último mes) en pesos",
    "aepmag01":  "la antigüedad promedio de las cuentas activas en el buró en meses",
    "bi21s":     "el número de consultas al buró en los últimos 6 meses",
    "lmd34s":    "la mora de largo plazo — días promedio en mora (últimos 34 meses)",
    "ri27s":     "el número de cuentas en mora actualmente",
    "rle904":    "el porcentaje de créditos pagados a tiempo en los últimos 4 años",
    "tel32s":    "el límite total de crédito disponible en el buró en pesos",
    "tranbal09": "el balance transaccional (negativo=revolvente, positivo=transactor)",
    "at104s":    "el número de cuentas activas reportadas en el buró",
    "sa21s":     "el monto total de la línea de crédito aprobada en pesos",
    "at103s":    "el número total de cuentas (activas + cerradas) en el buró",
    "tel03s":    "la suma de límites de crédito en los últimos 3 meses en pesos",
    "at34af":    "el índice de apertura de nuevas cuentas de financiamiento",
    "g051s":     "el número de sectores económicos distintos con reporte activo en el buró",
    "agg9316":   "el saldo promedio de deudas en los últimos 16 trimestres en pesos",
    "wd03":      "la peor calificación de mora reciente — máximo días en atraso (últimos 12 meses)",
}

# Mapeos para decodificar one-hot → texto categórico
CATEGORY_MAP = {
    "category_agricultura_y_granja":      "Agricultura y Granja",
    "category_artesanias_y_arte":         "Artesanías y Arte",
    "category_bar_y_licores":             "Bar y Licores",
    "category_belleza_e_higiene":         "Belleza e Higiene",
    "category_carniceria_y_avicola":      "Carnicería y Avícola",
    "category_comercializacion":          "Comercialización",
    "category_confeccion":                "Confección",
    "category_drogueria_y_medicina":      "Droguería y Medicina",
    "category_electronica_y_tecnologia":  "Electrónica y Tecnología",
    "category_mercado_y_perecederos":     "Mercado y Perecederos",
    "category_miscelanea_y_papeleria":    "Miscelánea y Papelería",
    "category_restaurantes_y_comidas":    "Restaurantes y Comidas",
    "category_servicios":                 "Servicios",
    "category_servicios_y_alquiler":      "Servicios y Alquiler",
    "category_transporte_y_logistica":    "Transporte y Logística",
}

GOAL_MAP = {
    "credits_credit_goal_labor_payment":          "Pago de mano de obra",
    "credits_credit_goal_machinery_and_equipment":"Maquinaria y equipos",
    "credits_credit_goal_marketing_and_advertising":"Marketing y publicidad",
    "credits_credit_goal_nan":                    "No especificado",
    "credits_credit_goal_payment_suppliers":      "Pago a proveedores",
    "credits_credit_goal_stocking":               "Inventario/mercancía",
    "credits_credit_goal_supplies":               "Insumos y materiales",
}

EDUCATION_MAP = {
    "credits_contact_studies_None":                  "No especificado",
    "credits_contact_studies_college":               "Universitario (en curso)",
    "credits_contact_studies_high_school":           "Bachillerato",
    "credits_contact_studies_nan":                   "No especificado",
    "credits_contact_studies_postgraduate_studies":  "Posgrado",
    "credits_contact_studies_primary":               "Primaria",
    "credits_contact_studies_professional":          "Profesional universitario",
    "credits_contact_studies_technical":             "Técnico o tecnólogo",
}

GENDER_MAP = {
    "appusers_gender_None":   "No especificado",
    "appusers_gender_female": "Femenino",
    "appusers_gender_male":   "Masculino",
    "appusers_gender_other":  "Otro",
}

CANAL_COLS   = [f"CANAL_{n:02d}" for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
ALIANZA_COLS = [f"ALIANZA_{n:02d}" for n in [1, 2, 3, 4, 5, 6]]


def _decode_onehot(row: pd.Series, col_map: dict, label: str) -> str:
    """Decodifica grupo de one-hot a su valor categórico."""
    for col, val_label in col_map.items():
        if col in row and row[col] == 1:
            return val_label
    return "No especificado"


def _decode_canal(row: pd.Series) -> str:
    """Decodifica CANAL_XX (anónimo, mantener código)."""
    for col in CANAL_COLS:
        if col in row and row[col] == 1:
            return col.replace("_", " ")
    return "No especificado"


def _decode_alianza(row: pd.Series) -> str:
    """Decodifica ALIANZA_XX (anónimo, mantener código)."""
    for col in ALIANZA_COLS:
        if col in row and row[col] == 1:
            return col.replace("_", " ")
    return "Sin alianza"


# %% [markdown]
# ## Serialización TabLLM

# %%
def _fmt_val(val, is_tu: bool = False) -> str | None:
    """Formatea un valor numérico para el prompt. Retorna None si es desconocido/NaN."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if is_tu and (val == -1 or val == -2 or val == -3):
        return None  # sin información en el buró
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def serialize_tabllm(row: pd.Series, feature_names: list[str]) -> str:
    """Serialización compacta TabLLM: `descripción: valor` separadas por coma.

    Incluye las features numéricas/TU, los grupos one-hot decodificados y el texto
    semántico. Diseñada para TOP_FEATURES_LLM (~20 features → ~300 tokens).
    """
    all_desc = {**NUMERIC_DESC, **TU_DESC}
    parts = []
    for col in feature_names:
        v = row.get(col)
        is_tu = col in TU_DESC
        fv = _fmt_val(v, is_tu=is_tu)
        desc = all_desc.get(col, col)
        parts.append(f"{desc}: {fv}" if fv is not None else f"{desc}: sin historial")

    parts.append(f"categoría negocio: {_decode_onehot(row, CATEGORY_MAP, 'categoría')}")
    parts.append(f"objetivo crédito: {_decode_onehot(row, GOAL_MAP, 'objetivo')}")
    parts.append(f"educación: {_decode_onehot(row, EDUCATION_MAP, 'educación')}")
    parts.append(f"canal: {_decode_canal(row)}")

    for text_col, label in [
        ("subcategoria_texto", "subcategoría"),
        ("descripcion_negocio", "negocio"),
        ("tipo_credito", "tipo crédito"),
    ]:
        v = row.get(text_col, "")
        if v and not (isinstance(v, float) and pd.isna(v)):
            parts.append(f"{label}: {v}")
    return ", ".join(parts)


# %% [markdown]
# ## Construcción del prompt y llamada a Ollama

# %%
SYSTEM_BASE = (
    "Eres evaluador de riesgo crediticio de microcréditos en Colombia. "
    "Responde SOLO con un número entero."
)

USER_TEMPLATE = """\
DATOS: {serialized}

Probabilidad de mora (0-100, entero):"""


def _build_messages(row: pd.Series, feature_names: list[str], think: bool) -> list[dict]:
    """Construye los mensajes para la API de Ollama (chat format)."""
    serialized = serialize_tabllm(row, feature_names)
    return [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user",   "content": USER_TEMPLATE.format(serialized=serialized)},
    ]


def _parse_prob(text: str) -> float:
    """Extrae P(mora) 0-100 de la respuesta (elimina <think>...</think>) → float [0,1]."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for n in re.findall(r"\b([0-9]{1,3})\b", clean):
        val = int(n)
        if 0 <= val <= 100:
            return val / 100.0
    return float("nan")


def call_ollama(messages: list[dict], think: bool, retries: int = 3) -> float:
    """Llama a Ollama y retorna P(mora) como float [0, 1].

    think=False → respuesta directa, num_predict=20; think=True → razona, num_predict=512.
    """
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "think":    think,
        "options":  {"temperature": 0.0, "num_predict": 512 if think else 20},
    }
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEC)
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"]
            if not content.strip() and think:
                content = data["message"].get("thinking", "")
            return _parse_prob(content)
        except (requests.RequestException, KeyError, json.JSONDecodeError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return float("nan")


def check_ollama() -> None:
    """Verifica que Ollama esté corriendo y que el modelo esté disponible."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
    except requests.ConnectionError as exc:
        raise RuntimeError("Ollama no está corriendo. Iniciarlo con: ollama serve") from exc
    if OLLAMA_MODEL not in available:
        raise RuntimeError(
            f"Modelo '{OLLAMA_MODEL}' no encontrado en Ollama. "
            f"Disponibles: {available}. Instalar con: ollama pull {OLLAMA_MODEL}"
        )


# %% [markdown]
# ## Carga de datos y segmentación

# %%
def load_segmented(dataset_path: str) -> pd.DataFrame:
    """Carga el dataset y agrega columnas de segmentación por densidad de bureau."""
    df = pd.read_csv(dataset_path, parse_dates=["fecha_desembolso"])
    df["n_tu_missing"] = (df[TU_VARS] == -1).sum(axis=1)
    df["segmento"] = (df["n_tu_missing"] >= CORTE_ESPARSO).map({True: "esparso", False: "denso"})
    return df


# %% [markdown]
# ## Inferencia via Ollama con cache incremental
#
# El cache es por `credito_id_anon`: si el script se interrumpe, retoma desde
# donde quedó. Cada request es individual (Ollama no tiene batching real).

# %%
def run_inference_with_cache(df: pd.DataFrame, cache_path: Path, think: bool) -> np.ndarray:
    """Ejecuta inferencia en todo el dataset con cache incremental en `cache_path`."""
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
        cached_ids = set(cache["credito_id_anon"])
    else:
        cache = pd.DataFrame(columns=["credito_id_anon", "prob_llm"])
        cached_ids = set()

    pending_idx = [i for i in df.index if df.loc[i, "credito_id_anon"] not in cached_ids]

    if pending_idx:
        new_rows = []
        for req_num, i in enumerate(pending_idx, 1):
            row = df.loc[i]
            messages = _build_messages(row, TOP_FEATURES_LLM, think=think)
            prob = call_ollama(messages, think=think)
            new_rows.append({"credito_id_anon": row["credito_id_anon"], "prob_llm": prob})
            if req_num % SAVE_EVERY == 0 or req_num == len(pending_idx):
                cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
                cache.to_parquet(cache_path, index=False)
                new_rows = []

    # Deduplicar por id (re-runs previos pueden dejar filas repetidas en el cache)
    cache = cache.drop_duplicates("credito_id_anon", keep="last")
    merged = df[["credito_id_anon"]].merge(cache, on="credito_id_anon", how="left")
    return merged["prob_llm"].values


# %% [markdown]
# ## Evaluación honesta sobre el conjunto de test

# %%
def credit_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcula AUC, Gini, KS, Brier y PR-AUC."""
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    brier = brier_score_loss(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks, "Brier": brier, "PR_AUC": prauc}


def evaluate_on_test(df: pd.DataFrame, probs: np.ndarray) -> dict:
    """Evalúa las probabilidades sobre el conjunto de test, total y por segmento.

    Retorna {segmento: {métricas, n}}. Omite segmentos con <MIN_SEG_CASOS casos.
    """
    probs = np.asarray(probs, dtype=float)  # el merge puede devolver dtype object
    test_mask = (df["set"] == "test").values & ~np.isnan(probs)
    y = df["target"].values
    segmentos = df["segmento"].values

    summary = {}
    y_te, p_te = y[test_mask], probs[test_mask]
    summary["total"] = {**credit_metrics(y_te, p_te), "n": int(test_mask.sum())}
    for seg in ("esparso", "denso"):
        mask = test_mask & (segmentos == seg)
        if mask.sum() >= MIN_SEG_CASOS:
            summary[seg] = {**credit_metrics(y[mask], probs[mask]), "n": int(mask.sum())}
        else:
            summary[seg] = {"n": int(mask.sum()), "note": "insuficiente para AUC confiable"}
    return summary


# %% [markdown]
# ## Inferencia + evaluación end-to-end (nodo Kedro)
#
# `run_llm_inference` recibe rutas y un `mode` ("nothink" / "think"), ejecuta la
# inferencia con cache y devuelve las métricas honestas en test por segmento.

# %%
def run_llm_inference(dataset_path: str, cache_path: str, mode: str, output_dir: str) -> dict:
    """Ejecuta inferencia LLM TabLLM y evalúa en test.

    Args:
        dataset_path: ruta al CSV del dataset.
        cache_path: ruta al parquet de cache incremental (por modo).
        mode: "nothink" (respuesta directa) o "think" (reasoning mode).
        output_dir: directorio donde guardar `llm_metrics_tabllm_<mode>.json`.

    Returns:
        dict con métricas en test (total y por segmento) más metadatos.
    """
    if mode not in ("nothink", "think"):
        raise ValueError(f"mode debe ser 'nothink' o 'think', recibido: {mode!r}")

    check_ollama()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_segmented(dataset_path)
    probs = run_inference_with_cache(df, Path(cache_path), think=(mode == "think"))
    summary = evaluate_on_test(df, probs)

    results = {
        "mode":       mode,
        "model":      OLLAMA_MODEL,
        "method":     "TabLLM zero-shot via Ollama — probabilidad 0-100, eval en test",
        "test":       summary,
        "n_features": len(TOP_FEATURES_LLM),
        "n_samples":  int(len(df)),
        "n_nan":      int(np.isnan(probs).sum()),
    }
    with open(output_path / f"llm_metrics_tabllm_{mode}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# %% [markdown]
# ## Figura — Comparación non-thinking vs thinking (si ambos modos existen)

# %%
def plot_modes_comparison(results_by_mode: dict, figs_dir: Path) -> None:
    """Barras de AUC en test por segmento para los modos disponibles."""
    seg_order = ["esparso", "denso", "total"]
    seg_labels = ["Bureau\nesparso", "Bureau\ndenso", "Total"]
    x = np.arange(len(seg_order))
    width = 0.3
    colors = {"nothink": "#D65F5F", "think": "#8B5CF6"}

    fig, ax = plt.subplots(figsize=(10, 5))
    modes = [m for m in ("nothink", "think") if m in results_by_mode]
    for offset, mode in enumerate(modes):
        test = results_by_mode[mode]["test"]
        aucs = [test.get(s, {}).get("AUC", 0) for s in seg_order]
        bars = ax.bar(x + (offset - 0.5) * width, aucs, width,
                      color=colors[mode], alpha=0.85,
                      label=f"Qwen3-8B {mode}", edgecolor="white")
        for bar, val in zip(bars, aucs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                        f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(seg_labels)
    ax.set_ylabel("AUC-ROC (conjunto de test)")
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    ax.set_title("Qwen3-8B: Non-thinking vs. Thinking — TabLLM zero-shot (test)",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "25_tabllm_nothink_vs_think.png", bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## Entry point

# %%
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    DATASET = str(BASE / "data" / "dataset_tesis.csv")
    MODELS = BASE / "models"
    FIGS = BASE / "figures"
    FIGS.mkdir(exist_ok=True)

    cache_paths = {
        "nothink": str(MODELS / "llm_probs_tabllm_nothink.parquet"),
        "think":   str(MODELS / "llm_probs_tabllm_think.parquet"),
    }

    results_by_mode = {}
    for mode in ("nothink", "think"):
        results_by_mode[mode] = run_llm_inference(
            dataset_path=DATASET,
            cache_path=cache_paths[mode],
            mode=mode,
            output_dir=str(MODELS),
        )

    plot_modes_comparison(results_by_mode, FIGS)
