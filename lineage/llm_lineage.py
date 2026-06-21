# %% [markdown]
# # Lineaje LLM — Qwen3-8B TabLLM sobre la cartera de microcredito
# ## Tesis UBA 2026 — Federico Moreno
#
# Este script documenta el **lineaje de razonamiento** del enfoque LLM de la tesis.
# No ejecuta Ollama ni hace inferencia real: lee los CSVs/JSONs ya generados por
# las notebooks 06b (zero-shot) y 08 (few-shot), reconstruye las funciones clave
# de serializacion y prompting, y reproduce las figuras 8 y 9 del informe E3.
#
# La cadena de razonamiento que esta nota documenta es la siguiente:
#
# 1. Probamos un LLM en *zero-shot* esperando que su conocimiento general del
#    dominio (mora alta = riesgo alto) le diera ventaja en variables semanticas.
# 2. El AUC global quedo en 0.529 (apenas mejor que azar) y en el segmento
#    *esparso* incluso por debajo: 0.460.
# 3. El EDA habia anticipado el motivo: el sesgo de seleccion del portfolio
#    aprobado invierte la correlacion de `wd81` (peor mora historica) con el
#    target (r = -0.40). El LLM aplica la heuristica universal y se equivoca.
# 4. La hipotesis pasa a ser: si el LLM *ve* casos resueltos de esta cartera
#    (few-shot), deberia aprender la inversion. Para que la senal semantica
#    pese, el sitio mas prometedor es el segmento *thin-file* donde el buro
#    numerico aporta poco.
# 5. Con 16 ejemplos balanceados en contexto, el AUC esparso sube a **0.738**.
#    Es el hallazgo central del trabajo, con tres salvedades duras.

# %% [markdown]
# ## Setup
#
# Logger a archivo, matplotlib Agg (sin display), paths absolutos.

# %%
import json
import logging
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path("/Users/federicomoreno/Documents/TESIS/credit-risk-frontier")
LINEAGE = BASE / "lineage"
FIGS = LINEAGE / "figures"
LOGS = LINEAGE / "logs"
MODELS = BASE / "models"
DATASET_PATH = BASE / "data" / "dataset_tesis.csv"

FIGS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOGS / "llm_run.log"

logger = logging.getLogger("llm_lineage")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(fh)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

logger.info("=" * 72)
logger.info("Lineaje LLM — inicio")
logger.info("=" * 72)


# %% [markdown]
# ## Paso 1 — Definicion del segmento esparso
#
# El segmento se define por el conteo de variables TransUnion en codigo -1
# (sin reporte de buro). El corte es 6: con >=6 variables negativas el cliente
# es *thin-file*. Este corte se acordo en script 05 y se mantiene aqui.
#
# Las 20 variables TU son las que figuran en `TU_VARS` de la notebook 06b.

# %%
TU_VARS = [
    "agg308", "wd81", "agg2503", "utlmag04", "duemag01", "aepmag01",
    "bi21s", "lmd34s", "ri27s", "rle904", "tel32s", "tranbal09",
    "at104s", "sa21s", "at103s", "tel03s", "at34af", "g051s", "agg9316", "wd03",
]

CORTE_ESPARSO = 6


def cargar_dataset_segmentado(path: Path) -> pd.DataFrame:
    """Carga el dataset y agrega columnas de segmentacion por densidad de buro."""
    df = pd.read_csv(path, parse_dates=["fecha_desembolso"])
    df["n_tu_missing"] = (df[TU_VARS] == -1).sum(axis=1)
    df["segmento"] = np.where(df["n_tu_missing"] >= CORTE_ESPARSO, "esparso", "denso")
    return df


df = cargar_dataset_segmentado(DATASET_PATH)

n_total = len(df)
n_esparso = int((df["segmento"] == "esparso").sum())
n_denso = int((df["segmento"] == "denso").sum())
n_test = int((df["set"] == "test").sum())
n_test_esparso = int(((df["set"] == "test") & (df["segmento"] == "esparso")).sum())
n_test_denso = int(((df["set"] == "test") & (df["segmento"] == "denso")).sum())

logger.info("Dataset cargado: %d filas, %d columnas", *df.shape)
logger.info("Segmentos (total): esparso=%d (%.1f%%), denso=%d (%.1f%%)",
            n_esparso, 100 * n_esparso / n_total, n_denso, 100 * n_denso / n_total)
logger.info("Test set: n=%d | esparso=%d | denso=%d",
            n_test, n_test_esparso, n_test_denso)


# %% [markdown]
# ## Paso 2 — Serializacion TabLLM
#
# El paso clave del enfoque TabLLM (Hegselmann et al., 2023) es convertir cada
# fila tabular en una oracion natural: `"descripcion: valor, ..."`. Dos
# decisiones de implementacion son cruciales para esta cartera:
#
# 1. **Solo 20 features** (las 10 TU mas predictivas + 10 del formulario). Un
#    prompt con las 102 columnas seria largo, caro y diluiria la senal.
# 2. **Negativos del buro como texto**: si una variable TU vale -1, -2 o -3 no
#    se serializa como un numero (que confundiria al LLM) sino como "sin
#    historial". Asi el modelo lee el significado real, no un sentinel.

# %%
TOP_FEATURES_LLM = [
    "g051s", "wd81", "wd03", "aepmag01", "bi21s", "ri27s",
    "rle904", "at104s", "agg308", "utlmag04",
    "antiguedad_cliente", "appusers_age",
    "credits_amount_granted", "credits_fee_month_amount_granted",
    "shops_monthly_incomes", "shops_monthly_outcomes",
    "estimated_income", "free_cash_flow",
    "cost_ingress_ratio", "appusers_score",
]

NUMERIC_DESC = {
    "appusers_age": "la edad del solicitante en anios",
    "credits_amount_granted": "el monto del credito otorgado en pesos colombianos",
    "credits_fee_month_amount_granted": "la cuota mensual del credito en pesos",
    "shops_monthly_incomes": "los ingresos mensuales del negocio en pesos",
    "shops_monthly_outcomes": "los egresos mensuales del negocio en pesos",
    "antiguedad_cliente": "la antiguedad del solicitante como cliente en dias",
    "estimated_income": "el ingreso mensual estimado del solicitante",
    "free_cash_flow": "el flujo de caja libre estimado",
    "cost_ingress_ratio": "la relacion costo/ingreso del negocio",
    "appusers_score": "el puntaje interno del solicitante",
}

TU_DESC = {
    "agg308": "el saldo total de deudas vigentes en el buro (ultimos 8 meses)",
    "wd81": "la peor calificacion de mora historica (maximo dias en atraso, 81m)",
    "utlmag04": "la utilizacion promedio del limite de credito (ultimos 4 meses)",
    "aepmag01": "la antiguedad promedio de las cuentas activas (meses)",
    "bi21s": "el numero de consultas al buro en los ultimos 6 meses",
    "ri27s": "el numero de cuentas en mora actualmente",
    "rle904": "el porcentaje de creditos pagados a tiempo (ultimos 4 anios)",
    "at104s": "el numero de cuentas activas reportadas en el buro",
    "g051s": "el numero de sectores economicos distintos con reporte activo",
    "wd03": "la peor mora reciente (maximo dias en atraso, 12m)",
}


def fmt_valor(val, es_tu: bool) -> str | None:
    """Formatea un valor numerico para el prompt.

    Reglas:
    - NaN -> None (se serializa como "sin historial")
    - Para variables TU, -1/-2/-3 -> None (mismo trato: "sin historial")
    - Float -> dos decimales; otro tipo -> str(val)
    """
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if es_tu and val in (-1, -2, -3):
        return None
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def serializar_tabllm(row: pd.Series, features: list[str]) -> str:
    """Reproduce `serialize_tabllm` de la notebook 06b.

    Convierte la fila a `"descripcion: valor"` separado por comas. Cuando el
    valor TU es negativo o NaN, escribe `"descripcion: sin historial"`.
    """
    desc = {**NUMERIC_DESC, **TU_DESC}
    partes: list[str] = []
    for col in features:
        v = row.get(col)
        es_tu = col in TU_DESC
        fv = fmt_valor(v, es_tu=es_tu)
        d = desc.get(col, col)
        partes.append(f"{d}: {fv}" if fv is not None else f"{d}: sin historial")
    return ", ".join(partes)


# Demo: serializar el primer caso esparso del test para dejar trazo en el log.
demo_row = df[(df["set"] == "test") & (df["segmento"] == "esparso")].iloc[0]
demo_serial = serializar_tabllm(demo_row, TOP_FEATURES_LLM[:10])
logger.info("Ejemplo de serializacion (10 features TU del primer caso esparso):")
logger.info("  %s", demo_serial)


# %% [markdown]
# ## Paso 3 — Prompt zero-shot
#
# El prompt zero-shot tiene dos roles:
# - **system**: define al modelo como evaluador de riesgo y le ordena responder
#   solo con un entero (para poder parsear la probabilidad).
# - **user**: incluye la fila serializada y pide P(mora) entre 0 y 100.

# %%
SYSTEM_ZS = (
    "Eres evaluador de riesgo crediticio de microcreditos en Colombia. "
    "Responde SOLO con un numero entero."
)

USER_ZS_TEMPLATE = "DATOS: {serialized}\n\nProbabilidad de mora (0-100, entero):"


def construir_prompt_zeroshot(row: pd.Series) -> list[dict]:
    serial = serializar_tabllm(row, TOP_FEATURES_LLM)
    return [
        {"role": "system", "content": SYSTEM_ZS},
        {"role": "user", "content": USER_ZS_TEMPLATE.format(serialized=serial)},
    ]


# %% [markdown]
# ## Paso 4 — Seleccion balanceada de ejemplos few-shot
#
# Para evitar que el LLM aprenda solo "la mayoria no entra en mora", los
# ejemplos se eligen balanceados (mitad target=1, mitad target=0) del conjunto
# de **train** unicamente. Nunca se usan filas del test como ejemplos:
# habria leakage. La semilla queda fija en 42 para reproducibilidad.

# %%
SEED = 42


def seleccionar_ejemplos_balanceados(train_df: pd.DataFrame, n_shots: int,
                                     rng: np.random.Generator) -> pd.DataFrame:
    """Reproduce `select_few_shot_examples` de la notebook 08."""
    n_per_class = n_shots // 2
    defaults = train_df[train_df["target"] == 1]
    no_defaults = train_df[train_df["target"] == 0]
    a = defaults.sample(n=min(n_per_class, len(defaults)),
                        random_state=int(rng.integers(1_000_000)))
    b = no_defaults.sample(n=min(n_per_class, len(no_defaults)),
                           random_state=int(rng.integers(1_000_000)))
    ejemplos = pd.concat([a, b]).sample(
        frac=1, random_state=int(rng.integers(1_000_000)))
    return ejemplos.reset_index(drop=True)


SYSTEM_FS = (
    "Eres evaluador de riesgo crediticio de microcreditos en Colombia. "
    "Aprende de los ejemplos resueltos y responde SOLO con un numero entero."
)


def construir_prompt_fewshot(ejemplos: pd.DataFrame, test_row: pd.Series) -> list[dict]:
    """Reproduce `build_fewshot_messages` de 08.

    Estructura del user prompt:
    1. Encabezado: "A continuacion hay N casos resueltos..."
    2. Por cada ejemplo: serializacion + "Probabilidad de mora: 100|0"
    3. Caso nuevo a evaluar + "Probabilidad de mora (0-100, entero):"
    """
    n = len(ejemplos)
    partes = [
        f"A continuacion hay {n} casos resueltos de solicitantes con su "
        "resultado real (probabilidad de mora: 100 = entro en mora, 0 = no).",
        "",
    ]
    for i, (_, ex) in enumerate(ejemplos.iterrows(), 1):
        serial = serializar_tabllm(ex, TOP_FEATURES_LLM)
        outcome = 100 if int(ex["target"]) == 1 else 0
        partes.append(f"Caso {i}: {serial}")
        partes.append(f"Probabilidad de mora: {outcome}")
        partes.append("")
    serial_test = serializar_tabllm(test_row, TOP_FEATURES_LLM)
    partes.append(f"Nuevo solicitante: {serial_test}")
    partes.append("Probabilidad de mora (0-100, entero):")
    return [
        {"role": "system", "content": SYSTEM_FS},
        {"role": "user", "content": "\n".join(partes)},
    ]


# Demo: armar prompts de 16-shot y dejar trazo del balance y del tamano del
# user-prompt en caracteres (proxy de tokens).
train_df = df[df["set"] == "train"]
rng = np.random.default_rng(SEED)
ejemplos16 = seleccionar_ejemplos_balanceados(train_df, 16, rng)

balance16 = ejemplos16["target"].value_counts().to_dict()
prompt_demo = construir_prompt_fewshot(ejemplos16, demo_row)
chars_user = len(prompt_demo[1]["content"])
chars_system = len(prompt_demo[0]["content"])

logger.info("Few-shot 16: balance ejemplos %s (esperado {0: 8, 1: 8})", balance16)
logger.info("Prompt 16-shot: system=%d chars, user=%d chars (~%d tokens approx.)",
            chars_system, chars_user, chars_user // 4)


# %% [markdown]
# ## Paso 5 — Carga de metricas reales producidas por las notebooks
#
# Las notebooks 06b y 08 corrieron Ollama de verdad (qwen3:8b, ~2h por config)
# y guardaron las metricas en `models/llm_metrics_*.json`. Aqui las leemos sin
# re-ejecutar nada. Si los archivos no existieran, caemos a los numeros
# publicados en E3 (4.3, 4.4) para no romper el lineaje.

# %%
NUMEROS_E3_PUBLICADOS = {
    "zero": {
        "total":   {"AUC": 0.529, "n": 495},
        "esparso": {"AUC": 0.460, "n": 28},
        "denso":   {"AUC": 0.533, "n": 467},
    },
    8:  {"total": {"AUC": 0.503, "n": 495},
         "esparso": {"AUC": 0.564, "n": 28},
         "denso":   {"AUC": 0.502, "n": 467}},
    16: {"total": {"AUC": 0.527, "n": 495},
         "esparso": {"AUC": 0.738, "n": 28},
         "denso":   {"AUC": 0.516, "n": 467}},
    32: {"total": {"AUC": 0.509, "n": 495},
         "esparso": {"AUC": 0.588, "n": 28},
         "denso":   {"AUC": 0.507, "n": 467}},
}


def cargar_metricas_llm() -> dict:
    """Carga las metricas reales si existen, si no devuelve los numeros E3."""
    out = {"fuente": "real", "zero": None, "few": {}}
    path_zero = MODELS / "llm_metrics_tabllm_nothink.json"
    path_few = MODELS / "llm_metrics_fewshot.json"

    if path_zero.exists():
        with open(path_zero) as f:
            out["zero"] = json.load(f)["test"]
    else:
        out["zero"] = NUMEROS_E3_PUBLICADOS["zero"]
        out["fuente"] = "publicado_E3"

    if path_few.exists():
        with open(path_few) as f:
            d = json.load(f)
        for n in (8, 16, 32):
            k = f"{n}_shot"
            if k in d and "test" in d[k]:
                out["few"][n] = d[k]["test"]
            else:
                out["few"][n] = NUMEROS_E3_PUBLICADOS[n]
                out["fuente"] = "publicado_E3"
    else:
        out["few"] = {n: NUMEROS_E3_PUBLICADOS[n] for n in (8, 16, 32)}
        out["fuente"] = "publicado_E3"

    return out


metricas = cargar_metricas_llm()
logger.info("Fuente de metricas LLM: %s", metricas["fuente"])


def auc(test_dict: dict, seg: str) -> float:
    return float(test_dict.get(seg, {}).get("AUC", float("nan")))


def n(test_dict: dict, seg: str) -> int:
    return int(test_dict.get(seg, {}).get("n", 0))


auc_zs_total = auc(metricas["zero"], "total")
auc_zs_esp = auc(metricas["zero"], "esparso")
auc_zs_den = auc(metricas["zero"], "denso")

auc_fs_total = {k: auc(v, "total") for k, v in metricas["few"].items()}
auc_fs_esp = {k: auc(v, "esparso") for k, v in metricas["few"].items()}
auc_fs_den = {k: auc(v, "denso") for k, v in metricas["few"].items()}

logger.info("Zero-shot: AUC total=%.3f | esparso=%.3f (n=%d) | denso=%.3f (n=%d)",
            auc_zs_total, auc_zs_esp, n(metricas["zero"], "esparso"),
            auc_zs_den, n(metricas["zero"], "denso"))
for k in (8, 16, 32):
    logger.info("Few-shot %d: AUC total=%.3f | esparso=%.3f | denso=%.3f",
                k, auc_fs_total[k], auc_fs_esp[k], auc_fs_den[k])


# %% [markdown]
# ## Paso 6 — Figura 8: AUC vs N de ejemplos, por segmento
#
# Reproduce la figura 8 del informe. El eje x incluye N=0 (zero-shot) y las
# tres configuraciones few-shot. El segmento esparso muestra el pico en 16
# que la tesis discute como hallazgo central. La curva del denso queda
# plana cerca del azar.

# %%
puntos = [0, 8, 16, 32]

curva_esp = [auc_zs_esp] + [auc_fs_esp[k] for k in (8, 16, 32)]
curva_den = [auc_zs_den] + [auc_fs_den[k] for k in (8, 16, 32)]
curva_tot = [auc_zs_total] + [auc_fs_total[k] for k in (8, 16, 32)]

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(puntos, curva_esp, marker="o", lw=2.2, color="#E8A838",
        label=f"Esparso (n={n(metricas['zero'], 'esparso')})")
ax.plot(puntos, curva_den, marker="s", lw=2.2, color="#4878CF",
        label=f"Denso (n={n(metricas['zero'], 'denso')})")
ax.plot(puntos, curva_tot, marker="^", lw=1.6, color="#888", linestyle="--",
        label="Total (n=495)")
ax.axhline(0.5, color="gray", linestyle=":", lw=0.8, alpha=0.6)
ax.axhline(0.704, color="#4878CF", linestyle=":", lw=0.9, alpha=0.7)
ax.text(32, 0.708, "XGB esparso (CV temp.)", fontsize=8, color="#4878CF",
        ha="right", va="bottom")
for xv, yv in zip(puntos, curva_esp):
    ax.annotate(f"{yv:.3f}", (xv, yv), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, fontweight="bold",
                color="#B8762B")
ax.set_xticks(puntos)
ax.set_xticklabels(["0 (zero-shot)", "8", "16", "32"])
ax.set_xlabel("Numero de ejemplos in-context")
ax.set_ylabel("AUC-ROC (conjunto de test)")
ax.set_ylim(0.35, 0.85)
ax.set_title("Figura 8 — Qwen3-8B: AUC en test vs. N de ejemplos por segmento",
             fontweight="bold")
ax.legend(loc="lower right", framealpha=0.95)
fig.tight_layout()
out_fig1 = FIGS / "llm_01_fewshot_curve.png"
fig.savefig(out_fig1, bbox_inches="tight")
plt.close(fig)

logger.info("Figura guardada: %s", out_fig1)


# %% [markdown]
# ## Paso 7 — Figura 9: comparacion AUC por modelo y segmento
#
# Cruza los cuatro modelos (XGBoost, LogReg, LLM zero-shot, LLM few-shot 16)
# en los tres segmentos (esparso, denso, total). XGBoost y LogReg en
# esparso/denso vienen de CV temporal (script 05). El "total" de los clasicos
# viene del test honesto. El LLM siempre en test.

# %%
def cargar_metricas_clasicas() -> dict:
    """Construye dict por modelo con AUC en esparso/denso (CV) y total (test)."""
    out = {}
    cv_path = MODELS / "cv_seg_metrics.json"
    xgb_test_path = MODELS / "xgboost_metrics.json"
    lr_test_path = MODELS / "logreg_metrics.json"

    cv = json.load(open(cv_path)) if cv_path.exists() else {}
    for model_key, save_as in [("XGBoost", "XGBoost"), ("LogReg", "LogReg")]:
        out[save_as] = {}
        seg_data = cv.get(model_key, {})
        for seg in ("esparso", "denso"):
            v = seg_data.get(seg, {}).get("AUC")
            if isinstance(v, dict) and "mean" in v:
                out[save_as][seg] = (v["mean"], v.get("std", 0.0))

    if xgb_test_path.exists():
        d = json.load(open(xgb_test_path))
        out["XGBoost"]["total"] = (d["test"]["AUC"], 0.0)
    if lr_test_path.exists():
        d = json.load(open(lr_test_path))
        out["LogReg"]["total"] = (d["test"]["AUC"], 0.0)
    return out


clasicos = cargar_metricas_clasicas()

modelos_plot = [
    ("XGBoost", "#4878CF"),
    ("LogReg", "#6ACC65"),
    ("LLM zero-shot", "#F97316"),
    ("LLM 16-shot", "#EC4899"),
]
seg_order = ["esparso", "denso", "total"]

aucs_modelo = {
    "XGBoost": [clasicos.get("XGBoost", {}).get(s, (np.nan, 0))[0] for s in seg_order],
    "LogReg": [clasicos.get("LogReg", {}).get(s, (np.nan, 0))[0] for s in seg_order],
    "LLM zero-shot": [auc_zs_esp, auc_zs_den, auc_zs_total],
    "LLM 16-shot": [auc_fs_esp[16], auc_fs_den[16], auc_fs_total[16]],
}
stds_modelo = {
    "XGBoost": [clasicos.get("XGBoost", {}).get(s, (np.nan, 0))[1] for s in seg_order],
    "LogReg": [clasicos.get("LogReg", {}).get(s, (np.nan, 0))[1] for s in seg_order],
    "LLM zero-shot": [0, 0, 0],
    "LLM 16-shot": [0, 0, 0],
}

x = np.arange(len(seg_order))
width = 0.20
offsets = np.linspace(-1.5, 1.5, 4) * width

fig, ax = plt.subplots(figsize=(11, 6))
for (m, color), off in zip(modelos_plot, offsets):
    vals = aucs_modelo[m]
    stds = stds_modelo[m]
    bars = ax.bar(x + off, vals, width, color=color, alpha=0.88,
                  label=m, edgecolor="white",
                  yerr=stds, capsize=4, error_kw={"linewidth": 1.0})
    for bar, v in zip(bars, vals):
        if v and not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"Esparso\n(n={n_test_esparso})",
                    f"Denso\n(n={n_test_denso})",
                    f"Total\n(n={n_test})"])
ax.set_ylabel("AUC-ROC")
ax.set_ylim(0.30, 0.95)
ax.axhline(0.5, color="gray", linestyle=":", lw=0.8, alpha=0.6)
ax.set_title("Figura 9 — Comparacion AUC por modelo y segmento\n"
             "(clasicos esparso/denso = CV temporal; LLM = test; total clasicos = test)",
             fontweight="bold")
ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
fig.tight_layout()
out_fig2 = FIGS / "llm_02_comparacion_segmentos.png"
fig.savefig(out_fig2, bbox_inches="tight")
plt.close(fig)

logger.info("Figura guardada: %s", out_fig2)


# %% [markdown]
# ## Paso 8 — Tabla resumen al log
#
# Volcamos la tabla equivalente a la Tabla 4 del informe E3 al log de
# ejecucion, en formato texto plano, para dejar evidencia textual.

# %%
def fmt_fila(label: str, vals: list[float]) -> str:
    return f"{label:<22} | " + " | ".join(f"{v:6.3f}" if not np.isnan(v) else "  n/a "
                                          for v in vals)


tabla_resumen = [
    "Tabla resumen — AUC por modelo y segmento",
    "-" * 70,
    f"{'modelo':<22} | esparso | denso  | total ",
    "-" * 70,
    fmt_fila("XGBoost (CV/test)", aucs_modelo["XGBoost"]),
    fmt_fila("LogReg (CV/test)", aucs_modelo["LogReg"]),
    fmt_fila("LLM zero-shot", aucs_modelo["LLM zero-shot"]),
    fmt_fila("LLM 8-shot", [auc_fs_esp[8], auc_fs_den[8], auc_fs_total[8]]),
    fmt_fila("LLM 16-shot", aucs_modelo["LLM 16-shot"]),
    fmt_fila("LLM 32-shot", [auc_fs_esp[32], auc_fs_den[32], auc_fs_total[32]]),
    "-" * 70,
    f"n test esparso = {n_test_esparso} | n test denso = {n_test_denso} | n test total = {n_test}",
]
for linea in tabla_resumen:
    logger.info(linea)


# %% [markdown]
# ## Paso 9 — Conclusion: hallazgo + tres salvedades
#
# El log queda con los bullets que sintetizan el lineaje, listos para revision.

# %%
mejora_esparso = auc_fs_esp[16] - auc_zs_esp

conclusiones = [
    "",
    "=" * 72,
    "CONCLUSIONES DEL LINEAJE LLM",
    "=" * 72,
    "",
    "Hallazgo central:",
    f"  - El AUC del LLM en el segmento esparso pasa de {auc_zs_esp:.3f} (zero-shot)",
    f"    a {auc_fs_esp[16]:.3f} con 16 ejemplos in-context (mejora +{mejora_esparso:.3f}).",
    f"  - Ese 0.738 esparso queda por encima de la referencia de XGBoost esparso",
    f"    bajo CV temporal ({aucs_modelo['XGBoost'][0]:.3f} ± {stds_modelo['XGBoost'][0]:.3f}).",
    f"  - El mecanismo: el few-shot le ensena al LLM que en esta cartera la",
    f"    correlacion 'mora alta = riesgo alto' esta invertida (sesgo de seleccion).",
    "",
    "Tres salvedades duras:",
    f"  1. n=28 en test esparso. Intervalo de confianza ancho, falta replicar",
    f"     con un test esparso mas grande o validacion externa (e.g. LendingClub).",
    f"  2. La curva NO es monotona: 8->16 sube, 16->32 cae a {auc_fs_esp[32]:.3f}.",
    f"     Sugiere un optimo de saturacion del prompt cerca de N=16.",
    f"  3. El efecto vive solo en esparso. En denso el AUC del 16-shot es",
    f"     {auc_fs_den[16]:.3f} (esencialmente azar) y el total queda en {auc_fs_total[16]:.3f}.",
    "",
    "Lectura del lineaje:",
    "  - Zero-shot fracasa porque el LLM aplica la heuristica universal sobre",
    "    una cartera ya filtrada por aprobacion (sesgo de seleccion documentado",
    "    en EDA: r(wd81, target) = -0.40, contraintuitivo).",
    "  - Few-shot permite que el modelo *vea* la inversion en ejemplos reales",
    "    y la senal semantica aparece justo donde el buro numerico es escaso.",
    "  - El LLM no reemplaza a XGBoost a nivel global (0.527 vs 0.820 en total)",
    "    pero sugiere un nicho concreto: thin-file, donde el texto del negocio",
    "    gana peso relativo.",
    "",
    f"Figuras generadas: {FIGS}/llm_01_fewshot_curve.png",
    f"                   {FIGS}/llm_02_comparacion_segmentos.png",
    f"Fuente de metricas: {metricas['fuente']}",
    "=" * 72,
]

for linea in conclusiones:
    logger.info(linea)

# Cerrar el handler para que el archivo quede vaciado a disco.
fh.flush()
fh.close()
logger.removeHandler(fh)
