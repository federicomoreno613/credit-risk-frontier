# %% [markdown]
# # LLM Few-Shot — Qwen3-8B con 8/16/32 ejemplos (via Ollama)
# ## Tesis: LLMs vs. ML Clásico — UBA 2026
#
# Extiende el zero-shot de 06b añadiendo ejemplos resueltos en el prompt.
# Diseño: n ejemplos balanceados (n/2 default, n/2 no-default) seleccionados del
# **conjunto de entrenamiento** (`set == "train"`) — nunca del test.
#
# **Backend:** Ollama local (qwen3:8b), igual que 06b. Se reutiliza la misma
# serialización TabLLM y la llamada a Ollama importándolas del módulo 06b, para
# mantener una única fuente de verdad.
#
# **Validación temporal (no StratifiedKFold).** Los ejemplos se toman del train.
# La métrica **de test** por segmento sigue siendo la honesta para comparar contra
# XGBoost/LogReg (scripts 03–06b). Así se evita el data leakage temporal del k-fold
# barajado que usaba la versión previa.
#
# **Métrica general (E4).** Como el LLM no distingue train/test, además puede puntuar
# train+val+test para reportar una métrica sobre todo el dataset. Para evitar la
# filtración (*leakage*), cada caso se excluye de sus propios ejemplos: si un caso
# coincide con un ejemplo base, ese ejemplo se sustituye por otro de la misma clase
# (ver `examples_for_case`). Se controla con el parámetro `scope` ("all" | "test").
#
# **Configuraciones:** n = 8, 16, 32 ejemplos.
#
# **Referencias:**
# - Hegselmann et al. (2023) — TabLLM: Few-shot classification of tabular data

# %% [markdown]
# ## Setup

# %%
import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N_SHOTS_LIST = [8, 16, 32]
SEED = 42

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


# %% [markdown]
# ## Reutilizar helpers de 06b
#
# El módulo 06b solo define funciones y constantes a nivel de módulo (no ejecuta
# inferencia al importarse), por lo que puede cargarse de forma segura. Esto evita
# duplicar la serialización TabLLM y la lógica de Ollama.

# %%
def _load_06b():
    """Carga 06b_llm_prompting.py como módulo (su nombre empieza con dígito)."""
    path = Path(__file__).resolve().parent / "06b_llm_prompting.py"
    spec = importlib.util.spec_from_file_location("llm_prompting_06b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# %% [markdown]
# ## Selección de ejemplos few-shot (balanceados, del train)

# %%
def select_few_shot_examples(train_df: pd.DataFrame, n_shots: int,
                             rng: np.random.Generator) -> pd.DataFrame:
    """Selecciona n_shots ejemplos balanceados del train (n/2 default, n/2 no)."""
    n_per_class = n_shots // 2
    defaults = train_df[train_df["target"] == 1]
    no_defaults = train_df[train_df["target"] == 0]
    sampled_def = defaults.sample(n=min(n_per_class, len(defaults)),
                                  random_state=int(rng.integers(1_000_000)))
    sampled_no = no_defaults.sample(n=min(n_per_class, len(no_defaults)),
                                    random_state=int(rng.integers(1_000_000)))
    examples = pd.concat([sampled_def, sampled_no]).sample(
        frac=1, random_state=int(rng.integers(1_000_000)))
    return examples.reset_index(drop=True)


# %% [markdown]
# ## Construcción del prompt few-shot

# %%
SYSTEM_FEWSHOT = (
    "Eres evaluador de riesgo crediticio de microcréditos en Colombia. "
    "Aprende de los ejemplos resueltos y responde SOLO con un número entero."
)


def build_fewshot_messages(m06b, examples: pd.DataFrame, test_row: pd.Series) -> list[dict]:
    """Construye los mensajes Ollama: ejemplos resueltos + caso a evaluar.

    Cada ejemplo muestra su serialización TabLLM y el desenlace real (mora 0/100).
    Reutiliza `serialize_tabllm` y `TOP_FEATURES_LLM` de 06b.
    """
    features = m06b.TOP_FEATURES_LLM
    n = len(examples)
    parts = [f"A continuación hay {n} casos resueltos de solicitantes con su "
             "resultado real (probabilidad de mora: 100 = entró en mora, 0 = no).", ""]
    for i, (_, ex) in enumerate(examples.iterrows(), 1):
        serialized = m06b.serialize_tabllm(ex, features)
        outcome = 100 if int(ex["target"]) == 1 else 0
        parts.append(f"Caso {i}: {serialized}")
        parts.append(f"Probabilidad de mora: {outcome}")
        parts.append("")
    test_serialized = m06b.serialize_tabllm(test_row, features)
    parts.append(f"Nuevo solicitante: {test_serialized}")
    parts.append("Probabilidad de mora (0-100, entero):")
    return [
        {"role": "system", "content": SYSTEM_FEWSHOT},
        {"role": "user",   "content": "\n".join(parts)},
    ]


# %% [markdown]
# ## Selección sin filtración (leakage) para inferir sobre todo el dataset
#
# Para reportar una **métrica general** el LLM tiene que puntuar train+val+test. El
# riesgo es la filtración (*leakage*): si un caso que se está evaluando aparece además
# como uno de sus propios ejemplos few-shot, el modelo "ve la respuesta" y la métrica
# queda inflada. Como los ejemplos base salen del train, ese choque solo puede ocurrir
# con casos del train.
#
# Solución: se mantiene un set de ejemplos **fijo** (mismo prompt para casi todos los
# casos, para no cambiar la configuración experimental) y, únicamente cuando el caso a
# evaluar coincide con uno de los ejemplos base, se **reemplaza ese ejemplo** por otro
# de la **misma clase** tomado de un pool de reemplazo del train (nunca el propio caso).
# Así cada caso se puntúa con exactamente `n_shots` ejemplos y jamás consigo mismo.

# %%
def build_replacement_pool(train_df: pd.DataFrame, base_examples: pd.DataFrame,
                           rng: np.random.Generator) -> dict:
    """Pool de reemplazo por clase: filas del train que NO están en los ejemplos base."""
    base_ids = set(base_examples["credito_id_anon"])
    pool = {}
    for cls in (0, 1):
        cand = train_df[(train_df["target"] == cls) &
                        (~train_df["credito_id_anon"].isin(base_ids))]
        pool[cls] = cand.sample(frac=1, random_state=int(rng.integers(1_000_000))
                                ).reset_index(drop=True)
    return pool


def examples_for_case(base_examples: pd.DataFrame, case_id: str, pool: dict) -> pd.DataFrame:
    """Devuelve `n_shots` ejemplos que excluyen `case_id` (garantía anti-leakage).

    Si `case_id` no está entre los ejemplos base, se devuelve `base_examples` intacto
    (el caso común: todo val/test y la gran mayoría del train). Si está, se sustituye
    esa fila por un ejemplo de la misma clase del pool de reemplazo cuyo id no sea el
    propio caso ni otro ejemplo ya presente.
    """
    ids = set(base_examples["credito_id_anon"])
    if case_id not in ids:
        return base_examples

    rows, used = [], set()
    for _, ex in base_examples.iterrows():
        if ex["credito_id_anon"] != case_id:
            rows.append(ex)
            continue
        cls = int(ex["target"])
        for _, cand in pool[cls].iterrows():
            cid = cand["credito_id_anon"]
            if cid == case_id or cid in ids or cid in used:
                continue
            used.add(cid)
            rows.append(cand)
            break
    return pd.DataFrame(rows).reset_index(drop=True)


# %% [markdown]
# ## Inferencia few-shot con cache incremental (sobre todo el dataset)

# %%
def run_fewshot_inference(m06b, df: pd.DataFrame, base_examples: pd.DataFrame,
                          pool: dict, cache_path: Path, scope: str = "all") -> np.ndarray:
    """Infieren P(mora) con cache incremental, sin filtración de datos.

    Args:
        scope: "all" → puntúa train+val+test (para la métrica general);
               "test" → solo `set == "test"` (comportamiento honesto original).

    Cada caso se puntúa con `examples_for_case`, que excluye el propio caso de sus
    ejemplos. El resto de las filas queda NaN.
    """
    if scope == "test":
        target_index = df[df["set"] == "test"].index
    else:
        target_index = df.index

    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
        cached_ids = set(cache["credito_id_anon"])
    else:
        cache = pd.DataFrame(columns=["credito_id_anon", "prob_llm"])
        cached_ids = set()

    pending = [i for i in target_index if df.loc[i, "credito_id_anon"] not in cached_ids]
    if pending:
        new_rows = []
        for req_num, i in enumerate(pending, 1):
            case_id = df.loc[i, "credito_id_anon"]
            examples = examples_for_case(base_examples, case_id, pool)
            messages = build_fewshot_messages(m06b, examples, df.loc[i])
            prob = m06b.call_ollama(messages, think=False)
            new_rows.append({"credito_id_anon": case_id, "prob_llm": prob})
            if req_num % m06b.SAVE_EVERY == 0 or req_num == len(pending):
                cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
                cache.to_parquet(cache_path, index=False)
                new_rows = []

    cache = cache.drop_duplicates("credito_id_anon", keep="last")
    merged = df[["credito_id_anon"]].merge(cache, on="credito_id_anon", how="left")
    return merged["prob_llm"].values


# %% [markdown]
# ## Evaluación general (todo el dataset) por segmento
#
# La métrica de **test** sigue siendo la honesta para comparar contra XGBoost/LogReg
# (esos modelos no pueden puntuar su propio train sin inflar el número). La métrica
# **general** existe porque el LLM no distingue train/test: puntúa cualquier fila igual.

# %%
def evaluate_general(m06b, df: pd.DataFrame, probs: np.ndarray) -> dict:
    """AUC/Gini/KS por segmento sobre todo el dataset (filas con prob no-NaN)."""
    probs = np.asarray(probs, dtype=float)
    valid = ~np.isnan(probs)
    y = df["target"].values
    seg = df["segmento"].values

    summary = {"total": {**m06b.credit_metrics(y[valid], probs[valid]),
                         "n": int(valid.sum())}}
    for s in ("esparso", "denso"):
        mask = valid & (seg == s)
        if mask.sum() >= m06b.MIN_SEG_CASOS:
            summary[s] = {**m06b.credit_metrics(y[mask], probs[mask]), "n": int(mask.sum())}
        else:
            summary[s] = {"n": int(mask.sum()), "note": "insuficiente para AUC confiable"}
    return summary


# %% [markdown]
# ## Few-shot end-to-end (nodo Kedro)
#
# `run_fewshot` recibe el dataset, el número de ejemplos y un directorio de salida,
# y retorna las métricas honestas en test por segmento.

# %%
def run_fewshot(dataset_path: str, n_shots: int, output_dir: str,
                scope: str = "all") -> dict:
    """Ejecuta few-shot con n_shots ejemplos del train, sin filtración de datos.

    Args:
        dataset_path: ruta al CSV del dataset.
        n_shots: número de ejemplos balanceados a incluir en el prompt.
        output_dir: directorio para cache y `llm_metrics_fewshot_<n>.json`.
        scope: "all" → puntúa train+val+test y reporta métrica de test **y** general;
               "test" → solo test (comportamiento original de E3).

    Returns:
        dict con métricas en test y —si scope="all"— la métrica general por segmento.
    """
    m06b = _load_06b()
    m06b.check_ollama()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = output_path / f"llm_probs_fewshot_{n_shots}.parquet"

    df = m06b.load_segmented(dataset_path)
    train_df = df[df["set"] == "train"]
    rng = np.random.default_rng(SEED)
    examples = select_few_shot_examples(train_df, n_shots, rng)
    pool = build_replacement_pool(train_df, examples, rng)

    probs = run_fewshot_inference(m06b, df, examples, pool, cache_path, scope=scope)
    summary_test = m06b.evaluate_on_test(df, probs)

    results = {
        "n_shots":    n_shots,
        "model":      m06b.OLLAMA_MODEL,
        "scope":      scope,
        "method":     ("Few-shot TabLLM via Ollama — ejemplos del train, sin leakage "
                       "(cada caso excluido de sus propios ejemplos)"),
        "test":       summary_test,
        "n_features": len(m06b.TOP_FEATURES_LLM),
        "n_test":     int((df["set"] == "test").sum()),
    }
    if scope == "all":
        results["general"] = evaluate_general(m06b, df, probs)
        results["n_general"] = int((~np.isnan(np.asarray(probs, dtype=float))).sum())
    with open(output_path / f"llm_metrics_fewshot_{n_shots}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# %% [markdown]
# ## Figura — AUC vs n_shots

# %%
def plot_auc_vs_shots(results_by_shots: dict, figs_dir: Path) -> None:
    """AUC en test por segmento en función del número de ejemplos."""
    colors = {"esparso": "#E8A838", "denso": "#4878CF", "total": "#888888"}
    fig, ax = plt.subplots(figsize=(9, 5))
    shots = sorted(results_by_shots.keys())
    for seg in ("esparso", "denso", "total"):
        xs, ys = [], []
        for n in shots:
            s = results_by_shots[n]["test"].get(seg, {})
            if "AUC" in s:
                xs.append(n)
                ys.append(s["AUC"])
        if xs:
            ax.plot(xs, ys, marker="o", color=colors[seg], label=seg.capitalize(), lw=2)
    ax.set_xlabel("Número de ejemplos (n-shot)")
    ax.set_ylabel("AUC-ROC (conjunto de test)")
    ax.set_xticks(shots)
    ax.set_xticklabels([f"{n}-shot" for n in shots])
    ax.axhline(0.5, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    ax.set_title("Qwen3-8B Few-Shot: AUC vs. número de ejemplos (test)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "26_fewshot_auc_vs_n.png", bbox_inches="tight")
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

    results_by_shots = {}
    for n_shots in N_SHOTS_LIST:
        results_by_shots[n_shots] = run_fewshot(
            dataset_path=DATASET, n_shots=n_shots, output_dir=str(MODELS))

    combined = {f"{n}_shot": r for n, r in results_by_shots.items()}
    combined.update({"model": "qwen3:8b", "shots_evaluated": N_SHOTS_LIST})
    with open(MODELS / "llm_metrics_fewshot.json", "w") as f:
        json.dump(combined, f, indent=2)

    plot_auc_vs_shots(results_by_shots, FIGS)
