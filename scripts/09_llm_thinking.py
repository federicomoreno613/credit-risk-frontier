# %% [markdown]
# # 09 — Qwen3-8B en MODO DE RAZONAMIENTO (thinking) + selección de ejemplos por similitud
# ## Tesis / E4 — LLMs vs. ML clásico, UBA 2026
#
# Aporte metodológico de este trabajo respecto de los antecedentes: en lugar de pedirle al
# modelo una respuesta directa, se le da espacio para **razonar paso a paso** antes de estimar
# la probabilidad de mora. Esto ataca además el problema del amontonamiento de la probabilidad
# verbalizada (números pegados a 0/50/100): el número final surge de un razonamiento y no de un
# ancla mental.
#
# Cambios respecto de 06b/08:
#   1. Prompt de razonamiento (system + user) que termina forzando `PROBABILIDAD_DE_MORA: <0-100>`.
#   2. Parseo robusto: busca esa línea; como respaldo toma el ÚLTIMO número 0-100 (no el primero,
#      que con razonamiento suele ser un número del texto: dependientes, ingresos, etc.).
#   3. `think=True` (num_predict=512).
#   4. Few-shot con selección de ejemplos por SIMILITUD (kNN) balanceada por clase — ejemplos
#      relevantes al caso, no al azar. Mantiene la garantía anti-filtración (el caso nunca es su
#      propio ejemplo).
#
# Reusa de 06b: serialize_tabllm, TOP_FEATURES_LLM, load_segmented, credit_metrics,
# evaluate_on_test, evaluate_general (de 08), OLLAMA_URL/MODEL, TU_VARS, CORTE_ESPARSO.
#
# Backend: Ollama local (qwen3:8b). Correr en el Mac con `ollama serve` activo.
#
# Uso:
#   python scripts/09_llm_thinking.py --mode zero                 # zero-shot razonado, test
#   python scripts/09_llm_thinking.py --mode few --shots 16       # few-shot 16 razonado, test
#   python scripts/09_llm_thinking.py --mode few --shots 16 --scope all   # + métrica general
#   python scripts/09_llm_thinking.py --mode few --shots 8 32     # varias configs

# %%
import argparse
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SEED = 42


# %% [markdown]
# ## Reusar 06b y 08 como módulos (sin duplicar serialización ni métricas)

# %%
def _load_module(fname: str, alias: str):
    path = Path(__file__).resolve().parent / fname
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m06b = _load_module("06b_llm_prompting.py", "m06b")
m08 = _load_module("08_llm_fewshot.py", "m08")


# %% [markdown]
# ## 1. Prompt de razonamiento

# %%
SYSTEM_THINKING = (
    "Eres un evaluador experto de riesgo crediticio de microcréditos en Colombia. "
    "Para cada solicitud analizas la capacidad de pago, la estabilidad del negocio y el "
    "historial disponible, y estimas la probabilidad de mora. Razonas paso a paso antes de "
    "dar tu estimación final."
)

SYSTEM_THINKING_FEWSHOT = (
    "Eres un evaluador experto de riesgo crediticio de microcréditos en Colombia. "
    "Se te muestran casos resueltos con su desenlace real; aprende de ellos. Para el nuevo "
    "solicitante analizas la capacidad de pago, la estabilidad del negocio y el historial "
    "disponible, y razonas paso a paso antes de dar tu estimación final de probabilidad de mora."
)

_INSTR = (
    "Analiza el caso y estima la probabilidad de que este crédito entre en mora "
    "(atraso mayor a 60 días). Considera la capacidad de pago, la coherencia entre ingresos "
    "y monto solicitado, y el historial de buró. Razona brevemente y termina tu respuesta con "
    "una única línea con este formato EXACTO:\n"
    "PROBABILIDAD_DE_MORA: <entero de 0 a 100>"
)


def build_zero_messages(row: pd.Series) -> list[dict]:
    serialized = m06b.serialize_tabllm(row, m06b.TOP_FEATURES_LLM)
    user = f"DATOS DEL SOLICITANTE: {serialized}\n\n{_INSTR}"
    return [
        {"role": "system", "content": SYSTEM_THINKING},
        {"role": "user", "content": user},
    ]


def build_few_messages(examples: pd.DataFrame, case_row: pd.Series) -> list[dict]:
    feats = m06b.TOP_FEATURES_LLM
    n = len(examples)
    parts = [
        f"A continuación hay {n} casos resueltos de solicitantes con su resultado real "
        "(mora: 100 = entró en mora, 0 = no entró en mora).",
        "",
    ]
    for i, (_, ex) in enumerate(examples.iterrows(), 1):
        parts.append(f"Caso {i}: {m06b.serialize_tabllm(ex, feats)}")
        parts.append(f"Resultado real — probabilidad de mora: {100 if int(ex['target']) == 1 else 0}")
        parts.append("")
    parts.append(f"NUEVO SOLICITANTE: {m06b.serialize_tabllm(case_row, feats)}")
    parts.append("")
    parts.append(_INSTR)
    return [
        {"role": "system", "content": SYSTEM_THINKING_FEWSHOT},
        {"role": "user", "content": "\n".join(parts)},
    ]


# %% [markdown]
# ## 2. Parseo robusto de la probabilidad

# %%
_RE_LABELED = re.compile(r"PROBABILIDAD_DE_MORA\s*:\s*([0-9]{1,3})")

# Flag global: razonamiento nativo de Ollama (think=True). True por defecto (qwen3);
# el CLI lo pone en False para modelos que no lo soportan (p. ej. gemma3).
_THINK_NATIVE = True


def parse_prob(text: str) -> float:
    """Extrae P(mora) en [0,1] SOLO de la línea canónica `PROBABILIDAD_DE_MORA: NN`.

    ESTRICTO a propósito. En modo razonamiento el modelo escribe muchos números en el texto
    (ingresos, edades, números de casos de ejemplo). Si el razonamiento se trunca antes de
    escribir la línea final, NO hay respuesta válida: devolver el "último número que aparezca"
    produce un score espurio (un dato suelto del texto), que fue justo el bug de la corrida con
    num_predict=512. Sin la línea canónica se devuelve NaN → el caso se cuenta como inválido y
    no contamina la métrica. Si aparecen varias líneas etiquetadas se toma la ÚLTIMA (la final).
    """
    matches = _RE_LABELED.findall(text)
    for val_str in reversed(matches):
        val = int(val_str)
        if 0 <= val <= 100:
            return val / 100.0
    return float("nan")


# %% [markdown]
# ## 3. Llamada a Ollama en modo razonamiento

# %%
def call_ollama_think(messages: list[dict], retries: int = 3) -> dict:
    """Llama a Qwen3-8B en modo razonamiento y devuelve TODO el output.

    Retorna un dict con:
      - prob      : score en [0,1] parseado del número final verbalizado (PROBABILIDAD_DE_MORA).
      - content   : texto de la respuesta visible (el razonamiento + el número final).
      - thinking  : cadena de razonamiento interna (campo `thinking` de Ollama), si viene aparte.
      - logprob   : log-probabilidad del/los token(s) del número final, si Ollama la expone
                    (permite un score continuo más fino que el número redondeado).
      - eval_count: nº de tokens generados (control de costo/consistencia).
    Guardar el texto completo hace la evaluación auditable y reproducible: el número final
    (prob) es el score para la ROC/AUC, pero queda registrado el razonamiento que lo produjo.
    """
    # _THINK_NATIVE: True para qwen3 (razonamiento nativo con think=True); False para modelos
    # sin ese mecanismo (p. ej. gemma3), donde el razonamiento lo induce el prompt.
    think_native = globals().get("_THINK_NATIVE", True)
    payload = {
        "model": m06b.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": think_native,
        "logprobs": True,          # pedir log-probabilidades de los tokens generados
        # num_predict alto: en modo razonamiento el modelo necesita espacio para PENSAR y
        # ADEMÁS escribir la línea final PROBABILIDAD_DE_MORA. Con 512 el razonamiento se
        # truncaba antes de dar la respuesta (content vacío) y el parser caía en un número
        # suelto del texto → scores espurios. 3072 da margen para razonar y responder.
        "options": {"temperature": 0.0, "num_predict": 3072},
    }
    for attempt in range(retries):
        try:
            # timeout amplio: en think nativo con few-shot (16 ejemplos ≈ prompt largo) la
            # generación supera los 120s y el timeout corto producía NaN silenciosos.
            resp = requests.post(m06b.OLLAMA_URL, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            content = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""
            text_for_parse = content if content.strip() else thinking
            prob = parse_prob(text_for_parse)
            logprob = _extract_final_logprob(data)
            return {
                "prob": prob,
                "content": content,
                "thinking": thinking,
                "logprob": logprob,
                "eval_count": data.get("eval_count"),
            }
        except (requests.RequestException, KeyError, json.JSONDecodeError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"prob": float("nan"), "content": "", "thinking": "",
                        "logprob": float("nan"), "eval_count": None}


def _extract_final_logprob(data: dict) -> float:
    """Extrae la logprob del último token numérico de la respuesta, si Ollama la expone.

    Ollama devuelve (en versiones que soportan logprobs) una lista bajo `logprobs` con
    entradas {token, logprob}. Se toma la logprob del último token que es un dígito, que
    corresponde al número final PROBABILIDAD_DE_MORA. Si no está disponible, devuelve NaN
    (el pipeline sigue funcionando con el número verbalizado como score).
    """
    lp = data.get("logprobs")
    if not lp:
        # algunas versiones lo anidan en message
        lp = data.get("message", {}).get("logprobs")
    if not lp or not isinstance(lp, list):
        return float("nan")
    last = float("nan")
    for entry in lp:
        tok = str(entry.get("token", ""))
        if any(c.isdigit() for c in tok):
            val = entry.get("logprob")
            if val is not None:
                last = float(val)
    return last


# %% [markdown]
# ## 4. Selección de ejemplos por SIMILITUD (kNN) balanceada por clase
#
# Para cada caso a evaluar se recuperan los `n_shots/2` vecinos más cercanos de cada clase
# dentro del train (distancia euclídea en el espacio de variables numéricas estandarizadas).
# Los ejemplos resultan RELEVANTES al caso, no aleatorios. Se excluye el propio caso
# (garantía anti-filtración) — como el caso puede estar en el train, se filtra por id.

# %%
# NOTA: se excluyen credits_amount_granted y credits_fee_month_amount_granted (monto y cuota
# del crédito otorgado) por LEAKAGE — la selección de vecinos usa solo datos disponibles a
# priori, coherente con la serialización y con los modelos clásicos sin leakage.
_KNN_FEATURES = [f for f in [
    "g051s", "wd81", "wd03", "antiguedad_cliente", "appusers_age",
    "shops_monthly_incomes", "shops_monthly_outcomes", "estimated_income",
    "free_cash_flow", "cost_ingress_ratio", "appusers_score",
]]


def _build_knn_space(train_df: pd.DataFrame):
    """Devuelve (matriz estandarizada del train, medias, desvíos, features usadas)."""
    feats = [f for f in _KNN_FEATURES if f in train_df.columns]
    X = train_df[feats].apply(pd.to_numeric, errors="coerce")
    # los códigos negativos del buró son centinelas → tratarlos como faltante para la distancia
    X = X.mask(X < 0)
    mu = X.mean()
    sd = X.std().replace(0, 1.0)
    Xz = ((X - mu) / sd).fillna(0.0).values
    return Xz, mu, sd, feats


def _standardize_case(row: pd.Series, mu, sd, feats) -> np.ndarray:
    x = pd.to_numeric(pd.Series({f: row.get(f) for f in feats}), errors="coerce")
    x = x.mask(x < 0)
    return ((x - mu) / sd).fillna(0.0).values


def knn_examples_for_case(case_row: pd.Series, case_id: str,
                          train_df: pd.DataFrame, knn_space, n_shots: int) -> pd.DataFrame:
    """`n_shots` ejemplos balanceados: los más similares al caso, mitad por clase, sin el propio caso."""
    Xz, mu, sd, feats = knn_space
    n_per_class = n_shots // 2
    xz = _standardize_case(case_row, mu, sd, feats)
    dist = np.sqrt(((Xz - xz) ** 2).sum(axis=1))
    order = np.argsort(dist)

    picked, seen_ids = [], set()
    need = {0: n_per_class, 1: n_per_class}
    tgt = train_df["target"].values
    ids = train_df["credito_id_anon"].values
    idx = train_df.index.values
    for pos in order:
        cls = int(tgt[pos])
        cid = ids[pos]
        if cid == case_id or cid in seen_ids or need[cls] == 0:
            continue
        picked.append(idx[pos])
        seen_ids.add(cid)
        need[cls] -= 1
        if need[0] == 0 and need[1] == 0:
            break
    ex = train_df.loc[picked]
    # barajar el orden de presentación (que no queden todas las de una clase juntas)
    return ex.sample(frac=1, random_state=SEED).reset_index(drop=True)


# %% [markdown]
# ## 5. Inferencia con cache incremental

# %%
def run_inference(df: pd.DataFrame, target_index, cache_path: Path,
                  build_messages_fn) -> np.ndarray:
    """Puntúa las filas de `target_index` con cache por credito_id_anon."""
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
    else:
        cache = pd.DataFrame(columns=["credito_id_anon", "prob_llm",
                                      "respuesta_texto", "razonamiento", "logprob_final", "n_tokens"])
    cached = set(cache["credito_id_anon"])

    pending = [i for i in target_index if df.loc[i, "credito_id_anon"] not in cached]
    print(f"  {len(pending)} casos por inferir ({len(cached)} en cache)")
    new_rows = []
    for k, i in enumerate(pending, 1):
        row = df.loc[i]
        out = call_ollama_think(build_messages_fn(row))
        new_rows.append({
            "credito_id_anon": row["credito_id_anon"],
            "prob_llm": out["prob"],
            "respuesta_texto": out["content"],   # texto completo de la respuesta (razonamiento + número)
            "razonamiento": out["thinking"],     # cadena de razonamiento interna, si viene aparte
            "logprob_final": out["logprob"],     # logprob del número final (score continuo más fino)
            "n_tokens": out["eval_count"],
        })
        if k % 25 == 0 or k == len(pending):
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            cache.to_parquet(cache_path, index=False)
            new_rows = []
            print(f"    {k}/{len(pending)}")
    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_parquet(cache_path, index=False)
    cache = cache.drop_duplicates("credito_id_anon", keep="last")
    merged = df[["credito_id_anon"]].merge(cache, on="credito_id_anon", how="left")
    return merged["prob_llm"].values


# %% [markdown]
# ## 6. Orquestación zero-shot / few-shot

# %%
def _subset_index(df: pd.DataFrame, target_index, limit: int, limit_per_class: int):
    """Recorta target_index para pruebas rápidas.

    OJO: el test está ORDENADO POR TARGET (primero los 252 de clase 0, después los 243
    de clase 1), así que `limit` a secas toma solo clase 0 y el AUC es incomputable.
    Para subconjuntos con métrica usar `limit_per_class` (primeros N de CADA clase).
    """
    if limit_per_class and limit_per_class > 0:
        tgt = df.loc[target_index, "target"]
        idx0 = target_index[(tgt == 0).values][:limit_per_class]
        idx1 = target_index[(tgt == 1).values][:limit_per_class]
        return idx0.append(idx1)
    if limit and limit > 0:
        return target_index[:limit]
    return target_index


def run_zero(dataset_path: str, output_dir: str, scope: str = "test", limit: int = 0,
             limit_per_class: int = 0) -> dict:
    m06b.check_ollama()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = m06b.load_segmented(dataset_path)
    target_index = df.index if scope == "all" else df.index[df["set"] == "test"]
    target_index = _subset_index(df, target_index, limit, limit_per_class)
    cache = out / f"llm_probs_thinking_zero_{scope}.parquet"
    probs = run_inference(df, target_index, cache, build_zero_messages)
    res = {"mode": "thinking-zero", "model": m06b.OLLAMA_MODEL, "scope": scope,
           "test": m06b.evaluate_on_test(df, probs)}
    if scope == "all":
        res["general"] = m08.evaluate_general(m06b, df, probs)
    (out / f"llm_metrics_thinking_zero_{scope}.json").write_text(json.dumps(res, indent=2))
    return res


def run_few(dataset_path: str, n_shots: int, output_dir: str, scope: str = "test", limit: int = 0,
            limit_per_class: int = 0) -> dict:
    m06b.check_ollama()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = m06b.load_segmented(dataset_path)
    train_df = df[df["set"] == "train"].copy()
    knn_space = _build_knn_space(train_df)

    target_index = df.index if scope == "all" else df.index[df["set"] == "test"]
    target_index = _subset_index(df, target_index, limit, limit_per_class)
    cache = out / f"llm_probs_thinking_few{n_shots}_{scope}.parquet"

    def build(row):
        ex = knn_examples_for_case(row, row["credito_id_anon"], train_df, knn_space, n_shots)
        return build_few_messages(ex, row)

    probs = run_inference(df, target_index, cache, build)
    res = {"mode": "thinking-few", "n_shots": n_shots, "model": m06b.OLLAMA_MODEL,
           "scope": scope, "selection": "knn-similar-balanced",
           "test": m06b.evaluate_on_test(df, probs)}
    if scope == "all":
        res["general"] = m08.evaluate_general(m06b, df, probs)
    (out / f"llm_metrics_thinking_few{n_shots}_{scope}.json").write_text(json.dumps(res, indent=2))
    return res


# %% [markdown]
# ## 7. CLI

# %%
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Qwen3-8B en modo razonamiento (E4)")
    ap.add_argument("--mode", choices=["zero", "few"], required=True)
    ap.add_argument("--shots", type=int, nargs="+", default=[16],
                    help="nº de ejemplos (uno o varios), solo para --mode few")
    ap.add_argument("--scope", choices=["test", "all"], default="test",
                    help="test = 495 casos (comparación honesta); all = 4.897 (métrica general)")
    ap.add_argument("--dataset", default=str(BASE / "data" / "dataset_tesis.csv"))
    ap.add_argument("--out", default=str(BASE / "models"))
    ap.add_argument("--limit", type=int, default=0,
                    help="si >0, evalúa solo los primeros N casos (OJO: el test está ordenado "
                         "por target → los primeros 252 son clase 0; sin métrica posible)")
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="si >0, primeros N casos de CADA clase (subconjunto balanceado con AUC)")
    ap.add_argument("--ollama-model", default=None,
                    help="sobrescribe el modelo de Ollama (p. ej. gemma4:12b). Default: qwen3:8b")
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto",
                    help="razonamiento nativo de Ollama: auto = solo qwen3 (comportamiento "
                         "histórico); on/off lo fuerza (gemma4 soporta thinking nativo)")
    ap.add_argument("--features", choices=["full", "tu", "tu-sem", "sin-tu"], default="full",
                    help="ablación de fuentes: tu = solo buró TU; tu-sem = buró + semántica "
                         "(texto/categóricas); sin-tu = negocio completo sin buró; full = todo")
    args = ap.parse_args()

    # ablación de fuentes de información (misma instrucción, cambia qué datos se serializan)
    if args.features != "full":
        _tu_feats = [f for f in m06b.TOP_FEATURES_LLM if f in m06b.TU_DESC]
        _neg_feats = [f for f in m06b.TOP_FEATURES_LLM if f not in m06b.TU_DESC]
        if args.features == "tu":
            m06b.TOP_FEATURES_LLM = _tu_feats
            m06b.INCLUDE_CATEGORICALS = False
            m06b.INCLUDE_TEXT = False
        elif args.features == "tu-sem":
            m06b.TOP_FEATURES_LLM = _tu_feats
        elif args.features == "sin-tu":
            m06b.TOP_FEATURES_LLM = _neg_feats
        print(f"[features] perfil={args.features} → {len(m06b.TOP_FEATURES_LLM)} vars numéricas, "
              f"categóricas={getattr(m06b, 'INCLUDE_CATEGORICALS', True)}, "
              f"texto={getattr(m06b, 'INCLUDE_TEXT', True)}")

    # permitir cambiar de modelo sin tocar el código (Qwen / Gemma / otros en Ollama)
    if args.ollama_model:
        m06b.OLLAMA_MODEL = args.ollama_model
        # en auto, el razonamiento nativo (think=True) se activa solo para qwen3; para otros
        # modelos se desactiva y el razonamiento lo induce el prompt.
        if args.think == "auto" and not args.ollama_model.lower().startswith("qwen3"):
            globals()["_THINK_NATIVE"] = False
    if args.think != "auto":
        globals()["_THINK_NATIVE"] = args.think == "on"
    print(f"[modelo] usando OLLAMA_MODEL = {m06b.OLLAMA_MODEL}"
          f"  (think nativo = {globals().get('_THINK_NATIVE', True)})")

    if args.mode == "zero":
        r = run_zero(args.dataset, args.out, scope=args.scope, limit=args.limit,
                     limit_per_class=args.limit_per_class)
        print("\n=== zero-shot razonado ===")
        print(json.dumps(r["test"]["total"], indent=2, ensure_ascii=False))
    else:
        for n in args.shots:
            print(f"\n===== few-shot razonado, {n} ejemplos (kNN), scope={args.scope} =====")
            r = run_few(args.dataset, n, args.out, scope=args.scope, limit=args.limit,
                        limit_per_class=args.limit_per_class)
            print(json.dumps(r["test"]["total"], indent=2, ensure_ascii=False))
