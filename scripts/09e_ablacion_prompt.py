"""09e_ablacion_prompt.py — Ablación del PROMPT en zero-shot (E4).

Pregunta: ¿el resultado del LLM depende de cómo está escrito el prompt?
¿Aporta poner un rol experto y explicar qué es cada variable?

Diseño: se mantiene TODO constante (mismo modelo, misma serialización TabLLM,
mismo modo de razonamiento nativo, mismo parseo, mismo conjunto de prueba, temp=0)
y SOLO se varía el texto del prompt. Se corren varias variantes zero-shot y se
compara el AUC. Si el AUC no se mueve, el prompt no es el cuello de botella —
lo que gobierna el resultado es la señal de los datos (sesgo de selección).

Reutiliza la maquinaria real del pipeline (09_llm_thinking.py → 06b/08):
serialize_tabllm, TOP_FEATURES_LLM, call_ollama_think, parse_prob, evaluate_on_test.

Uso (correr en tu Mac con Ollama levantado):
    # Prueba rápida (30 por clase = 60 casos) para ver si algo se mueve:
    python scripts/09e_ablacion_prompt.py --limit-per-class 30

    # Corrida completa sobre el test (495 casos) para las 4 variantes:
    python scripts/09e_ablacion_prompt.py

    # Elegir modelo / subconjunto de variantes:
    python scripts/09e_ablacion_prompt.py --ollama-model qwen3:8b --variantes v1_actual v3_explicativo

Salida:
    models/ablacion_prompt/probs_<variante>_<scope>.parquet   (una por variante, con cache)
    models/ablacion_prompt/ablacion_prompt_<scope>.csv        (tabla comparativa de AUC)
    models/ablacion_prompt/ablacion_prompt_<scope>.json       (métricas completas)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def _load_module(fname: str, alias: str):
    path = Path(__file__).resolve().parent / fname
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reutiliza el pipeline real: 09 trae 06b (serialización, features, parseo, eval) y
# la llamada a Ollama en modo razonamiento nativo.
m09 = _load_module("09_llm_thinking.py", "m09")
m06b = m09.m06b


# ---------------------------------------------------------------------------
# VARIANTES DE PROMPT
# Cada variante es (system, user_builder). El user_builder recibe la serialización
# ya construida (idéntica en todas) y devuelve el texto del mensaje de usuario.
# La serialización NO cambia entre variantes: siempre `descripción: valor`.
# ---------------------------------------------------------------------------

# --- v0: mínimo, SIN rol (estilo prompting directo original 06b) ---
V0_SYSTEM = (
    "Eres evaluador de riesgo crediticio de microcréditos en Colombia. "
    "Responde SOLO con un número entero."
)
def v0_user(serialized: str) -> str:
    return f"DATOS: {serialized}\n\nProbabilidad de mora (0-100, entero):"


# --- v1: ACTUAL (el de las cifras definitivas): rol experto + razonamiento + def. de mora ---
V1_SYSTEM = m09.SYSTEM_THINKING
def v1_user(serialized: str) -> str:
    return f"DATOS DEL SOLICITANTE: {serialized}\n\n{m09._INSTR}"


# --- v2: ROL FUERTE (persona senior detallada), mismo cuerpo que v1 ---
V2_SYSTEM = (
    "Eres un analista senior de riesgo crediticio con más de 15 años de experiencia "
    "evaluando microcréditos para microempresarios en Colombia. Has revisado decenas de "
    "miles de solicitudes y tenés un criterio afinado para detectar señales de riesgo en la "
    "capacidad de pago, la estabilidad del negocio y el comportamiento en el buró. "
    "Sos riguroso, prudente y calibrás bien tus probabilidades. Razonás paso a paso antes "
    "de dar tu estimación final."
)
def v2_user(serialized: str) -> str:
    return f"DATOS DEL SOLICITANTE: {serialized}\n\n{m09._INSTR}"


# --- v3: EXPLICATIVO: v1 + leyenda que explica los grupos de variables y ancla la escala ---
V3_SYSTEM = m09.SYSTEM_THINKING
_LEYENDA = (
    "Guía para interpretar los datos:\n"
    "- Variables del buró de crédito (TransUnion): describen el historial de mora del "
    "solicitante en el sistema financiero. Un valor 0 o 'sin historial' significa que NO "
    "registra mora previa; valores más altos indican más mora histórica registrada.\n"
    "- Variables del negocio (ingresos, egresos, flujo de caja libre, relación costo/ingreso): "
    "describen la capacidad de pago. Mayor flujo de caja y menor relación costo/ingreso "
    "indican mayor holgura financiera.\n"
    "- Antigüedad como cliente y 'Primer Crédito': indican cuánta relación previa hay con la "
    "entidad.\n"
    "- Texto del negocio: descripción libre de la actividad.\n\n"
    "Escala de salida: 0 = certeza de que PAGA (no entra en mora); 100 = certeza de que "
    "ENTRA EN MORA. Usá toda la escala; evitá redondear a 0, 50 o 100 salvo que corresponda."
)
def v3_user(serialized: str) -> str:
    return f"{_LEYENDA}\n\nDATOS DEL SOLICITANTE: {serialized}\n\n{m09._INSTR}"


VARIANTES = {
    "v0_minimo":      (V0_SYSTEM, v0_user, "Sin rol, prompt mínimo (estilo directo)"),
    "v1_actual":      (V1_SYSTEM, v1_user, "ACTUAL: rol experto + razonamiento + def. mora"),
    "v2_rol_fuerte":  (V2_SYSTEM, v2_user, "Rol experto reforzado (persona senior detallada)"),
    "v3_explicativo": (V3_SYSTEM, v3_user, "v1 + leyenda de variables + anclaje de escala"),
}


def build_messages(variant: str, row: pd.Series) -> list[dict]:
    system, user_builder, _ = VARIANTES[variant]
    serialized = m06b.serialize_tabllm(row, m06b.TOP_FEATURES_LLM)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_builder(serialized)},
    ]


# Parseo robusto: la variante v0 (prompt mínimo) devuelve un número pelado, sin la
# línea PROBABILIDAD_DE_MORA que espera m09.parse_prob → caería en NaN. Fallback:
# tomar el último entero 0-100 del texto de respuesta (no del razonamiento interno).
_RE_LABELED = __import__("re").compile(r"PROBABILIDAD_DE_MORA\s*:\s*([0-9]{1,3})")
_RE_ANYNUM = __import__("re").compile(r"\b([0-9]{1,3})\b")


def robust_prob(prob_pipeline: float, content: str) -> float:
    """Usa el prob del pipeline si es válido; si es NaN, re-parsea el texto crudo."""
    if prob_pipeline is not None and not (isinstance(prob_pipeline, float) and np.isnan(prob_pipeline)):
        return prob_pipeline
    if not isinstance(content, str) or not content.strip():
        return float("nan")
    m = _RE_LABELED.search(content)
    if m:
        v = int(m.group(1))
        return float(v) if 0 <= v <= 100 else float("nan")
    nums = [int(x) for x in _RE_ANYNUM.findall(content) if 0 <= int(x) <= 100]
    return float(nums[-1]) if nums else float("nan")


def run_variant(variant: str, df: pd.DataFrame, target_index, out_dir: Path,
                scope: str) -> np.ndarray:
    """Puntúa el subconjunto con una variante de prompt, con cache por caso."""
    cache_path = out_dir / f"probs_{variant}_{scope}.parquet"
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
    else:
        cache = pd.DataFrame(columns=["credito_id_anon", "prob_llm", "respuesta_texto",
                                      "razonamiento", "logprob_final", "n_tokens"])
    cached = set(cache["credito_id_anon"])
    pending = [i for i in target_index if df.loc[i, "credito_id_anon"] not in cached]
    print(f"  [{variant}] {len(pending)} por inferir ({len(cached)} en cache)")

    new_rows = []
    for k, i in enumerate(pending, 1):
        row = df.loc[i]
        out = m09.call_ollama_think(build_messages(variant, row))
        new_rows.append({
            "credito_id_anon": row["credito_id_anon"],
            "prob_llm": out["prob"],
            "respuesta_texto": out["content"],
            "razonamiento": out["thinking"],
            "logprob_final": out["logprob"],
            "n_tokens": out["eval_count"],
        })
        if k % 25 == 0 or k == len(pending):
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            cache.to_parquet(cache_path, index=False)
            new_rows = []
            print(f"    {variant}: {k}/{len(pending)}")
    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_parquet(cache_path, index=False)

    cache = cache.drop_duplicates("credito_id_anon", keep="last")
    # Re-parseo robusto sobre el texto crudo (recupera v0 y cualquier NaN por formato).
    cache["prob_robusta"] = [
        robust_prob(pp, ct) for pp, ct in zip(cache["prob_llm"], cache["respuesta_texto"])
    ]
    n_rescatados = int((cache["prob_llm"].isna() & cache["prob_robusta"].notna()).sum())
    if n_rescatados:
        print(f"    [{variant}] {n_rescatados} casos rescatados por re-parseo del texto crudo")
    merged = df[["credito_id_anon"]].merge(cache, on="credito_id_anon", how="left")
    return merged["prob_robusta"].values


def main():
    ap = argparse.ArgumentParser(description="Ablación de prompt zero-shot (E4)")
    ap.add_argument("--dataset", default=str(BASE / "data" / "dataset_tesis.csv"))
    ap.add_argument("--out", default=str(BASE / "models" / "ablacion_prompt"))
    ap.add_argument("--scope", choices=["test", "all"], default="test")
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="Primeros N de CADA clase (prueba rápida; el test está ordenado por target).")
    ap.add_argument("--ollama-model", default=None, help="Ej: qwen3:8b (default: el de 06b).")
    ap.add_argument("--variantes", nargs="+", default=list(VARIANTES.keys()),
                    choices=list(VARIANTES.keys()),
                    help="Subconjunto de variantes a correr.")
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto",
                    help="Razonamiento nativo (auto=on para qwen3).")
    args = ap.parse_args()

    if args.ollama_model:
        m06b.OLLAMA_MODEL = args.ollama_model
    if args.think == "auto":
        # off si el modelo no es qwen3 (mismo criterio que 09)
        if "qwen3" not in m06b.OLLAMA_MODEL:
            m09._THINK_NATIVE = False
    else:
        m09._THINK_NATIVE = args.think == "on"

    print(f"[modelo] {m06b.OLLAMA_MODEL}  (think nativo = {m09._THINK_NATIVE})")
    print(f"[variantes] {', '.join(args.variantes)}")

    m06b.check_ollama()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = m06b.load_segmented(args.dataset)
    target_index = df.index if args.scope == "all" else df.index[df["set"] == "test"]
    target_index = m09._subset_index(df, target_index, 0, args.limit_per_class)
    print(f"[datos] {len(target_index)} casos (scope={args.scope}, "
          f"limit_per_class={args.limit_per_class})")

    resultados = {}
    for v in args.variantes:
        print(f"\n=== Variante {v}: {VARIANTES[v][2]} ===")
        probs = run_variant(v, df, target_index, out_dir, args.scope)
        metrics = m06b.evaluate_on_test(df, probs)  # {total:{AUC,Gini,KS,...,n}, esparso, denso}
        resultados[v] = {"descripcion": VARIANTES[v][2], "metrics": metrics}
        print(f"    → AUC total = {metrics['total'].get('AUC')}")

    # Tabla comparativa (usa el segmento 'total')
    rows = []
    for v, r in resultados.items():
        tot = r["metrics"]["total"]
        rows.append({
            "variante": v,
            "descripcion": r["descripcion"],
            "auc": tot.get("AUC"),
            "gini": tot.get("Gini"),
            "ks": tot.get("KS"),
            "n": tot.get("n"),
        })
    tabla = pd.DataFrame(rows)
    csv_path = out_dir / f"ablacion_prompt_{args.scope}.csv"
    json_path = out_dir / f"ablacion_prompt_{args.scope}.json"
    tabla.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(resultados, indent=2, default=float))

    print("\n" + "=" * 60)
    print("RESULTADO DE LA ABLACIÓN DE PROMPT (zero-shot)")
    print("=" * 60)
    print(tabla.to_string(index=False))
    print(f"\nGuardado: {csv_path}")
    print(f"          {json_path}")
    auc_vals = [r["auc"] for r in rows if r["auc"] is not None]
    if auc_vals:
        print(f"\nRango de AUC entre variantes: {min(auc_vals):.4f} – {max(auc_vals):.4f} "
              f"(spread {max(auc_vals) - min(auc_vals):.4f})")
        print("Si el spread es chico (< ~0,03), el prompt NO es el cuello de botella: "
              "el resultado lo gobierna la señal de los datos, no la redacción.")


if __name__ == "__main__":
    main()
