#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPT (OpenAI) en modo razonamiento — MISMA serialización y prompt que Qwen/Gemma (09).

Reutiliza build_zero_messages / build_few_messages / parse_prob / run_inference /
_subset_index de `09_llm_thinking.py`; lo ÚNICO que cambia es el transporte: en vez de
Ollama, el SDK de OpenAI. Así la comparación es justa (mismo input, distinto modelo).

Particularidades de los GPT-5.x (razonamiento) vs. el call anterior:
  - NO aceptan `temperature` distinta de 1 ni `max_tokens` → se usa `max_completion_tokens`
    (incluye los tokens de razonamiento, que se facturan como output).
  - `reasoning_effort` controla cuánto razona (default de la API: none = no razona).
  - `logprobs` puede no estar soportado con razonamiento → se degrada limpio (columna NaN).
  - Un error de CONFIGURACIÓN (param inválido, auth, modelo inexistente) ABORTA la corrida
    con mensaje, en vez de degradar a NaN silencioso: el bug clásico era quemar 495 llamadas
    devolviendo todo NaN. Solo los errores TRANSITORIOS (rate limit, red, 5xx) reintentan.

La API key se lee de OPENAI_API_KEY del entorno o, si falta, de la línea OPENAI_API_KEY= de
`/Users/federicomoreno/Documents/TESIS/.env` (parseo de esa línea sola: el archivo tiene otras
líneas sin formato KEY=VALUE, así que `source` no sirve). NUNCA se imprime ni se loguea.

Uso (Mac de Federico):
    python3 scripts/09c_llm_gpt.py --mode zero --smoke --openai-model MODELO --out models/gpt
    python3 scripts/09c_llm_gpt.py --mode few --shots 16 --limit-per-class 100 \
        --openai-model MODELO --reasoning-effort medium --out models/gpt

Salida: parquet con credito_id_anon, prob_llm, respuesta_texto, razonamiento(=""),
logprob_final, n_tokens — idéntico esquema que Qwen/Gemma, para reusar métricas y figuras.
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, BASE / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m09 = _load("09_llm_thinking.py", "m09")
m06b = m09.m06b

# --- configuración global (se fija en main() desde el CLI) ---
_CLIENT = None
_OPENAI_MODEL = None
_REASONING_EFFORT = "medium"
_MAX_COMPLETION = 8192   # razonar consume de este presupuesto; si finish_reason=length, subirlo
_LOGPROBS_OK = True      # se apaga solo si el modelo rechaza logprobs (no es fatal)
_CONSEC_TRANSIENT = 0    # circuit breaker: N casos seguidos con transitorios agotados → abortar
_MAX_CONSEC_TRANSIENT = 5


def _load_env_key():
    """Carga OPENAI_API_KEY desde TESIS/.env si no está ya en el entorno."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_path = BASE.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            return


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("Falta el paquete openai. Instalar: pip install openai") from e
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("No hay OPENAI_API_KEY en el entorno ni en TESIS/.env.")
        _CLIENT = OpenAI(api_key=key)
    return _CLIENT


def _preflight(model_id: str) -> None:
    """Verifica que el modelo exista para esta key ANTES de cargar el dataset."""
    import openai
    client = _get_client()
    try:
        client.models.retrieve(model_id)
        print(f"[preflight] modelo OK: {model_id}")
    except openai.NotFoundError:
        disponibles = sorted(m.id for m in client.models.list() if "gpt-5" in m.id)
        raise SystemExit(f"[FATAL] modelo '{model_id}' no existe para esta key. "
                         f"gpt-5* disponibles: {disponibles}")


def call_openai(messages: list[dict], retries: int = 4) -> dict:
    """Devuelve el MISMO dict que call_ollama_think: {prob, content, thinking, logprob, eval_count}.

    El razonamiento de GPT es interno (tokens invisibles); la respuesta visible va en
    `content` y el prompt ya pide terminar con la línea PROBABILIDAD_DE_MORA: NN, que
    parse_prob (estricto) toma como score. Sin esa línea, prob = NaN (caso inválido).
    """
    global _LOGPROBS_OK, _CONSEC_TRANSIENT
    import openai
    client = _get_client()
    attempt = 0
    while True:
        kwargs = dict(
            model=_OPENAI_MODEL,
            messages=messages,
            max_completion_tokens=_MAX_COMPLETION,
            reasoning_effort=_REASONING_EFFORT,
        )
        if _LOGPROBS_OK:
            kwargs["logprobs"] = True
        try:
            resp = client.chat.completions.create(**kwargs)
        except (openai.BadRequestError, openai.PermissionDeniedError) as e:
            # el rechazo de logprobs puede venir como 400 o como 403 según el modelo
            if _LOGPROBS_OK and "logprob" in str(e).lower():
                _LOGPROBS_OK = False
                print("    [WARN] el modelo rechaza logprobs; sigo sin esa columna")
                continue
            raise SystemExit(f"[FATAL] parámetro inválido/permiso para {_OPENAI_MODEL}: {e}") from e
        except (openai.AuthenticationError, openai.NotFoundError) as e:
            raise SystemExit(f"[FATAL] auth/modelo: {e}") from e
        except (openai.RateLimitError, openai.InternalServerError,
                openai.APIConnectionError) as e:
            attempt += 1
            if attempt >= retries:
                # OJO: insufficient_quota también llega como 429; sin este freno, una cuota
                # agotada a mitad de corrida cachearía NaN para todos los casos restantes.
                _CONSEC_TRANSIENT += 1
                if _CONSEC_TRANSIENT >= _MAX_CONSEC_TRANSIENT:
                    raise SystemExit(
                        f"[FATAL] {_CONSEC_TRANSIENT} casos consecutivos con errores "
                        f"transitorios agotados ({type(e).__name__}) — ¿cuota agotada o red "
                        "caída? Abortando para no envenenar el cache.") from e
                print(f"    [WARN] transitorio agotado tras {retries} intentos: "
                      f"{type(e).__name__}")
                return {"prob": float("nan"), "content": "", "thinking": "",
                        "logprob": float("nan"), "eval_count": None}
            time.sleep(min(2 ** attempt, 30))
            continue
        except openai.APIStatusError as e:
            raise SystemExit(f"[FATAL] error de API no transitorio ({e.status_code}): {e}") from e

        _CONSEC_TRANSIENT = 0
        choice = resp.choices[0]
        content = choice.message.content or ""
        if choice.finish_reason == "length" and not content.strip():
            print("    [WARN] finish_reason=length: el razonamiento agotó "
                  "max_completion_tokens; subir --max-completion-tokens")
        prob = m09.parse_prob(content)
        logprob = float("nan")
        if _LOGPROBS_OK:
            try:
                toks = choice.logprobs.content or []
                for t in toks:
                    if any(c.isdigit() for c in str(t.token)):
                        logprob = float(t.logprob)
            except (AttributeError, TypeError):
                pass
        return {"prob": prob, "content": content, "thinking": "",
                "logprob": logprob, "eval_count": getattr(resp.usage, "completion_tokens", None)}


def _probs_on_target(df, target_index, probs: np.ndarray) -> np.ndarray:
    """probs viene alineado POSICIONALMENTE con df (run_inference); recorta al target."""
    pos = df.index.get_indexer(target_index)
    return probs[pos].astype(float)


def _prepare_cache(cache: Path, smoke: bool) -> None:
    """Deja el cache en estado sano ANTES de run_inference.

    - smoke: se borra siempre → el smoke paga sus 6 llamadas y valida la config ACTUAL
      (un smoke cacheado imprime OK sin probar nada).
    - real: (a) purga las filas con prob_llm NaN para que el relanzamiento las reintente
      (run_inference saltea todo id presente en el cache, NaN incluido); (b) valida vía
      sidecar que el cache venga del MISMO modelo+effort — sin esto, relanzar con otra
      config mezcla probs de dos modelos en un parquet que se rotula como uno solo.
    """
    if smoke:
        if cache.exists():
            cache.unlink()
        return
    side = cache.with_suffix(".config.json")
    cfg = {"model": _OPENAI_MODEL, "reasoning_effort": _REASONING_EFFORT,
           "max_completion_tokens": _MAX_COMPLETION}
    if cache.exists():
        if side.exists():
            prev = json.loads(side.read_text())
            if (prev.get("model"), prev.get("reasoning_effort")) != \
               (cfg["model"], cfg["reasoning_effort"]):
                raise SystemExit(
                    f"[FATAL] el cache {cache.name} fue generado con "
                    f"{prev.get('model')}/effort={prev.get('reasoning_effort')} y ahora se "
                    f"pide {cfg['model']}/effort={cfg['reasoning_effort']}. Usar otro --out "
                    "o borrar ese parquet (y su .config.json) a propósito.")
        else:
            print(f"  [cache] {cache.name} existe sin sidecar de config; se asume la "
                  "config actual y se registra")
        d = pd.read_parquet(cache)
        nan_mask = pd.to_numeric(d["prob_llm"], errors="coerce").isna()
        if nan_mask.any():
            print(f"  [cache] purgando {int(nan_mask.sum())} filas NaN de {cache.name} "
                  "(se reintentarán)")
            d[~nan_mask].to_parquet(cache, index=False)
    side.write_text(json.dumps(cfg, indent=2))


def _smoke_report(df, target_index, probs: np.ndarray, label: str) -> None:
    vals = _probs_on_target(df, target_index, probs)
    sub = df.loc[target_index]
    print(f"\n=== SMOKE {label} ({len(sub)} casos) ===")
    for cid, tgt, p in zip(sub["credito_id_anon"], sub["target"], vals):
        print(f"  {cid} (target={int(tgt)}) -> prob={p}")
    nan_rate = float(np.isnan(vals).mean())
    print(f"  nan_rate={nan_rate:.2f}")
    if nan_rate > 0.4:
        raise SystemExit(f"[SMOKE FAIL] {label}: nan_rate {nan_rate:.0%} > 40% — revisar "
                         "params/parser antes de la corrida real.")
    print("  SMOKE OK")


def main():
    global _OPENAI_MODEL, _REASONING_EFFORT, _MAX_COMPLETION
    ap = argparse.ArgumentParser(description="GPT (OpenAI) en modo razonamiento (E4)")
    ap.add_argument("--mode", choices=["zero", "few"], required=True)
    ap.add_argument("--shots", type=int, nargs="+", default=[16])
    ap.add_argument("--scope", choices=["test", "all"], default="test")
    ap.add_argument("--limit", type=int, default=0,
                    help="si >0, primeros N casos (OJO: el test está ordenado por target; "
                         "para subconjuntos con métrica usar --limit-per-class)")
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="si >0, primeros N casos de CADA clase (subconjunto balanceado)")
    ap.add_argument("--smoke", action="store_true",
                    help="prueba de humo: 3 casos por clase, cache aparte, sin métricas; "
                         "aborta si >40%% NaN")
    ap.add_argument("--openai-model", required=True, help="id del modelo OpenAI a usar")
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["none", "low", "medium", "high", "xhigh"])
    ap.add_argument("--max-completion-tokens", type=int, default=8192)
    ap.add_argument("--dataset", default=str(BASE / "data" / "dataset_tesis.csv"))
    ap.add_argument("--out", default=str(BASE / "models" / "gpt"))
    args = ap.parse_args()
    _OPENAI_MODEL = args.openai_model
    _REASONING_EFFORT = args.reasoning_effort
    _MAX_COMPLETION = args.max_completion_tokens

    _load_env_key()
    _preflight(_OPENAI_MODEL)
    print(f"[config] effort={_REASONING_EFFORT} max_completion_tokens={_MAX_COMPLETION}")

    # monkeypatch: run_inference de m09 llama a call_ollama_think; lo apuntamos a call_openai
    m09.call_ollama_think = call_openai

    df = m06b.load_segmented(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target_index = df.index if args.scope == "all" else df.index[df["set"] == "test"]
    if args.smoke:
        target_index = m09._subset_index(df, target_index, 0, 3)
    else:
        target_index = m09._subset_index(df, target_index, args.limit, args.limit_per_class)
    suffix = "_smoke" if args.smoke else ""
    # los JSON de subconjuntos llevan _nNNN para no pisar la métrica del test completo
    partial = bool(args.limit or args.limit_per_class)
    jsuffix = f"_n{len(target_index)}" if partial else ""

    def _finish(probs, res_base: dict, json_path: Path, label: str):
        if args.smoke:
            _smoke_report(df, target_index, probs, label)
            return
        vals = _probs_on_target(df, target_index, probs)
        # evaluar SOLO el target: el parquet puede traer más casos cacheados (p. ej. una
        # corrida 495 previa al regenerar el JSON _n200) y evaluate_on_test usa todo lo
        # no-NaN del test — sin esta máscara el JSON quedaría mal etiquetado.
        masked = np.full(len(probs), np.nan)
        masked[df.index.get_indexer(target_index)] = vals
        res = {**res_base,
               "reasoning_effort": _REASONING_EFFORT,
               "max_completion_tokens": _MAX_COMPLETION,
               "logprobs_disponibles": _LOGPROBS_OK,
               "n_target": int(len(target_index)),
               "nan_rate": float(np.isnan(vals).mean()),
               "test": m06b.evaluate_on_test(df, masked)}
        json_path.write_text(json.dumps(res, indent=2))
        print(f"\n=== {label} (n_target={len(target_index)}, "
              f"nan_rate={res['nan_rate']:.3f}) ===")
        print(json.dumps(res["test"].get("total", res["test"]), indent=2, ensure_ascii=False))

    if args.mode == "zero":
        cache = out / f"gpt_probs_zero_{args.scope}{suffix}.parquet"
        _prepare_cache(cache, args.smoke)
        probs = m09.run_inference(df, target_index, cache, m09.build_zero_messages)
        _finish(probs,
                {"model": _OPENAI_MODEL, "mode": "zero", "scope": args.scope},
                out / f"gpt_metrics_zero_{args.scope}{jsuffix}.json",
                "GPT zero-shot")
    else:
        train_df = df[df["set"] == "train"].copy()
        knn_space = m09._build_knn_space(train_df)
        for n in args.shots:
            cache = out / f"gpt_probs_few{n}_{args.scope}{suffix}.parquet"
            _prepare_cache(cache, args.smoke)

            def build(row, _n=n):
                ex = m09.knn_examples_for_case(row, row["credito_id_anon"], train_df,
                                               knn_space, _n)
                return m09.build_few_messages(ex, row)

            probs = m09.run_inference(df, target_index, cache, build)
            _finish(probs,
                    {"model": _OPENAI_MODEL, "mode": "few", "n_shots": n,
                     "scope": args.scope, "selection": "knn-similar-balanced"},
                    out / f"gpt_metrics_few{n}_{args.scope}{jsuffix}.json",
                    f"GPT few-shot {n}")


if __name__ == "__main__":
    main()
