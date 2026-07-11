#!/usr/bin/env bash
# ============================================================================
# Lanza las corridas de GPT (OpenAI API) del protocolo E4, en segundo plano.
# Correr en el Mac. Deja logs en models/gpt/logs/ y sobrevive al cierre de la terminal.
#
#   bash run_gpt54.sh                     # smoke + etapa 200 balanceados (zero + few16)
#   RUN_FULL_495=1 bash run_gpt54.sh      # además completa el test entero (495)
#   OPENAI_MODEL_ID=gpt-5.4 bash run_gpt54.sh   # sobrescribe el id del modelo
#
# Seguir el avance:   tail -f models/gpt/logs/gpt_*.log
# Ver si sigue vivo:  cat models/gpt/logs/gpt.pid  (y luego: ps -p <PID>)
# Frenar todo:        kill $(cat models/gpt/logs/gpt.pid)
# El cache parquet es incremental: relanzar retoma donde quedó y la etapa de 200
# se reutiliza al completar las 495.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- 0) chequeos previos ----------------------------------------------------
# La key se carga dentro del script python desde TESIS/.env; acá solo se verifica
# que exista alguna fuente, sin imprimirla jamás.
if [ -z "${OPENAI_API_KEY:-}" ] && ! grep -q '^OPENAI_API_KEY=' ../.env 2>/dev/null; then
  echo "ERROR: no hay OPENAI_API_KEY ni en el entorno ni en ../.env"; exit 1
fi
PY=$(command -v python3)   # el python del sistema tiene el paquete openai; el venv de poetry no
export PYTHONUNBUFFERED=1
MODEL="${OPENAI_MODEL_ID:-gpt-5.4}"
EFFORT="${REASONING_EFFORT:-medium}"
mkdir -p models/gpt/logs
# lock: dos corridas simultáneas escribirían el mismo parquet pisándose los resultados pagos
if [ -f models/gpt/logs/gpt.pid ] && ps -p "$(cat models/gpt/logs/gpt.pid)" >/dev/null 2>&1; then
  echo "ERROR: ya hay una corrida activa (PID $(cat models/gpt/logs/gpt.pid)). Frenarla antes:"
  echo "  kill \$(cat models/gpt/logs/gpt.pid)"; exit 1
fi
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="models/gpt/logs/gpt_${STAMP}.log"
LOG_ABS="$(pwd)/$LOG"
PID_FILE="models/gpt/logs/gpt.pid"
PID_ABS="$(pwd)/$PID_FILE"
export MODEL EFFORT PY

# --- 1) la tanda de corridas -------------------------------------------------
# Orden: smoke barato (12 llamadas, aborta si el parser/params fallan) → etapa 200
# balanceados (gate de costo/NaN) → opcionalmente el test completo (495).
run_all() {
  # el python corre en background + wait para que un TERM al wrapper mate también al
  # proceso en curso (bash difiere los traps mientras hay un hijo en foreground)
  trap 'echo "TERM/INT recibido: frenando el proceso en curso"; kill $(jobs -p) 2>/dev/null; exit 143' TERM INT
  run_py() { "$PY" "$@" & wait $!; }
  echo "== $(date) INICIO GPT ($MODEL, effort=$EFFORT) =="
  echo "-- smoke zero (6 casos) --"
  run_py scripts/09c_llm_gpt.py --mode zero --smoke --openai-model "$MODEL" --reasoning-effort "$EFFORT" --out models/gpt
  echo "-- smoke few16 (6 casos) --"
  run_py scripts/09c_llm_gpt.py --mode few --shots 16 --smoke --openai-model "$MODEL" --reasoning-effort "$EFFORT" --out models/gpt
  echo "-- zero-shot (200 balanceados) --"
  run_py scripts/09c_llm_gpt.py --mode zero --limit-per-class 100 --openai-model "$MODEL" --reasoning-effort "$EFFORT" --out models/gpt
  echo "-- few-shot 16 (200 balanceados) --"
  run_py scripts/09c_llm_gpt.py --mode few --shots 16 --limit-per-class 100 --openai-model "$MODEL" --reasoning-effort "$EFFORT" --out models/gpt
  if [ "${RUN_FULL_495:-0}" = "1" ]; then
    echo "-- zero-shot (test completo 495) --"
    run_py scripts/09c_llm_gpt.py --mode zero --openai-model "$MODEL" --reasoning-effort "$EFFORT" --out models/gpt
    echo "-- few-shot 16 (test completo 495) --"
    run_py scripts/09c_llm_gpt.py --mode few --shots 16 --openai-model "$MODEL" --reasoning-effort "$EFFORT" --out models/gpt
  else
    echo "-- gate: revisar nan_rate/costo de los 200 y relanzar con RUN_FULL_495=1 --"
  fi
  echo "== $(date) FIN =="
}

# --- 2) lanzar en segundo plano, resistente al cierre de terminal ------------
nohup bash -c "set -euo pipefail; MODEL='$MODEL'; EFFORT='$EFFORT'; PY='$PY'; RUN_FULL_495='${RUN_FULL_495:-0}'; $(declare -f run_all); run_all" >"$LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "Lanzado en segundo plano. PID $(cat "$PID_FILE")"
echo "Log:        $LOG_ABS"
echo "Seguir:     tail -f $LOG_ABS"
echo "Frenar:     kill \$(cat $PID_ABS)"
