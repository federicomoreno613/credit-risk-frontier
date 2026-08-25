"""Tablero de avance de las corridas LLM — un "top" con barras y ETA.

Lee los JSONL de data/pipeline/razonamientos/ y muestra, por configuración:
barra de progreso, casos válidos, velocidad (casos/min sobre los últimos
registros con timestamp) y tiempo restante estimado.

Uso:
  poetry run python pipeline/estado.py            una foto
  poetry run python pipeline/estado.py --watch    refresca cada 30 s (Ctrl+C sale)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contrato as C  # noqa: E402

ANCHO_BARRA = 32


def barra(hechos: int, total: int) -> str:
    llenos = int(ANCHO_BARRA * hechos / total) if total else 0
    return "█" * llenos + "░" * (ANCHO_BARRA - llenos)


def velocidad_y_eta(registros: list[dict], pendientes: int) -> tuple[str, str]:
    """Casos/min y ETA a partir de los últimos 30 timestamps disponibles."""
    tiempos = sorted(r["ts"] for r in registros if r.get("ts"))
    if len(tiempos) < 2:
        return "-", "-"
    ventana = tiempos[-30:]
    ritmo = (len(ventana) - 1) / max(ventana[-1] - ventana[0], 1e-9)  # casos/seg
    if ritmo <= 0:
        return "-", "-"
    eta_seg = pendientes / ritmo
    horas, resto = divmod(int(eta_seg), 3600)
    return f"{ritmo * 60:.1f}/min", f"{horas}h{resto // 60:02d}m"


def esperadas() -> list[tuple[str, str]]:
    """Configuraciones del contrato: (archivo, etiqueta)."""
    configs = [(f"qwen_{p}_few{s}.jsonl", f"qwen  {p:<23} few{s}")
               for p in C.PERFILES for s in C.SHOTS]
    configs += [(f"gpt_{p}_few{s}.jsonl", f"gpt   {p:<23} few{s}")
                for p in C.PERFILES for s in C.SHOTS]
    return configs


def foto(total_test: int) -> None:
    ahora = time.strftime("%H:%M:%S")
    print(f"=== avance LLM sobre test (n={total_test}) — {ahora} ===")
    hubo_algo = False
    for archivo, etiqueta in esperadas():
        path = C.RAZONAMIENTOS / archivo
        if not path.exists():
            continue
        hubo_algo = True
        registros = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        ids = {str(r["evaluation_id"]) for r in registros}
        hechos = len(ids)
        validos = sum(1 for r in registros if r.get("valida"))
        vel, eta = velocidad_y_eta(registros, total_test - hechos)
        estado = "LISTO " if hechos >= total_test else "corre "
        print(f"{etiqueta}  [{barra(hechos, total_test)}] "
              f"{hechos:>4}/{total_test}  validos={validos:<4} {estado} {vel:>9}  ETA {eta}")
    if not hubo_algo:
        print("(sin corridas todavía: no hay JSONL en data/pipeline/razonamientos/)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="refrescar cada 30 s")
    parser.add_argument("--intervalo", type=int, default=30)
    args = parser.parse_args()

    variables = C.cargar_variables()
    total_test = int((variables["set"] == "test").sum())

    if not args.watch:
        foto(total_test)
        return
    try:
        while True:
            print("\033c", end="")  # limpiar pantalla
            foto(total_test)
            time.sleep(args.intervalo)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
