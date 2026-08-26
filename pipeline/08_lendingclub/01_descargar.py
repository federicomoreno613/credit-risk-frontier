"""Descarga el subset LendingClub de Feng et al. 2023 desde Hugging Face.

Baja los tres parquet (train/valid/test) de `TheFinAI/cra-lendingclub`,
registra sha256 + conteos en `data/lendingclub/manifiesto.json` y elimina
el stub sintético viejo si sigue presente. Reproducible y sin credenciales.

Uso:
    poetry run python pipeline/08_lendingclub/01_descargar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lc  # noqa: E402

STUB_VIEJO = lc.DATA_DIR / "lendingclub_processed.csv"


def descargar(split: str) -> Path:
    destino = lc.parquet_path(split)
    if destino.exists():
        print(f"[skip] {destino.name} ya existe")
        return destino
    url = lc.HF_URL.format(repo=lc.HF_REPO, split=split)
    print(f"[get ] {url}")
    respuesta = requests.get(url, timeout=300)
    respuesta.raise_for_status()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(respuesta.content)
    return destino


def main() -> None:
    if STUB_VIEJO.exists():
        STUB_VIEJO.unlink()
        print(f"[del ] stub sintético eliminado: {STUB_VIEJO}")

    manifiesto = {"fuente": lc.HF_REPO, "archivos": {}}
    for split in lc.SPLITS:
        path = descargar(split)
        frame = pd.read_parquet(path)
        # Validación mínima del contrato: columnas y parseo de los 21 atributos.
        assert {"query", "answer", "gold", "text"} <= set(frame.columns), frame.columns
        lc.parsear_texto(frame["text"].iloc[0])
        manifiesto["archivos"][split] = {
            "sha256": lc.sha256(path),
            "filas": int(len(frame)),
            "tasa_bad": round(float(frame["gold"].mean()), 4),
        }
        print(f"[ok  ] {split}: {len(frame)} filas, tasa_bad={frame['gold'].mean():.3f}")

    lc.guardar_json(lc.MANIFIESTO, manifiesto)
    print(f"[ok  ] manifiesto -> {lc.MANIFIESTO}")
    print("siguiente: poetry run python pipeline/08_lendingclub/02_contaminacion.py")


if __name__ == "__main__":
    main()
