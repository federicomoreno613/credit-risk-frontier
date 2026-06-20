"""
Codificacion de columnas PII en los datasets del proyecto.

Columnas tratadas:
- ciudad        → LOC_001 ... (01_pagos_cuota)
- tipo_de_negocio → SEC_01 ... (01_pagos_cuota)
- origen        → CANAL_01 ... (01_pagos_cuota)
- sexo          → 0/1/99     (01_pagos_cuota)
- credits_subcategory → SUB_01 ... (02_dataset_modelo)

Los mapeos inversos quedan en data/00_keys/ (NO subir a git).
Los datasets se sobreescriben con los valores codificados.
"""

import json
import random
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW   = BASE / "data" / "01_raw"
PRI   = BASE / "data" / "03_primary"
KEYS  = BASE / "data" / "00_keys"
KEYS.mkdir(exist_ok=True)

SEED = 42


# ── helpers ─────────────────────────────────────────────────────────────────

def _shuffle(values: list, seed: int) -> list:
    """Shuffles a list with fixed seed so codes are not alphabetically guessable."""
    v = values[:]
    random.seed(seed)
    random.shuffle(v)
    return v


def make_mapping(values: list[str], prefix: str, width: int = 3, seed: int = SEED) -> dict[str, str]:
    """
    Returns {original_value: code} where codes are prefix_001, prefix_002 ...
    Order is shuffled (seeded) so the code number does not reveal the category name.
    """
    unique = sorted(set(str(v).strip() for v in values if pd.notna(v)))
    shuffled = _shuffle(unique, seed)
    return {v: f"{prefix}_{str(i+1).zfill(width)}" for i, v in enumerate(shuffled)}


def normalize_negocio(val: str) -> str:
    """Normalize business type: lowercase, strip, collapse whitespace."""
    if pd.isna(val):
        return val
    return re.sub(r"\s+", " ", str(val).strip().lower())


def save_key(mapping: dict, name: str) -> None:
    path = KEYS / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  [KEY] Guardado: {path.name}  ({len(mapping)} valores)")


# ── 01_pagos_cuota.csv ───────────────────────────────────────────────────────

print("\n=== 01_pagos_cuota.csv ===")
cuota_path = RAW / "01_pagos_cuota.csv"
df_cuota = pd.read_csv(cuota_path)
print(f"  Filas: {len(df_cuota):,}  |  Columnas: {list(df_cuota.columns)}")

# --- ciudad → LOC_xxx ---
ciudad_map = make_mapping(df_cuota["ciudad"].dropna().tolist(), prefix="LOC", width=3)
save_key(ciudad_map, "map_ciudad")
df_cuota["ciudad"] = df_cuota["ciudad"].map(ciudad_map)

# --- tipo_de_negocio: normalizar primero, luego codificar ---
df_cuota["tipo_de_negocio"] = df_cuota["tipo_de_negocio"].apply(normalize_negocio)
negocio_map = make_mapping(df_cuota["tipo_de_negocio"].dropna().tolist(), prefix="SEC", width=2)
save_key(negocio_map, "map_tipo_negocio")
df_cuota["tipo_de_negocio"] = df_cuota["tipo_de_negocio"].map(negocio_map)

# --- origen → CANAL_xx ---
origen_map = make_mapping(df_cuota["origen"].dropna().tolist(), prefix="CANAL", width=2)
save_key(origen_map, "map_origen")
df_cuota["origen"] = df_cuota["origen"].map(origen_map)

# --- sexo → 0/1/99 ---
sexo_map = {"F": 0, "M": 1, "D": 99}
save_key({str(k): v for k, v in sexo_map.items()}, "map_sexo")
df_cuota["sexo"] = df_cuota["sexo"].map(sexo_map)

df_cuota.to_csv(cuota_path, index=False)
print(f"  Sobreescrito: {cuota_path.name}")

# ── 01_pagos_mora.csv ────────────────────────────────────────────────────────
# Solo contiene credito_id_anon, max_dias_retraso, target → ya anonimizado, sin accion.
print("\n=== 01_pagos_mora.csv ===")
print("  Sin columnas PII — no se modifica.")

# ── 02_dataset_modelo.csv ────────────────────────────────────────────────────

print("\n=== 02_dataset_modelo.csv ===")
modelo_path = PRI / "02_dataset_modelo.csv"
df_modelo = pd.read_csv(modelo_path)
print(f"  Filas: {len(df_modelo):,}  |  Columnas PII: credits_subcategory")

# --- credits_subcategory → SUB_xx ---
subcat_map = make_mapping(df_modelo["credits_subcategory"].dropna().tolist(), prefix="SUB", width=2)
save_key(subcat_map, "map_credits_subcategory")
df_modelo["credits_subcategory"] = df_modelo["credits_subcategory"].map(subcat_map)

df_modelo.to_csv(modelo_path, index=False)
print(f"  Sobreescrito: {modelo_path.name}")

# ── Resumen final ────────────────────────────────────────────────────────────
print("\n=== Resumen de mapeos generados ===")
for f in sorted(KEYS.glob("*.json")):
    with open(f) as fh:
        m = json.load(fh)
    print(f"  {f.name}: {len(m)} valores")

print("\nListo. Los archivos originales han sido sobreescritos con valores codificados.")
print(f"Claves inversas en: {KEYS}")
