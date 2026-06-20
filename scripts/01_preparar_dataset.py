"""
01_preparar_dataset.py
======================
Prepara el dataset final para la tesis: LLMs vs. ML Clásico para Credit Scoring.

Fuente: ../../credit-risk-frontier/data/04_model_input/dataset_limpio.csv
        (5.351 filas × 115 cols — ya procesado y limpio)

Pasos:
1. Cargar dataset limpio
2. Eliminar columnas duplicadas y problemáticas
3. Codificar columnas sensibles (estrategia comercial) al estilo 03_codificar_pii.py
4. Agregar columnas de texto original para el LLM (join con fuente vía hash)
5. Verificar ausencia de PII en nombres de columna
6. Guardar dataset_tesis.csv + mapeos inversos en 00_keys/

Columnas de texto para LLM (Opción B — representación nativa):
  - subcategoria_texto: subcategoría del negocio (100% cobertura)
  - descripcion_negocio: descripción libre del negocio (88% cobertura)
  - otra_categoria_negocio: texto libre opcional del solicitante (23% cobertura)
  - tipo_credito: "Primer Crédito" / "Segundo Crédito" (100% cobertura)
XGBoost usa las columnas numéricas + one-hot existentes. El LLM recibe estas columnas de texto.
"""

import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

# ── Rutas ───────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "../credit-risk-frontier/data/04_model_input/dataset_limpio.csv"
SOURCE = BASE / "../credit-risk-frontier/data/03_primary/results_with_clip_clusters_transunion.csv"
OUTPUT = BASE / "data" / "dataset_tesis.csv"
KEYS = BASE / "data" / "00_keys"
KEYS.mkdir(parents=True, exist_ok=True)

SEED = 42
ANON_SALT = b"tesis_uba_credit_risk_anon_2025"


# ── Helpers (misma lógica que 03_codificar_pii.py) ──────────────────────────

def _shuffle(values: list, seed: int) -> list:
    """Shuffles a list with fixed seed so codes are not alphabetically guessable."""
    v = values[:]
    random.seed(seed)
    random.shuffle(v)
    return v


def make_mapping(values: list[str], prefix: str, width: int = 2, seed: int = SEED) -> dict[str, str]:
    """
    Returns {original_col: new_col} donde new_col = prefix_01, prefix_02, ...
    El orden está shuffled (seeded) para que el número no revele el nombre original.
    """
    unique = sorted(set(str(v).strip() for v in values if pd.notna(v)))
    shuffled = _shuffle(unique, seed)
    return {v: f"{prefix}_{str(i + 1).zfill(width)}" for i, v in enumerate(shuffled)}


def save_key(mapping: dict, name: str) -> None:
    path = KEYS / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  [KEY] Guardado: {path.name}  ({len(mapping)} valores)")


# ── 1. Cargar ────────────────────────────────────────────────────────────────

print("\n=== 01_preparar_dataset.py ===")
print(f"Cargando: {INPUT.resolve()}")
df = pd.read_csv(INPUT)
print(f"  Input:  {df.shape[0]:,} filas × {df.shape[1]} cols")


# ── 2. Eliminar columnas duplicadas y problemáticas ──────────────────────────

# Las 15 columnas category_* con tildes y espacios son duplicado de las snake_case
category_accentuated = [
    "category_Agricultura y Granja",
    "category_Artesanías y Arte",
    "category_Bar y Licores",
    "category_Belleza e Higiene",
    "category_Carnicería y Avícola",
    "category_Comercialización",
    "category_Confección",
    "category_Droguería y Medicina",
    "category_Electrónica y Tecnología",
    "category_Mercado y Perecederos",
    "category_Miscelánea y Papelería",
    "category_Restaurantes y Comidas",
    "category_Servicios",
    "category_Servicios y Alquiler",
    "category_Transporte y Logística",
]

# Columnas adicionales a eliminar
extra_drop = [
    "aniomes",                              # derivado de fecha_desembolso, redundante
    "shops_monthly_outcomes/shops_monthly_incomes",  # nombre con '/' problemático; info en cost_ingress_ratio
]

cols_to_drop = category_accentuated + extra_drop
# Solo eliminar las que existen (robustez)
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop)
print(f"  Eliminadas {len(cols_to_drop)} columnas: {len(category_accentuated)} dups categoría + {len(extra_drop)} problemáticas")


# ── 3. Codificación de columnas de estrategia comercial ─────────────────────

# --- credits_bulk_load_source_* → CANAL_xx ---
bulk_cols = sorted([c for c in df.columns if c.startswith("credits_bulk_load_source_")])
bulk_map = make_mapping(bulk_cols, prefix="CANAL", width=2, seed=SEED)
save_key(bulk_map, "map_bulk_source")
df = df.rename(columns=bulk_map)
print(f"  Renombradas {len(bulk_map)} columnas credits_bulk_load_source_* → CANAL_xx")

# --- alianzas_* → ALIANZA_xx ---
alianza_cols = sorted([c for c in df.columns if c.startswith("alianzas_")])
alianza_map = make_mapping(alianza_cols, prefix="ALIANZA", width=2, seed=SEED)
save_key(alianza_map, "map_alianzas")
df = df.rename(columns=alianza_map)
print(f"  Renombradas {len(alianza_map)} columnas alianzas_* → ALIANZA_xx")


# ── 4. Agregar columnas de texto para el LLM ────────────────────────────────
# Join con el dataset fuente vía hash SHA-256 (misma función que usó 04_clean_dataset.py)
# XGBoost: ignora estas columnas y usa las numéricas/one-hot
# LLM (Qwen3-8B): usa estas columnas para construir el prompt semántico

def anon_hash(val: str) -> str:
    s = str(val).strip()
    return hashlib.sha256(ANON_SALT + s.encode("utf-8")).hexdigest()[:12]


print("\n  Cargando texto original desde fuente...")
src = pd.read_csv(
    SOURCE,
    usecols=["credits_credit_id", "credits_subcategory",
             "shops_description_x", "shops_other_category", "tipo_de_credito"],
    low_memory=False,
)
src["credito_id_anon"] = src["credits_credit_id"].apply(anon_hash)
# Deduplicar por crédito (hay una fila por crédito en la fuente)
src = src.drop_duplicates("credito_id_anon")

# Limpiar subcategoría: quitar prefijo y underscores → texto legible
src["subcategoria_texto"] = (
    src["credits_subcategory"]
    .str.replace("subcategory_", "", regex=False)
    .str.replace("_", " ", regex=False)
    .str.strip()
)

text_cols = src[["credito_id_anon", "subcategoria_texto",
                  "shops_description_x", "shops_other_category", "tipo_de_credito"]]
text_cols = text_cols.rename(columns={
    "shops_description_x":  "descripcion_negocio",
    "shops_other_category": "otra_categoria_negocio",
    "tipo_de_credito":      "tipo_credito",
})

df = df.merge(text_cols, on="credito_id_anon", how="left")

# Reporte de cobertura
for col in ["subcategoria_texto", "descripcion_negocio", "otra_categoria_negocio", "tipo_credito"]:
    n = df[col].notna().sum()
    print(f"  {col}: {n:,} / {len(df):,} ({n/len(df)*100:.0f}%)")


# ── 5. Verificación de PII en nombres de columna ────────────────────────────

# Patrones que podrían indicar PII residual en nombre de columna
pii_patterns = re.compile(
    r"\b(nombre|telefono|email|correo|cedula|documento|dni|nit|"
    r"apellido|direccion|address|phone|name|contact_name)\b",
    re.IGNORECASE,
)
pii_cols = [c for c in df.columns if pii_patterns.search(c)]
if pii_cols:
    print(f"\n  [ADVERTENCIA] Posibles columnas PII detectadas: {pii_cols}")
else:
    print("  [OK] Sin columnas PII detectadas en nombres de columna")

# Resumen de nulos por grupo
print("\n  Nulos por grupo de variables:")
groups = {
    "Meta":       ["credito_id_anon", "fecha_desembolso", "target", "set"],
    "TransUnion": [c for c in df.columns if c in [
        "agg308", "wd81", "agg2503", "utlmag04", "duemag01", "aepmag01",
        "bi21s", "lmd34s", "ri27s", "rle904", "tel32s", "tranbal09",
        "at104s", "sa21s", "at103s", "tel03s", "at34af", "g051s", "agg9316", "wd03",
    ]],
    "Formulario": [c for c in df.columns if c.startswith("appusers_") or c.startswith("shops_") or
                   c.startswith("credits_amount") or c.startswith("credits_fee") or
                   c in ["credits_interest_amount", "credits_surety_bond_amount",
                          "credits_digital_instrumentation_amount", "credits_dependants_amount",
                          "credits_family_expenses", "antiguedad_cliente", "estimated_income"]],
    "Derivadas":  [c for c in df.columns if c in [
        "free_cash_flow", "cost_ingress_ratio", "debts_savings",
        "score_debets", "relacion_edad_deuda", "appusers_score",
    ]],
}
for gname, gcols in groups.items():
    gcols_exist = [c for c in gcols if c in df.columns]
    if gcols_exist:
        total_null = df[gcols_exist].isnull().sum().sum()
        print(f"    {gname}: {len(gcols_exist)} cols | {total_null:,} nulos totales")


# ── 5. Guardar ───────────────────────────────────────────────────────────────

df.to_csv(OUTPUT, index=False)

# Resumen final
n_default = int(df["target"].sum())
n_total = len(df)
default_rate = n_default / n_total * 100

sets = df["set"].value_counts().to_dict()
train_n = sets.get("train", 0)
val_n = sets.get("val", 0)
test_n = sets.get("test", 0)

n_text_cols = 4  # subcategoria_texto, descripcion_negocio, otra_categoria_negocio, tipo_credito
print(f"\n{'=' * 55}")
print(f"✓ Guardado: data/dataset_tesis.csv | {n_total:,} filas × {df.shape[1]} cols")
print(f"  ({df.shape[1] - n_text_cols} numéricas/one-hot para XGBoost + {n_text_cols} texto para LLM)")
print(f"  Tasa de default: {default_rate:.1f}% ({n_default:,} / {n_total:,})")
print(f"  Split — train: {train_n:,} | val: {val_n:,} | test: {test_n:,}")
print(f"  Mapeos en: data/00_keys/")
print(f"{'=' * 55}")
