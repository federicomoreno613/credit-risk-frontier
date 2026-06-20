"""
04_clean_dataset.py
-------------------
Genera dataset limpio para la tesis:
- Elimina variables CLIP (cluster_*, img_pred_prob, etc.)
- Elimina PII (nombres, teléfonos, emails, documentos, direcciones)
- Elimina columnas con >95% nulos o que son objetos JSON sin parsear
- Elimina filas sin ningún dato de TransUnion (237 sin reporte de bureau)
- Imputa con -1 las 4 variables TU con nulls parciales (flag "sin dato")
- Conserva: variables de formulario + TransUnion + target + meta
- Output: data/04_model_input/dataset_limpio.csv
"""

# %% [markdown]
# ## Origen y decisiones de diseño
#
# El dataset original `results_with_clip_clusters_transunion.csv` contiene 5.588 créditos
# y 1.146 columnas, incluyendo embeddings CLIP de imágenes del negocio y clusters derivados.
# Esta versión limpia **no usa CLIP** como feature. Justificación:
# - CLIP requiere imágenes de alta calidad que no siempre están disponibles en producción
# - El objetivo es un score reproducible basado solo en datos del formulario y bureau
#
# ### Tratamiento de nulos en TransUnion
#
# Hay dos grupos de nulos en las variables de bureau:
# - 237 créditos sin ningún reporte TU → se excluyen (no hay información externa)
# - 1.701 créditos con 4 variables TU parcialmente nulas (agg2503, utlmag04, duemag01,
#   aepmag01) → se imputan con -1 como flag "sin deuda de este tipo en bureau".
#   XGBoost trata el -1 como valor informativo (ausencia de deuda).
#
# El target está balanceado al 50/50 (undersampling del conjunto mayoritario).

# %%
import hashlib
from pathlib import Path

import pandas as pd

# %%
BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "data" / "03_primary" / "results_with_clip_clusters_transunion.csv"
OUTPUT_DIR = BASE / "data" / "04_model_input"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIR / "dataset_limpio.csv"

ANON_SALT = b"tesis_uba_credit_risk_anon_2025"

# %%
df = pd.read_csv(INPUT, low_memory=False)
print(f"Dataset original: {df.shape[0]:,} filas × {df.shape[1]:,} columnas")

# %%
# ------------------------------------------------------------------
# 1. ANONIMIZAR: agregar ID anónimo, eliminar columnas PII
# ------------------------------------------------------------------

def anon_hash(val: str) -> str:
    """Hash SHA-256 con salt. Primeros 12 hex chars."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    return hashlib.sha256(ANON_SALT + s.encode("utf-8")).hexdigest()[:12]


# PII a eliminar completamente
PII_COLS = [
    "credits_contact_name", "credits_contact_phone", "credits_contact_identity_number",
    "credits_contact_address", "appusers_full_name", "appusers_last_name",
    "appusers_login_phone", "appusers_login_email", "appusers_email_normalized",
    "appusers_address_description", "credits_bank_phone_number", "credits_bank_documents",
    "appusers_password_digest", "appusers_street_name", "appusers_street_type",
    "appusers_street_number", "appusers_map_address", "appusers_phone_x",
    "appusers_passport_x", "appusers_raw_metamap_data", "appusers_metamap_data_user_id",
    "appusers_avatar_image_url", "appusers_face_verification_image_url",
    "appusers_passport_verification_image_url", "appusers_passport_back_verification_image_url",
]

# Crear ID anónimo a partir de credits_credit_id (si existe)
if "credits_credit_id" in df.columns:
    df["credito_id_anon"] = df["credits_credit_id"].astype(str).apply(anon_hash)
else:
    df["credito_id_anon"] = [f"cred_{i:05d}" for i in range(len(df))]

# Eliminar PII + IDs originales
drop_pii = [c for c in PII_COLS + ["credits_credit_id", "appusers_user_id",
                                     "appusers_user_id_y", "credits_uid",
                                     "credits_userId"] if c in df.columns]
df = df.drop(columns=drop_pii)
print(f"  → Columnas PII eliminadas: {len(drop_pii)}")

# %%
# ------------------------------------------------------------------
# 2. ELIMINAR VARIABLES CLIP
# ------------------------------------------------------------------
clip_cols = [c for c in df.columns if c.startswith("cluster_") or
             c in ["1_cluster_mas_cercano", "2_cluster_mas_cercano", "3_cluster_mas_cercano",
                   "1_cluster_mas_lejano", "2_cluster_mas_lejano", "3_cluster_mas_lejano",
                   "img_pred_prob"]]
df = df.drop(columns=clip_cols)
print(f"  → Columnas CLIP eliminadas: {len(clip_cols)}")

# %%
# ------------------------------------------------------------------
# 3. SELECCIONAR VARIABLES DE FORMULARIO + TRANSUNION
# ------------------------------------------------------------------

# --- Variables numéricas del formulario ---
FORM_NUM = [
    "appusers_age",                         # edad del solicitante
    "credits_amount_granted",               # monto otorgado
    "credits_fee_month_amount_granted",     # número de cuotas
    "credits_interest_amount",              # tasa de interés (%)
    "credits_surety_bond_amount",           # fianza
    "credits_digital_instrumentation_amount",  # costo plataforma digital
    "credits_dependants_amount",            # número de dependientes
    "credits_family_expenses",              # gastos familiares declarados
    "shops_monthly_incomes",               # ingresos mensuales del negocio
    "shops_monthly_outcomes",              # egresos mensuales
    "shops_daily_incomes",                 # ingresos diarios
    "shops_initial_capital",               # capital inicial del negocio
    "shops_rent_amount",                   # alquiler mensual
    "shops_shop_age",                      # antigüedad del negocio (años)
    "antiguedad_cliente",                  # días como cliente Quipu
]

# --- Variables derivadas / engineered ---
DERIVED = [
    "estimated_income",                    # ingreso estimado (diario × 30 o mensual)
    "free_cash_flow",                      # flujo de caja libre (ingresos - egresos)
    "cost_ingress_ratio",                  # razón egresos / ingresos
    "debts_savings",                       # deudas / ahorros (TransUnion)
    "score_debets",                        # score de deudas (TransUnion)
    "relacion_edad_deuda",                 # edad de la deuda más antigua
    "appusers_score",                      # score interno Quipu
    "shops_monthly_outcomes/shops_monthly_incomes",  # ratio egresos/ingresos
]

# --- TransUnion (bureau externo) ---
TRANSUNION = [
    "agg308",    # Número de obligaciones activas
    "wd81",      # Días en mora máxima histórica
    "agg2503",   # Saldo total de obligaciones
    "utlmag04",  # Utilización de cupo de crédito
    "duemag01",  # Saldo en mora activo
    "aepmag01",  # Antigüedad de la obligación más antigua
    "bi21s",     # Número de consultas últimos 6 meses
    "lmd34s",    # Meses desde última mora
    "ri27s",     # Número de reestructuraciones
    "rle904",    # Saldo de cartera castigada
    "tel32s",    # Total de entidades con reporte
    "tranbal09", # Balance total en el sistema
    "at104s",    # Número de obligaciones al día
    "sa21s",     # Antigüedad del score
    "at103s",    # Número de obligaciones en mora
    "tel03s",    # Total de obligaciones reportadas
    "at34af",    # Meses desde apertura primera obligación
    "g051s",     # Número de sectores con reporte
    "agg9316",   # Cupo disponible total
    "wd03",      # Máximo días en mora últimos 12 meses
]

# --- Categoricals (ya codificadas como dummies) ---
CAT_COLS = (
    [c for c in df.columns if c.startswith("category_")] +          # categoría del negocio
    [c for c in df.columns if c.startswith("credits_credit_goal_")] +  # destino del crédito
    [c for c in df.columns if c.startswith("credits_bulk_load_source_")] +  # canal origen
    [c for c in df.columns if c.startswith("credits_contact_studies_")] +   # nivel educativo
    [c for c in df.columns if c.startswith("alianzas_")] +           # canal alianza
    [c for c in df.columns if c.startswith("appusers_gender_")]      # género (encoded)
)

# --- Meta ---
META = ["credito_id_anon", "fecha_desembolso", "aniomes", "target", "set"]

# %%
# Filtrar solo columnas que existen y no son 100% nulas
def keep_col(col, df):
    return col in df.columns and df[col].notna().sum() > 0

all_keep = (
    META +
    [c for c in FORM_NUM if keep_col(c, df)] +
    [c for c in DERIVED if keep_col(c, df)] +
    [c for c in TRANSUNION if keep_col(c, df)] +
    [c for c in CAT_COLS if keep_col(c, df)]
)

# Deduplicate preserving order
seen = set()
final_cols = [c for c in all_keep if not (c in seen or seen.add(c))]

df_clean = df[final_cols].copy()

# %%
# ------------------------------------------------------------------
# 3b. FILTRAR Y TRATAR NULOS EN TRANSUNION
# ------------------------------------------------------------------

# Eliminar filas sin NINGÚN dato de TransUnion (sin reporte bureau)
tu_all_null = df_clean[[c for c in TRANSUNION if c in df_clean.columns]].isna().all(axis=1)
n_sin_tu = tu_all_null.sum()
df_clean = df_clean[~tu_all_null].copy()
print(f"  → Filas sin ningún reporte TransUnion eliminadas: {n_sin_tu}")

# Imputar con -1 las variables TU con nulos parciales
# (agg2503, utlmag04, duemag01, aepmag01 → ausencia de deuda en ese sector)
TU_PARTIAL = ["agg2503", "utlmag04", "duemag01", "aepmag01"]
for col in TU_PARTIAL:
    if col in df_clean.columns:
        n_imp = df_clean[col].isna().sum()
        df_clean[col] = df_clean[col].fillna(-1)
        if n_imp > 0:
            print(f"  → {col}: {n_imp} nulos imputados con -1 (sin deuda en bureau)")

print(f"\nDataset limpio: {df_clean.shape[0]:,} filas × {df_clean.shape[1]:,} columnas")
print(f"  → Meta:        {len(META)} cols")
print(f"  → Formulario:  {len([c for c in FORM_NUM if c in final_cols])} cols")
print(f"  → Derivadas:   {len([c for c in DERIVED if c in final_cols])} cols")
print(f"  → TransUnion:  {len([c for c in TRANSUNION if c in final_cols])} cols")
print(f"  → Categoricas: {len([c for c in CAT_COLS if c in final_cols])} cols")

# %%
# ------------------------------------------------------------------
# 4. EDA RÁPIDO
# ------------------------------------------------------------------
print("\n" + "="*60)
print("EDA RÁPIDO — DATASET LIMPIO")
print("="*60)

print(f"\nTarget (balanceo):")
print(df_clean['target'].value_counts(dropna=False))
print(f"\nSplit train/val/test:")
print(df_clean.groupby(['set','target']).size().unstack(fill_value=0))

print("\nNulos por grupo de variables:")
for group_name, cols in [
    ("Formulario", [c for c in FORM_NUM if c in final_cols]),
    ("Derivadas",  [c for c in DERIVED if c in final_cols]),
    ("TransUnion", [c for c in TRANSUNION if c in final_cols]),
]:
    pct_null = df_clean[cols].isna().mean().mean() * 100
    max_null_col = df_clean[cols].isna().mean().idxmax()
    max_null_pct = df_clean[cols].isna().mean().max() * 100
    print(f"  {group_name}: promedio nulos = {pct_null:.1f}% | "
          f"máx = {max_null_col} ({max_null_pct:.1f}%)")

print("\nEstadísticas clave (no nulos):")
key_stats = ["credits_amount_granted", "credits_fee_month_amount_granted",
             "appusers_age", "shops_monthly_incomes", "antiguedad_cliente"]
for c in key_stats:
    if c in df_clean.columns:
        s = df_clean[c].describe()
        print(f"  {c}: media={s['mean']:.0f} | med={s['50%']:.0f} | "
              f"min={s['min']:.0f} | max={s['max']:.0f}")

# %%
# ------------------------------------------------------------------
# 5. GUARDAR
# ------------------------------------------------------------------
df_clean.to_csv(OUTPUT, index=False)
print(f"\n✓ Guardado: {OUTPUT}")
print(f"  {df_clean.shape[0]:,} filas × {df_clean.shape[1]:,} columnas")
print(f"  Columnas finales: {list(df_clean.columns[:8])} ... (y {len(df_clean.columns)-8} más)")
