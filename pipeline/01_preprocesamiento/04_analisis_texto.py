"""Paso 04 — Análisis profundo del texto de los créditos.

El texto (descripción del negocio, rubro, objetivo del crédito) es el insumo
diferencial de los perfiles ``tu_form_description`` y ``_full``: este paso
cuantifica QUÉ hay en ese texto y CUÁNTA señal predictiva lleva por sí solo,
para fundamentar la discusión de los experimentos LLM.

Análisis (todo sobre la cohorte v2, split del contrato):
  1. Cobertura por columna de texto (global, por partición, por segmento).
  2. Longitud y cardinalidad (¿texto libre real o plantillas repetidas?).
  3. Tasa de mora por rubro (train, con n mínimo).
  4. Tokens asociados a mora/no-mora (log-odds suavizado, train).
  5. Señal predictiva del texto solo: TF-IDF + Regresión Logística
     (train→test), por fuente y combinado, global y por segmento.
  6. La ausencia de texto como señal (indicadores de missingness).

Salida: docs/ANALISIS_TEXTO_CREDITOS.md (tablas en Markdown, reproducible).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402

SALIDA = C.ROOT / "docs" / "ANALISIS_TEXTO_CREDITOS.md"
COLUMNAS_TEXTO = [
    "descripcion_negocio",
    "subcategoria_texto",
    "tipo_credito",
    "otra_categoria_negocio",
    "objetivo_credito",
]
# Texto libre real (los otros son categóricos o binarios)
TEXTO_LIBRE = ["descripcion_negocio", "objetivo_credito", "otra_categoria_negocio"]

STOPWORDS_ES = {
    "de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "para", "con",
    "por", "que", "se", "del", "al", "es", "mi", "su", "sus", "como", "más",
    "mas", "o", "u", "e", "lo", "le", "me", "no", "si", "sí", "ya", "este",
    "esta", "esto", "son", "hay", "muy", "también", "tambien", "todo", "toda",
    "todos", "todas", "otro", "otra", "otros", "otras", "donde", "cuando",
    "tengo", "tiene", "tenemos", "hace", "hacer", "años", "año", "está",
    "esta", "estamos", "estoy", "ser", "fue", "he", "ha", "han",
}


def tokenizar(texto: str) -> list[str]:
    return [t for t in re.findall(r"[a-záéíóúñü]+", str(texto).lower())
            if len(t) >= 3 and t not in STOPWORDS_ES]


def tabla_md(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """DataFrame → tabla Markdown sin dependencias extra."""
    df = frame.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda v: "—" if pd.isna(v) else floatfmt.format(v))
    filas = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    for _, fila in df.iterrows():
        filas.append("| " + " | ".join(str(v) for v in fila) + " |")
    return "\n".join(filas)


def con_texto(serie: pd.Series) -> pd.Series:
    return serie.notna() & (serie.astype(str).str.strip() != "")


# ---------------------------------------------------------------------------
# 1. Cobertura
# ---------------------------------------------------------------------------

def cobertura(datos: pd.DataFrame) -> str:
    filas = []
    for col in COLUMNAS_TEXTO:
        tiene = con_texto(datos[col])
        fila = {"columna": col, "global": tiene.mean()}
        for s in ("train", "val", "test"):
            fila[s] = tiene[datos["set"] == s].mean()
        for seg in ("esparso", "denso"):
            fila[seg] = tiene[datos["segmento"] == seg].mean()
        tr = datos["set"] == "train"
        fila["mora (train)"] = tiene[tr & (datos["target"] == 1)].mean()
        fila["no mora (train)"] = tiene[tr & (datos["target"] == 0)].mean()
        filas.append(fila)
    return tabla_md(pd.DataFrame(filas))


# ---------------------------------------------------------------------------
# 2. Longitud y cardinalidad
# ---------------------------------------------------------------------------

def longitud_cardinalidad(datos: pd.DataFrame) -> str:
    filas = []
    for col in COLUMNAS_TEXTO:
        vals = datos.loc[con_texto(datos[col]), col].astype(str).str.strip()
        palabras = vals.str.split().str.len()
        filas.append({
            "columna": col,
            "n con texto": len(vals),
            "únicos": vals.nunique(),
            "% repetido": 1 - vals.nunique() / max(len(vals), 1),
            "palabras (mediana)": float(palabras.median()) if len(vals) else np.nan,
            "palabras (p90)": float(palabras.quantile(0.9)) if len(vals) else np.nan,
            "caracteres (mediana)": float(vals.str.len().median()) if len(vals) else np.nan,
        })
    tabla = tabla_md(pd.DataFrame(filas))

    top = (datos.loc[con_texto(datos["descripcion_negocio"]), "descripcion_negocio"]
           .astype(str).str.strip().str.lower().value_counts().head(8))
    repetidas = "\n".join(f"- ({n}×) «{t[:90]}»" for t, n in top.items())
    return tabla + "\n\nDescripciones más repetidas (¿plantillas?):\n\n" + repetidas


# ---------------------------------------------------------------------------
# 3. Mora por rubro (train)
# ---------------------------------------------------------------------------

def mora_por_rubro(train: pd.DataFrame, n_min: int = 50) -> str:
    g = (train.groupby("subcategoria_texto")["target"]
         .agg(n="size", tasa_mora="mean").reset_index())
    g = g[g["n"] >= n_min].sort_values("tasa_mora", ascending=False)
    base = train["target"].mean()
    partes = pd.concat([g.head(8), g.tail(8)]).drop_duplicates("subcategoria_texto")
    return (f"Tasa de mora base en train: {base:.3f} (rubros con n ≥ {n_min}).\n\n"
            + tabla_md(partes.reset_index(drop=True)))


# ---------------------------------------------------------------------------
# 4. Tokens asociados a mora (log-odds suavizado)
# ---------------------------------------------------------------------------

def tokens_distintivos(train: pd.DataFrame, col: str, n_min: int = 25) -> str:
    mora = Counter()
    sana = Counter()
    for _, fila in train.iterrows():
        toks = set(tokenizar(fila[col])) if pd.notna(fila[col]) else set()
        (mora if fila["target"] == 1 else sana).update(toks)
    total_m, total_s = sum(mora.values()), sum(sana.values())
    filas = []
    for tok in set(mora) | set(sana):
        n = mora[tok] + sana[tok]
        if n < n_min:
            continue
        lo = np.log((mora[tok] + 1) / (total_m + 1)) - np.log((sana[tok] + 1) / (total_s + 1))
        filas.append({"token": tok, "n": n, "log_odds_mora": lo})
    df = pd.DataFrame(filas).sort_values("log_odds_mora", ascending=False)
    if df.empty:
        return "_Sin tokens con frecuencia suficiente._"
    ambos = pd.concat([df.head(10), df.tail(10)])
    return tabla_md(ambos.reset_index(drop=True))


# ---------------------------------------------------------------------------
# 5. Señal predictiva del texto solo (TF-IDF + LogReg, train→test)
# ---------------------------------------------------------------------------

def auc_texto(train: pd.DataFrame, test: pd.DataFrame) -> str:
    fuentes = {
        "descripcion_negocio": ["descripcion_negocio"],
        "objetivo_credito": ["objetivo_credito"],
        "rubro + tipo": ["subcategoria_texto", "tipo_credito"],
        "todo el texto": COLUMNAS_TEXTO,
    }
    filas = []
    for nombre, cols in fuentes.items():
        juntar = lambda d: (d[cols].fillna("").astype(str)
                            .agg(" ".join, axis=1).str.lower())
        vec = TfidfVectorizer(tokenizer=tokenizar, token_pattern=None,
                              ngram_range=(1, 2), min_df=5, sublinear_tf=True)
        Xtr = vec.fit_transform(juntar(train))
        modelo = LogisticRegression(max_iter=2000, C=1.0)
        modelo.fit(Xtr, train["target"])
        prob = modelo.predict_proba(vec.transform(juntar(test)))[:, 1]
        fila = {"fuente": nombre, "features": Xtr.shape[1],
                "AUC test": roc_auc_score(test["target"], prob)}
        for seg in ("esparso", "denso"):
            m = (test["segmento"] == seg).to_numpy()
            fila[f"AUC {seg}"] = (roc_auc_score(test.loc[m, "target"], prob[m])
                                  if m.sum() > 30 else np.nan)
        filas.append(fila)
    return tabla_md(pd.DataFrame(filas))


# ---------------------------------------------------------------------------
# 6. La ausencia de texto como señal
# ---------------------------------------------------------------------------

def missingness(train: pd.DataFrame, test: pd.DataFrame) -> str:
    filas = []
    for col in TEXTO_LIBRE:
        ind_tr = con_texto(train[col]).astype(int)
        ind_te = con_texto(test[col]).astype(int)
        if ind_te.nunique() < 2:
            continue
        filas.append({
            "columna": col,
            "AUC test (solo indicador tiene/no tiene)":
                roc_auc_score(test["target"], ind_te),
            "% mora con texto (train)": train.loc[ind_tr == 1, "target"].mean(),
            "% mora sin texto (train)": train.loc[ind_tr == 0, "target"].mean(),
        })
    return tabla_md(pd.DataFrame(filas))


def main() -> None:
    datos = C.cargar_variables()
    train, _, test = C.dividir(datos)

    secciones = [
        "# Análisis del texto de los créditos (cohorte v2)",
        f"\nGenerado por `pipeline/01_preprocesamiento/04_analisis_texto.py` "
        f"sobre `02_variables.parquet` ({len(datos)} créditos; train "
        f"{len(train)} / test {len(test)}). Todo lo entrenado usa solo train.\n",
        "## 1. Cobertura\n\n" + cobertura(datos),
        "\n## 2. Longitud, cardinalidad y plantillas\n\n" + longitud_cardinalidad(datos),
        "\n## 3. Tasa de mora por rubro (train)\n\n" + mora_por_rubro(train),
        "\n## 4. Tokens asociados a mora — descripción del negocio (train)\n\n"
        + tokens_distintivos(train, "descripcion_negocio"),
        "\n## 5. Señal predictiva del texto solo (TF-IDF + LogReg, train→test)\n\n"
        + auc_texto(train, test),
        "\n## 6. La ausencia de texto como señal\n\n" + missingness(train, test),
    ]
    SALIDA.write_text("\n".join(secciones) + "\n", encoding="utf-8")
    print(f"escrito {SALIDA}")


if __name__ == "__main__":
    main()
