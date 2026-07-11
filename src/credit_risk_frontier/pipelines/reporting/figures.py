"""Nodos de figuras del reporting — cada uno construye y RETORNA una Figure.

Kedro persiste vía ``matplotlib.MatplotlibDataset`` (ningún ``savefig`` en el nodo).
Las figuras leen los AUC de los datasets de métricas del catálogo — nada hardcodeado.
fig3/fig4 corrigen los dos bugs del script 10: la clave ``"AUC"`` (no ``"auc"``, que
pintaba "pend." para el LLM) y los AUC clásicos leídos (baseline no-leak citable).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_XGB, C_LOG, C_LLM, C_TABFM, GREY = "#55A868", "#4C72B0", "#C44E52", "#00A2A2", "#8C8C8C"


def _auc(metrics: dict, seg: str = "total"):
    """AUC en test con la clave correcta 'AUC' (fix del bug de 10).

    Maneja las dos estructuras del proyecto:
    * clásicos (03/04): ``test`` es directamente ``{AUC, Gini, ...}`` (sin segmentos).
    * LLM (06b/08/09): ``test`` es ``{total|esparso|denso: {AUC, ...}}``.
    """
    if not metrics:
        return None
    test = metrics.get("test", {})
    if seg == "total" and "AUC" in test:      # estructura clásica (test plano)
        return test["AUC"]
    node = test.get(seg, {})
    return node.get("AUC") if isinstance(node, dict) else None


def _cv_auc(cv_seg: dict, model: str, seg: str):
    """AUC media del CV temporal por segmento (script 05)."""
    node = cv_seg.get(model, {}).get(seg, {})
    return node.get("AUC", {}).get("mean") if isinstance(node.get("AUC"), dict) else None


# ---------------------------------------------------------------------------
# fig3 — AUC del LLM según nº de ejemplos, por segmento (bug fix)
# ---------------------------------------------------------------------------

def plot_fig3_auc_ejemplos(llm_think_zero, llm_think_few16, llm_think_few32,
                           llm_fewshot_combined, xgb_metrics):
    """AUC del LLM razonado vs nº de ejemplos, con la referencia XGBoost no-leak."""
    xgb_total = _auc(xgb_metrics)
    configs = [("0", llm_think_zero), ("16", llm_think_few16), ("32", llm_think_few32)]
    xs = [c[0] for c in configs]
    tot = [_auc(c[1], "total") for c in configs]
    esp = [_auc(c[1], "esparso") for c in configs]
    den = [_auc(c[1], "denso") for c in configs]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for ys, lab, col in [(tot, "Toda la cartera", GREY),
                         (esp, "Historial escaso", C_LLM),
                         (den, "Historial denso", C_LOG)]:
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, label=lab, lw=1.6, ms=5)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.text(len(xs) - 0.95, 0.505, "azar", fontsize=6, color=GREY, va="bottom")
    if xgb_total:
        ax.axhline(xgb_total, color=C_XGB, ls="--", lw=1.2)
        ax.text(len(xs) - 0.95, xgb_total, "XGBoost", fontsize=6, color=C_XGB, va="center")
    ax.set_xlabel("Número de ejemplos en contexto")
    ax.set_ylabel("AUC en test")
    ax.set_title("Los ejemplos no rescatan al modelo de lenguaje:\n"
                 "sigue cerca del azar en todos los segmentos", fontsize=8, loc="left")
    ax.set_ylim(0.30, 0.82)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig4 — comparación por modelo y segmento (bug fix: AUC leídos, no hardcodeados)
# ---------------------------------------------------------------------------

def plot_fig4_comparacion(llm_think_few16, xgb_metrics, logreg_metrics, cv_seg_metrics):
    """Barras AUC por segmento: XGBoost/LogReg (no-leak) vs LLM razonado 16 ejemplos."""
    xgb = [_auc(xgb_metrics), _cv_auc(cv_seg_metrics, "XGBoost", "esparso"),
           _cv_auc(cv_seg_metrics, "XGBoost", "denso")]
    log = [_auc(logreg_metrics), _cv_auc(cv_seg_metrics, "LogReg", "esparso"),
           _cv_auc(cv_seg_metrics, "LogReg", "denso")]
    llm = [_auc(llm_think_few16, "total"), _auc(llm_think_few16, "esparso"),
           _auc(llm_think_few16, "denso")]
    segs = ["Toda la\ncartera", "Historial\nescaso", "Historial\ndenso"]
    x = np.arange(len(segs))
    w = 0.26

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(x - w, [v or 0 for v in xgb], w, color=C_XGB, label="XGBoost", zorder=3)
    ax.bar(x, [v or 0 for v in log], w, color=C_LOG, label="Reg. Logística", zorder=3)
    ax.bar(x + w, [v or 0 for v in llm], w, color=C_LLM, label="Qwen3-8B (16 ej., razonado)", zorder=3)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    for xi, vals in zip(x, zip(xgb, log, llm)):
        for dx, v in zip([-w, 0, w], vals):
            if v:
                ax.text(xi + dx, v + 0.008, f"{v:.2f}", ha="center", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(segs)
    ax.set_ylabel("AUC en test")
    ax.set_title("XGBoost domina en todos los segmentos;\n"
                 "el modelo de lenguaje no discrimina", fontsize=8, loc="left")
    ax.set_ylim(0.30, 0.92)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=6.5,
              bbox_to_anchor=(0.5, 1.02))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig22 — comparación AUC por segmento (todos los modelos, desde la tabla)
# ---------------------------------------------------------------------------

def plot_comparison_segmentos(comparison_table):
    """Barras de AUC por segmento para cada modelo de la tabla comparativa."""
    from .nodes import MODEL_COLORS, MODEL_NAMES, SEG_ORDER

    table = comparison_table
    if table.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "sin datos", ha="center")
        return fig

    pref = {"test_temporal": 0, "test": 1, "cv_temporal": 2}
    plot_tbl = (table.assign(_p=table["eval_type"].map(pref).fillna(9))
                .sort_values("_p").drop_duplicates(["model", "segment"], keep="first"))
    models = [m for m in MODEL_NAMES if m in set(plot_tbl["model"])]
    x = np.arange(len(SEG_ORDER))
    n = max(len(models), 1)
    width = min(0.20, 0.9 / n)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)

    fig, ax = plt.subplots(figsize=(max(10, n * 2), 6))
    for offset, mk in zip(offsets, models):
        sub = plot_tbl[plot_tbl["model"] == mk].set_index("segment")
        aucs = [sub.loc[s, "AUC"] if s in sub.index else 0 for s in SEG_ORDER]
        stds = [sub.loc[s, "AUC_std"] if s in sub.index else 0 for s in SEG_ORDER]
        stds = [0 if (isinstance(v, float) and np.isnan(v)) else v for v in stds]
        bars = ax.bar(x + offset * width, aucs, width, color=MODEL_COLORS.get(mk, "#888"),
                      alpha=0.85, label=MODEL_NAMES.get(mk, mk), edgecolor="white",
                      yerr=stds, capsize=4, error_kw={"linewidth": 1.2})
        for bar, val in zip(bars, aucs):
            if val and val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Bureau esparso", "Bureau denso", "Total"])
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    ax.set_title("Comparación AUC-ROC por segmento — Tesis UBA 2026\n"
                 "(clásicos: segmentos por CV temporal; LLM: test)", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
