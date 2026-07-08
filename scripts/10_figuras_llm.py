#!/usr/bin/env python3
# ============================================================================
# Genera las Figuras 3 y 4 de E4 a partir de los JSON del modo de razonamiento.
# Correr DESPUÉS de correr_thinking.sh, desde credit-risk-frontier/:
#     python scripts/10_figuras_llm.py
# Lee models/llm_metrics_thinking_*.json y escribe figures/fig3_*.png, fig4_*.png
# ============================================================================
import json, glob
from pathlib import Path
import numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
FIGS = BASE / "figures"; FIGS.mkdir(exist_ok=True)

C_XGB, C_LOG, C_LLM, GREY = "#55A868", "#4C72B0", "#C44E52", "#8C8C8C"

# --- cifras definitivas de los modelos clásicos (split temporal, test) ------
XGB_TOTAL, LOG_TOTAL = 0.7776, 0.6910
XGB_ESP, XGB_DEN = 0.704, 0.739          # cv temporal por segmento
LOG_ESP, LOG_DEN = 0.711, 0.683

def load(mode, shots=None, scope="test"):
    """Carga un JSON de razonamiento; devuelve None si aún no existe."""
    name = f"llm_metrics_thinking_{'zero' if mode=='zero' else f'few{shots}'}_{scope}.json"
    p = MODELS / name
    if not p.exists(): return None
    return json.loads(p.read_text())

def auc_of(d, seg="total"):
    if d is None: return None
    t = d.get("test", {})
    return t.get(seg, {}).get("auc") if isinstance(t.get(seg), dict) else t.get(seg)

def style():
    mpl.rcParams.update({"font.size":8,"axes.titlesize":8,"axes.labelsize":8,
        "xtick.labelsize":6,"ytick.labelsize":6.5,"legend.fontsize":6.5,
        "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":300})

# ============================================================================
# FIG 3 — AUC del LLM según nº de ejemplos, por segmento
# ============================================================================
def fig3():
    style()
    configs = [("0",load("zero")), ("8",load("few",8)), ("16",load("few",16)), ("32",load("few",32))]
    xs = [c[0] for c in configs]
    tot = [auc_of(c[1],"total") for c in configs]
    esp = [auc_of(c[1],"esparso") for c in configs]
    den = [auc_of(c[1],"denso") for c in configs]
    fig, ax = plt.subplots(figsize=(5.6,3.6))
    for ys,lab,col in [(tot,"Toda la cartera",GREY),(esp,"Historial escaso",C_LLM),(den,"Historial denso",C_LOG)]:
        pts=[(x,y) for x,y in zip(xs,ys) if y is not None]
        if pts: ax.plot([p[0] for p in pts],[p[1] for p in pts],"-o",color=col,label=lab,lw=1.6,ms=5)
    ax.axhline(0.5,color=GREY,ls=":",lw=1); ax.text(3.05,0.505,"azar",fontsize=6,color=GREY,va="bottom")
    ax.axhline(XGB_TOTAL,color=C_XGB,ls="--",lw=1.2); ax.text(3.05,XGB_TOTAL,"XGBoost",fontsize=6,color=C_XGB,va="center")
    ax.set_xlabel("Número de ejemplos en contexto"); ax.set_ylabel("AUC en test")
    ax.set_title("Los ejemplos no rescatan al modelo de lenguaje:\nsigue cerca del azar en todos los segmentos",fontsize=8,loc="left")
    ax.set_ylim(0.45,0.82); ax.legend(frameon=False,loc="upper left")
    fig.tight_layout(); out=FIGS/"fig3_auc_ejemplos.png"; fig.savefig(out,bbox_inches="tight")
    print("Fig3:",out); return out

# ============================================================================
# FIG 4 — comparación por modelo y segmento (barras agrupadas)
# ============================================================================
def fig4():
    style()
    llm16 = load("few",16)
    llm_tot, llm_esp, llm_den = auc_of(llm16,"total"), auc_of(llm16,"esparso"), auc_of(llm16,"denso")
    segs=["Toda la\ncartera","Historial\nescaso","Historial\ndenso"]
    xgb=[XGB_TOTAL,XGB_ESP,XGB_DEN]; log=[LOG_TOTAL,LOG_ESP,LOG_DEN]; llm=[llm_tot,llm_esp,llm_den]
    x=np.arange(len(segs)); w=0.26
    fig,ax=plt.subplots(figsize=(6.0,3.6))
    ax.bar(x-w,xgb,w,color=C_XGB,label="XGBoost",zorder=3)
    ax.bar(x,log,w,color=C_LOG,label="Reg. Logística",zorder=3)
    llm_plot=[v if v is not None else 0 for v in llm]
    ax.bar(x+w,llm_plot,w,color=C_LLM,label="Qwen3-8B (16 ej., razonado)",zorder=3)
    ax.axhline(0.5,color=GREY,ls=":",lw=1)
    for xi,v in zip(x+w,llm):
        if v is None: ax.text(xi,0.51,"pend.",ha="center",fontsize=5.5,color=GREY,rotation=90)
    for xi,vals in zip(x,zip(xgb,log,llm)):
        for dx,v in zip([-w,0,w],vals):
            if v: ax.text(xi+dx,v+0.008,f"{v:.2f}",ha="center",fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(segs); ax.set_ylabel("AUC en test")
    ax.set_title("XGBoost domina en todos los segmentos;\nel modelo de lenguaje no discrimina",fontsize=8,loc="left")
    ax.set_ylim(0.45,0.82); ax.legend(frameon=False,loc="upper right",ncol=1)
    fig.tight_layout(); out=FIGS/"fig4_comparacion_modelos.png"; fig.savefig(out,bbox_inches="tight")
    print("Fig4:",out); return out

if __name__=="__main__":
    have=sorted(p.name for p in MODELS.glob("llm_metrics_thinking_*.json"))
    print("JSON de razonamiento encontrados:", have if have else "NINGUNO todavía")
    fig3(); fig4()
    print("\nListo. Las barras/puntos 'pend.' se completan cuando estén todos los JSON.")
