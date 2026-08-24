"""Figuras de EDA y resultados producidas como datasets de Kedro."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textwrap import fill

from credit_risk_frontier import utils

BLUE, RED, GOLD, GREY = "#35618f", "#b44b4b", "#d59a32", "#6f7479"
TU_COLUMNS = list(utils.TU_VARS)
FORM_LABELS = {
    "appusers_age": "Edad del solicitante",
    "credits_dependants_amount": "Personas a cargo",
    "credits_family_expenses": "Gastos familiares mensuales",
    "shops_monthly_incomes": "Ingresos mensuales del negocio",
    "shops_monthly_outcomes": "Egresos mensuales del negocio",
    "shops_daily_incomes": "Ingresos diarios del negocio",
    "shops_initial_capital": "Capital inicial del negocio",
    "shops_rent_amount": "Arriendo mensual del negocio",
    "shops_shop_age": "Antigüedad del negocio",
}


def _official_tu_labels(dictionary: pd.DataFrame) -> dict[str, str]:
    """Devuelve código -> descripción completa de la hoja Variables_CreditVision."""
    return dict(zip(
        dictionary["codigo"].astype(str).str.strip().str.lower(),
        dictionary["definicion_oficial_CreditVision"].astype(str).str.strip(),
    ))


def _experiment_labels(dictionary: pd.DataFrame, width: int = 58) -> dict[str, str]:
    """Etiquetas legibles para las 29 variables del experimento."""
    labels = {
        code: fill(f"{description} ({code.upper()})", width=width)
        for code, description in _official_tu_labels(dictionary).items()
    }
    labels.update(FORM_LABELS)
    return labels


def plot_target_distribution(model_input: pd.DataFrame):
    counts = model_input["target"].value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    bars = ax.bar(["Pago normal\n(target=0)", "Mora >60 días\n(target=1)"], counts,
                  color=[BLUE, RED], width=.58)
    total = counts.sum()
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, value+55,
                f"{int(value):,}\n({100*value/total:.1f} %)".replace(",", "."),
                ha="center", fontsize=9)
    ax.set(title="Distribución del desenlace en la cohorte final",
           ylabel="Cantidad de créditos")
    ax.set_ylim(0, counts.max()*1.18); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2); fig.tight_layout(); return fig


def plot_temporal_change(model_input: pd.DataFrame):
    frame = model_input.copy()
    frame["mes"] = pd.to_datetime(frame["fecha_desembolso"]).dt.to_period("M").dt.to_timestamp()
    monthly = frame.groupby("mes")["target"].agg(["size", "mean"]).reset_index()
    split = frame.groupby("set")["target"].agg(["size", "mean"]).reindex(["train", "val", "test"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), gridspec_kw={"width_ratios": [1.7, 1]})
    ax1.plot(monthly["mes"], 100 * monthly["mean"], color=RED, marker="o", ms=3.5, lw=1.8)
    for _, row in monthly.loc[monthly["size"].lt(100)].iterrows():
        ax1.annotate(
            f"n={int(row['size'])}",
            (row["mes"], 100 * row["mean"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            color=GREY,
        )
    ax1.set(title="La frecuencia de mora cambia a lo largo del período",
            ylabel="Créditos con mora (%)", xlabel="Mes de desembolso")
    ax1.set_ylim(0, 105); ax1.grid(axis="y", alpha=.22)
    labels = ["Entrenamiento", "Validación", "Prueba"]
    bars = ax2.bar(labels, 100 * split["mean"], color=[BLUE, GOLD, RED])
    for bar, rate, n in zip(bars, split["mean"], split["size"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2,
                 f"{100*rate:.1f}%\n(n={int(n):,})".replace(",", "."), ha="center", fontsize=8)
    ax2.set(title="Partición temporal", ylabel="Créditos con mora (%)")
    ax2.set_ylim(0, 105); ax2.tick_params(axis="x", rotation=18)
    for ax in (ax1, ax2): ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_text_fields(summary: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    y = np.arange(len(summary))
    ax1.barh(y, 100 * summary["cobertura"], color=BLUE)
    ax1.set_yticks(y, summary["campo"]); ax1.invert_yaxis(); ax1.set_xlim(0, 105)
    ax1.set(title="Cobertura de los campos textuales", xlabel="Casos con dato (%)")
    for i, v in enumerate(summary["cobertura"]): ax1.text(100*v+1, i, f"{100*v:.1f}%", va="center", fontsize=8)
    ax2.barh(y, summary["mediana_caracteres"], color=GOLD)
    ax2.set_yticks(y, summary["campo"]); ax2.invert_yaxis()
    ax2.set(title="Extensión típica cuando hay dato", xlabel="Mediana de caracteres")
    for i, v in enumerate(summary["mediana_caracteres"]): ax2.text(v+.7, i, f"{v:.0f}", va="center", fontsize=8)
    for ax in (ax1, ax2): ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Campos disponibles en la base; el experimento usa solo la descripción del negocio",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def plot_history_missingness(model_input: pd.DataFrame):
    counts = model_input["n_tu_missing"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    colors = [BLUE if x < 6 else RED for x in counts.index]
    ax.bar(counts.index, counts.values, color=colors, width=.85)
    ax.axvline(5.5, color="black", ls="--", lw=1)
    ax.text(5.65, counts.max()*.96, "Historial esparso: 6 o más", va="top", fontsize=9)
    ax.set(title="Cantidad de atributos del buró sin información",
           xlabel="Atributos sin información (sobre 20)", ylabel="Cantidad de créditos")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    return fig


def plot_feature_missingness(
    model_input: pd.DataFrame,
    tu_dictionary: pd.DataFrame,
):
    """Compara la falta de información de las 29 variables estructuradas."""
    official = _official_tu_labels(tu_dictionary)
    rows = []
    for column in utils.TU_VARS:
        values = pd.to_numeric(model_input[column], errors="coerce")
        rows.append({
            "variable": column,
            "label": f"{column.upper()} — {fill(official[column], width=48)}",
            "missing": float((values.isna() | values.lt(0)).mean()),
            "source": "TransUnion",
        })
    for column in utils.FORM_DIRECT_VARS:
        values = pd.to_numeric(model_input[column], errors="coerce")
        rows.append({
            "variable": column,
            "label": FORM_LABELS[column],
            "missing": float(values.isna().mean()),
            "source": "Formulario",
        })
    data = pd.DataFrame(rows).sort_values("missing")
    colors = data["source"].map({"TransUnion": BLUE, "Formulario": GOLD})
    fig, ax = plt.subplots(figsize=(11.8, 10.8))
    bars = ax.barh(data["label"], 100 * data["missing"], color=colors)
    for bar, value in zip(bars, data["missing"]):
        ax.text(
            100 * value + .6,
            bar.get_y() + bar.get_height() / 2,
            f"{100 * value:.1f}%",
            va="center",
            fontsize=7.5,
        )
    ax.set(
        title="Falta de información en las 29 variables estructuradas",
        xlabel="Casos sin información (%)",
        ylabel="",
    )
    ax.set_xlim(0, min(105, max(12, 100 * data["missing"].max() + 9)))
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.18)
    fig.tight_layout()
    return fig


def plot_segment_rates(model_input: pd.DataFrame):
    """Muestra tamaño y mora de historial esparso/denso por partición."""
    order = [
        ("train", "esparso"), ("train", "denso"),
        ("val", "esparso"), ("val", "denso"),
        ("test", "esparso"), ("test", "denso"),
    ]
    stats = (
        model_input.groupby(["set", "segmento"])["target"]
        .agg(["size", "mean"])
    )
    rows = [stats.loc[key] for key in order]
    rates = np.array([row["mean"] for row in rows], dtype=float)
    sizes = np.array([row["size"] for row in rows], dtype=int)
    labels = [
        "Entrenamiento\nesparso", "Entrenamiento\ndenso",
        "Validación\nesparso", "Validación\ndenso",
        "Prueba\nesparso", "Prueba\ndenso",
    ]
    colors = [RED, BLUE, RED, BLUE, RED, BLUE]
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    bars = ax.bar(labels, 100 * rates, color=colors, width=.7)
    for bar, rate, n in zip(bars, rates, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.6,
            f"{100 * rate:.1f}%\n(n={n:,})".replace(",", "."),
            ha="center",
            fontsize=8,
        )
    ax.set(
        title="La frecuencia de mora cambia por período y densidad del historial",
        ylabel="Créditos con mora (%)",
        xlabel="Partición temporal y disponibilidad del buró",
    )
    ax.set_ylim(0, 108)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    return fig


def plot_subcategory_rates(model_input: pd.DataFrame):
    train = model_input[model_input["set"].eq("train")].copy()
    train["subcategoria"] = train["subcategoria_texto"].fillna("").astype(str).str.strip().str.lower()
    rates = (train[train["subcategoria"].ne("")].groupby("subcategoria")["target"]
             .agg(["size", "mean"]).query("size >= 40").sort_values("mean"))
    display = {
        "esteticas spa": "Estéticas y spa",
        "comidas rapidas": "Comidas rápidas",
        "peluqueria y manicuria": "Peluquería y manicuría",
        "venta de productos para el hogar": "Venta de productos para el hogar",
        "arreglos": "Arreglos",
        "tienda abarrotes": "Tienda de abarrotes",
        "construccion": "Construcción",
        "viveros": "Viveros",
        "reparacion electrodomesticos": "Reparación de electrodomésticos",
        "confeccion y comercializacion": "Confección y comercialización",
        "venta productos cuidado personal belleza": "Productos de cuidado personal y belleza",
        "venta de accesorios bolsos y o bisuteria": "Accesorios, bolsos y bisutería",
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    labels = [display.get(value, value.capitalize()) for value in rates.index]
    bars = ax.barh(labels, 100*rates["mean"], color=BLUE)
    overall = 100*train["target"].mean()
    ax.axvline(overall, color=RED, ls="--", lw=1.2, label=f"Promedio entrenamiento ({overall:.1f} %)")
    for bar, n in zip(bars, rates["size"]):
        ax.text(bar.get_width()+.6, bar.get_y()+bar.get_height()/2, f"n={int(n)}", va="center", fontsize=7)
    ax.set(title="La frecuencia de mora varía entre subcategorías del negocio",
           xlabel="Créditos con mora en entrenamiento (%)", ylabel="")
    ax.set_xlim(0, min(100, 100*rates["mean"].max()+12)); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.18)
    fig.tight_layout(); return fig


def _bureau_groups(series: pd.Series, name: str) -> pd.Series:
    out = pd.Series("Sin información", index=series.index, dtype="object")
    valid = series.ge(0) & series.notna()
    if name == "g051s":
        out.loc[valid] = pd.cut(series.loc[valid], [-.001, 0, 20, 40, 60, 80, 100],
                                labels=["0", "1–20", "21–40", "41–60", "61–80", "81–100"],
                                include_lowest=True).astype(str)
    else:
        ranked = pd.qcut(series.loc[valid].rank(method="first"), 5,
                         labels=["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (mayor)"])
        out.loc[valid] = ranked.astype(str)
    return out


def plot_bureau_bivariate(model_input: pd.DataFrame, tu_dictionary: pd.DataFrame):
    train = model_input[model_input["set"].eq("train")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    official = _official_tu_labels(tu_dictionary)
    orders = {
        "g051s": ["Sin información", "0", "1–20", "21–40", "41–60", "61–80", "81–100"],
        "wd81": ["Sin información", "Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (mayor)"],
    }
    titles = {
        code: fill(f"{official[code]} ({code.upper()})", width=48)
        for code in ("g051s", "wd81")
    }
    for ax, col in zip(axes, ("g051s", "wd81")):
        groups = _bureau_groups(train[col], col)
        stats = train.assign(grupo=groups).groupby("grupo", observed=False)["target"].agg(["size", "mean"])
        stats = stats.reindex([x for x in orders[col] if x in stats.index]).dropna(subset=["mean"])
        bars = ax.bar(range(len(stats)), 100*stats["mean"], color=GOLD)
        ax.axhline(100*train.target.mean(), color=RED, ls="--", lw=1)
        ax.set_xticks(range(len(stats)), stats.index, rotation=28, ha="right")
        ax.set(title=titles[col], ylabel="Créditos con mora (%)", xlabel="Grupos del valor observado")
        ax.title.set_fontsize(11.5)
        for bar, n in zip(bars, stats["size"]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"n={int(n)}", ha="center", fontsize=6.5)
        ax.set_ylim(0, 105); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.18)
    fig.suptitle("Asociaciones descriptivas de dos atributos del buró en entrenamiento", fontsize=11)
    fig.tight_layout(); return fig


def plot_training_associations(model_input: pd.DataFrame, tu_dictionary: pd.DataFrame):
    train = model_input[model_input["set"].eq("train")]
    allowed = utils.TU_VARS + utils.FORM_DIRECT_VARS
    numeric = train[allowed].apply(pd.to_numeric, errors="coerce")
    # Los códigos negativos del buró representan ausencia, no magnitudes negativas.
    for col in set(TU_COLUMNS).intersection(numeric.columns):
        numeric.loc[numeric[col].lt(0), col] = np.nan
    informative = [column for column in numeric if numeric[column].nunique(dropna=True) > 1]
    corr = numeric[informative].corrwith(train["target"]).dropna().sort_values(key=abs).tail(12)
    labels = dict(FORM_LABELS)
    labels.update({
        code: fill(f"{description} ({code.upper()})", width=58)
        for code, description in _official_tu_labels(tu_dictionary).items()
    })
    fig, ax = plt.subplots(figsize=(11.8, 8.5))
    colors = [RED if v > 0 else BLUE for v in corr]
    ax.barh([labels.get(x, x) for x in corr.index], corr.values, color=colors)
    ax.axvline(0, color="black", lw=.8)
    ax.set(title="Variables del experimento con mayor asociación individual en entrenamiento",
           xlabel="Correlación con mora (target=1)")
    ax.tick_params(axis="y", labelsize=12.5, length=0)
    ax.tick_params(axis="x", labelsize=11)
    ax.title.set_fontsize(15)
    ax.xaxis.label.set_fontsize(12)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.18)
    fig.tight_layout(); return fig


def plot_financial_outliers(model_input: pd.DataFrame):
    train = model_input[model_input["set"].eq("train")]
    columns = [("shops_monthly_incomes", "Ingresos mensuales del negocio"),
               ("credits_family_expenses", "Gastos familiares mensuales")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for ax, (col, label) in zip(axes, columns):
        values = pd.to_numeric(train[col], errors="coerce")
        positive = values[values.gt(0)]
        q1, q3 = positive.quantile([.25, .75]); threshold = q3 + 1.5*(q3-q1)
        for target, color, name in [(0, BLUE, "Pago normal"), (1, RED, "Mora")]:
            subset = values[(train.target.eq(target)) & values.gt(0)]
            ax.hist(np.log10(subset), bins=28, density=True, alpha=.48, color=color, label=name)
        ax.axvline(np.log10(threshold), color=GOLD, ls="--", lw=1.3)
        ax.set(title=label, xlabel="Valor declarado (escala logarítmica, COP)", ylabel="Densidad")
        ax.text(.98, .95, f"{int((positive>threshold).sum())} casos sobre Q3 + 1,5×RIC",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Los montos declarados contienen valores extremos", fontsize=11)
    fig.tight_layout(); return fig


def plot_logreg_coefficients(
    coefficients: pd.DataFrame,
    tu_dictionary: pd.DataFrame,
):
    """Presenta los coeficientes de la regresión sobre variables estandarizadas."""
    data = coefficients.sort_values("coeficiente_estandarizado").copy()
    feature_labels = _experiment_labels(tu_dictionary)
    labels = [feature_labels[code] for code in data["codigo"]]
    colors = [RED if value > 0 else BLUE for value in data["coeficiente_estandarizado"]]
    fig, ax = plt.subplots(figsize=(11.8, 13.5))
    ax.barh(labels, data["coeficiente_estandarizado"], color=colors)
    ax.axvline(0, color="black", lw=.8)
    ax.set(
        title="Coeficientes de la Regresión Logística con 29 variables estructuradas",
        xlabel="Cambio en el logaritmo de las chances de mora por un desvío estándar",
        ylabel="",
    )
    ax.tick_params(axis="y", labelsize=7.8, length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.18)
    fig.tight_layout()
    return fig


def plot_xgb_shap_summary(
    shap_summary: pd.DataFrame,
    tu_dictionary: pd.DataFrame,
):
    """Muestra qué variables más movieron las predicciones de XGBoost en prueba."""
    data = (
        shap_summary.sort_values("shap_medio_absoluto", ascending=False)
        .head(15)
        .sort_values("shap_medio_absoluto")
        .copy()
    )
    feature_labels = _experiment_labels(tu_dictionary)
    labels = [feature_labels[code] for code in data["codigo"]]
    fig, ax = plt.subplots(figsize=(11.8, 9.2))
    ax.barh(labels, data["shap_medio_absoluto"], color=BLUE)
    ax.set(
        title="Variables con mayor contribución en XGBoost sobre la prueba",
        xlabel=(
            "Promedio del valor SHAP absoluto "
            "(contribución al logaritmo de las chances)"
        ),
        ylabel="",
    )
    ax.tick_params(axis="y", labelsize=8.3, length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.18)
    fig.tight_layout()
    return fig


def plot_roc_classics(predictions: pd.DataFrame):
    """Curvas ROC de las dos configuraciones clásicas en prueba."""
    from sklearn.metrics import roc_auc_score, roc_curve

    specs = [
        ("logreg_tu_form", "Regresión Logística · 29 variables", GOLD),
        ("xgb_tu_form", "XGBoost · 29 variables", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    for model, label, color in specs:
        data = predictions[
            predictions["model"].eq(model) & predictions["probability"].notna()
        ]
        y_true = data["y_true"].to_numpy(dtype=int)
        probability = data["probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, probability)
        auc = roc_auc_score(y_true, probability)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="Azar")
    ax.set(
        title="Curvas ROC de los métodos clásicos",
        xlabel="Tasa de falsos positivos",
        ylabel="Tasa de verdaderos positivos",
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.16)
    fig.tight_layout()
    return fig


def plot_roc_qwen(predictions: pd.DataFrame):
    """Curvas ROC de Qwen separadas por modalidad de ejemplos."""
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = {
        "tu_form": "TransUnion + formulario",
        "tu_form_description": "TransUnion + formulario + descripción",
    }
    colors = {"tu_form": GOLD, "tu_form_description": RED}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharex=True, sharey=True)
    for ax, mode in zip(axes, ("zero", "few")):
        for profile in ("tu_form", "tu_form_description"):
            data = predictions[
                predictions["model"].eq("qwen3:8b")
                & predictions["mode"].eq(mode)
                & predictions["feature_profile"].eq(profile)
                & predictions["probability"].notna()
            ]
            if data.empty:
                continue
            y_true = data["y_true"].to_numpy(dtype=int)
            probability = data["probability"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y_true, probability)
            auc = roc_auc_score(y_true, probability)
            ax.plot(
                fpr, tpr, color=colors[profile], lw=2,
                label=f"{labels[profile]} (AUC={auc:.3f}; n={len(data)})",
            )
        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)
        ax.set_title("Sin ejemplos" if mode == "zero" else "Con ocho ejemplos")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, loc="lower right", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.16)
    axes[0].set_ylabel("Tasa de verdaderos positivos")
    fig.suptitle("Curvas ROC de Qwen según información y ejemplos", fontsize=11)
    fig.tight_layout()
    return fig


def _label(row):
    mode = {"trained": "", "zero": " · sin ejemplos", "few": " · 8 ejemplos"}[row["mode"]]
    profile = {
        "tu": "TransUnion",
        "tu_form": "TransUnion + formulario",
        "tu_form_description": "TransUnion + formulario + descripción",
    }[row["feature_profile"]]
    return f"{row['modelo_presentado']}{mode} · {profile}"


def plot_auc(results: pd.DataFrame):
    data = results.sort_values("AUC").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    colors = [RED if m == "Qwen3-8B" else (GOLD if m == "Regresión logística" else BLUE)
              for m in data["modelo_presentado"]]
    bars = ax.barh(range(len(data)), data["AUC"], color=colors)
    ax.errorbar(data["AUC"], range(len(data)),
                xerr=[data["AUC"]-data["AUC_ci_low"], data["AUC_ci_high"]-data["AUC"]],
                fmt="none", ecolor="black", capsize=2.5, lw=.8)
    ax.set_yticks(range(len(data)), [_label(r) for _, r in data.iterrows()])
    ax.axvline(.5, color="black", ls="--", lw=1); ax.set_xlim(.45, .86)
    ax.set(title="Capacidad de ordenar el riesgo en el período de prueba", xlabel="AUC (IC 95 %)")
    for bar, v in zip(bars, data["AUC"]): ax.text(v+.006, bar.get_y()+bar.get_height()/2, f"{v:.3f}".replace(".", ","), va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    return fig


def plot_text_delta(paired: pd.DataFrame):
    wanted = ["qwen3:8b_zero", "qwen3:8b_few"]
    data = paired[(paired["segment"].eq("total")) & paired["comparison"].isin(wanted)].copy()
    order = {k: i for i, k in enumerate(wanted)}; data["_order"] = data["comparison"].map(order); data = data.sort_values("_order")
    labels = {"qwen3:8b_zero": "Qwen3-8B · sin ejemplos",
              "qwen3:8b_few": "Qwen3-8B · 8 ejemplos"}
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.errorbar(data["delta_auc"], y,
                xerr=[data["delta_auc"]-data["ci_low"], data["ci_high"]-data["delta_auc"]],
                fmt="o", color=BLUE, ecolor=GREY, capsize=4, ms=6)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks(y, [labels[x] for x in data["comparison"]]); ax.invert_yaxis()
    ax.set(title="Cambio de AUC de Qwen al agregar la descripción del negocio",
           xlabel="ΔAUC = AUC con descripción − AUC sin descripción (IC 95 % pareado)")
    ax.set_xlim(-.09, .09); ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
