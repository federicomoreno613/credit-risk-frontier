"""Modelo #3 del PLAN §2.2 — transformer tabular preentrenado (TabPFN v2).

TabPFN v2 (Hollmann et al., 2025) es una prior-data fitted network: no se
entrena sobre nuestros datos, usa el train como CONTEXTO in-context y predice
el test en un pase. Sin hiperparámetros que buscar, mismo split del contrato.
Maneja NaN nativamente, así que preparar_numerico alcanza (los códigos TU
negativos ya quedan como ausencia).

Contexto: si train excede --max-contexto filas (límite de preentrenamiento del
modelo, 10.000 por defecto), se muestrea estratificado por target con C.SEED
y se documenta en la salida. Con la cohorte actual (~2.9k filas) entra entero.

TabFM (Google Research) queda como flag opcional --tabfm: requiere clonar
github.com/google-research/tabfm e instalar sus pesos (licencia NO comercial),
no se descarga acá. Instrucciones: clonar el repo, `pip install -e .`, bajar
los pesos según su README y re-correr con --tabfm.

Uso:
  poetry run python pipeline/04_tabulares/predecir.py
  poetry run python pipeline/04_tabulares/predecir.py --device cpu

Salidas:
  data/pipeline/predicciones/tabulares.parquet  (configuracion="tabpfn")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402
import monitoreo  # noqa: E402


def elegir_device(pedido: str) -> str:
    if pedido != "auto":
        return pedido
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def muestrear_contexto(train: pd.DataFrame, maximo: int) -> pd.DataFrame:
    """Muestreo estratificado por target con C.SEED si el train excede el límite."""
    if len(train) <= maximo:
        return train
    fraccion = maximo / len(train)
    muestra = (
        train.groupby("target", group_keys=False)
        .apply(lambda g: g.sample(frac=fraccion, random_state=C.SEED))
    )
    print(
        f"contexto muestreado: {len(muestra)}/{len(train)} filas "
        f"(estratificado por target, seed={C.SEED})"
    )
    return muestra


def predecir_tabpfn(train: pd.DataFrame, test: pd.DataFrame, device: str):
    from tabpfn import TabPFNClassifier

    X_train, y_train = C.preparar_numerico(train), train["target"].astype(int)
    X_test = C.preparar_numerico(test)
    modelo = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
    modelo.fit(X_train, y_train)  # "fit" = guardar el contexto in-context
    return modelo.predict_proba(X_test)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda"),
        help="dispositivo de inferencia (default: mps si está disponible, si no cpu)",
    )
    parser.add_argument(
        "--max-contexto", type=int, default=10_000,
        help="máximo de filas de train como contexto in-context (default 10000)",
    )
    parser.add_argument(
        "--tabfm", action="store_true",
        help="usar TabFM en vez de TabPFN (requiere instalación manual, ver docstring)",
    )
    args = parser.parse_args()

    if args.tabfm:
        raise SystemExit(
            "TabFM no está instalado en este entorno (licencia no comercial, pesos "
            "manuales). Clonar github.com/google-research/tabfm, instalarlo con "
            "`pip install -e .`, bajar los pesos según su README y adaptar "
            "predecir_tabpfn() como predecir_tabfm()."
        )

    device = elegir_device(args.device)
    print(f"TabPFN v2 en device={device}")

    variables = C.cargar_variables()
    train, _val, test = C.dividir(variables)  # sin hiperparámetros: val no se usa
    contexto = muestrear_contexto(train, args.max_contexto)
    probabilidades = predecir_tabpfn(contexto, test, device)

    y_test = test["target"].astype(int)
    predicciones = pd.DataFrame({
        "modelo": "tabpfn",
        "configuracion": "tabpfn",
        "credito_id_anon": test["credito_id_anon"].astype(str),
        "segmento": test["segmento"].astype(str),
        "y_true": y_test,
        "probabilidad": probabilidades,
        "valida": True,
    })
    path = C.guardar_predicciones("tabulares", predicciones)
    print(f"tabpfn AUC test: {roc_auc_score(y_test, probabilidades):.4f} "
          f"(contexto={len(contexto)} filas)")
    monitoreo.reporte_modelo(predicciones, "TabPFN v2 (test)")
    print(f"predicciones -> {path}")


if __name__ == "__main__":
    main()
