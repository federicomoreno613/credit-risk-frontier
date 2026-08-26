"""Contrato local del experimento LendingClub (dataset 2 del PLAN §2.5).

Fuente: subset publicado por Feng et al. (2023, CALM/FinBen) en Hugging Face
(`TheFinAI/cra-lendingclub`) — el mismo benchmark contra el que se compara,
sin Kaggle ni preprocesamiento propio. Cada fila trae:
  query  : prompt exacto del benchmark (good/bad)
  answer : etiqueta textual ('good' = pagó, 'bad' = mora)
  gold   : 0 = good, 1 = bad
  text   : registro serializado ("The state of <Atributo> is <valor>. ...")

Alcance decidido (2026-08): SOLO Qwen3-8B zero/few-shot sobre este subset,
precedido por tests de contaminación de preentrenamiento (Bordt et al. 2024).
No se re-entrenan modelos tabulares en LendingClub.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline"))
import contrato as C  # noqa: E402

HF_REPO = "TheFinAI/cra-lendingclub"
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{split}.parquet"
SPLITS = ("train", "valid", "test")

DATA_DIR = ROOT / "data" / "lendingclub"
MANIFIESTO = DATA_DIR / "manifiesto.json"
SALIDAS = DATA_DIR / "salidas"

# Los 21 atributos del subset de Feng, en el orden canónico del texto.
ATRIBUTOS = [
    "Installment", "Loan Purpose", "Loan Application Type", "Interest Rate",
    "Last Payment Amount", "Loan Amount", "Revolving Balance",
    "Delinquency In 2 years", "Inquiries In 6 Months", "Mortgage Accounts",
    "Grade", "Open Accounts", "Revolving Utilization Rate", "Total Accounts",
    "Fico Range Low", "Fico Range High", "Address State", "Employment Length",
    "Home Ownership", "Verification Status", "Annual Income",
]

_RE_CAMPO = re.compile(r"The state of (.+?) is (.+?)\.(?= The state of |$)")


def parquet_path(split: str) -> Path:
    return DATA_DIR / f"{split}.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def parsear_texto(texto: str) -> dict[str, str]:
    """Extrae los 21 pares atributo→valor del `text` serializado de Feng."""
    campos = dict(_RE_CAMPO.findall(texto))
    faltan = [a for a in ATRIBUTOS if a not in campos]
    if faltan:
        raise ValueError(f"no pude parsear {faltan} en: {texto[:120]}…")
    return campos


def serializar_feng(campos: dict[str, str]) -> str:
    """Reconstruye el `text` en el template exacto del benchmark."""
    cuerpo = " ".join(f"The state of {a} is {campos[a]}." for a in ATRIBUTOS)
    return f"The client has attributes as follows: {cuerpo}"


def chat(mensajes: list[dict], *, temperatura: float = 0.0,
         num_predict: int = 512, timeout: int = 600) -> str:
    """Llamada mínima a Qwen local vía Ollama, sin thinking, determinística."""
    respuesta = requests.post(
        C.utils.DEFAULT_OLLAMA_URL,
        json={
            "model": C.QWEN_MODEL,
            "messages": mensajes,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperatura,
                "seed": C.SEED,
                "num_ctx": 8192,
                "num_predict": num_predict,
            },
        },
        timeout=timeout,
    )
    respuesta.raise_for_status()
    return respuesta.json()["message"]["content"]


def guardar_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=C.utils.json_safe))
    return path
