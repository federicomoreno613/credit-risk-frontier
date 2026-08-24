# 07_finetuning: QLoRA Instruction Tuning

Modelo #8 del PLAN §2.2.2: fine-tuning eficiente (QLoRA, Dettmers et al. 2023) de Qwen 2.5 con Unsloth en Apple Silicon.

## Datos de Entrenamiento

- Fuente: `data/pipeline/03_humanizado.parquet` (filas con `set='train'` únicamente)
- Formato: instruction tuning, respuesta = `PROBABILIDAD_DE_MORA` (variable continua)
- Template: `{"instruction": "...", "response": "<prob>"}`
- Volumen: ~3.000 muestras train (70% de 4.201 filas)

## Plan de Integración

```python
import contrato as C

# Datos humanizados train
df_train = C.cargar_variables()
df_train = df_train[df_train['set'] == 'train']

# Preparar dataset instruction tuning
# (con instrucciones de C.serializar_perfil o análogas)

# Fine-tune con Unsloth
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(...)
# ... QLoRA config, SFT trainer ...

# Inferencia: caché JSONL de Qwen reanudable
# (nueva config="qwen-7b-qlora" en C.PROMPT_VARIANT)
```

## Referencias

- Dettmers et al. (2023): QLoRA – Efficient Finetuning of Quantized LLMs
- Unsloth: https://github.com/unslothai/unsloth
- Esbozo previo: `nbs/08_sft_qlora.py`
- Compatible con Apple Silicon (Metal aceleración)
- Datos: `data/pipeline/03_humanizado.parquet`

## Estado

Pendiente implementación, validación y métricas en caché JSONL.
