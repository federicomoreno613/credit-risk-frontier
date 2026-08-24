# Qwen local — modelos #5-7 del PLAN (zero-shot, thinking, few-shot)

Inferencia local con Ollama (`qwen3:8b`) sobre el test del contrato. El script
`predecir.py` está comentado de forma didáctica; este README resume el flujo.

## Etapas

1. **Texto humanizado** (viene del preprocesamiento): cada crédito ya es una
   oración con las 29 variables en palabras; `tu_form_description` agrega la
   descripción libre del negocio. Qwen y GPT leen el mismo texto.
2. **Prompt**: system corto + "DATOS DEL SOLICITANTE: ..." + instrucción de
   cerrar con `PROBABILIDAD_DE_MORA: <0-100>`. Ese entero /100 es el score.
3. **Few-shot** (`--shots 8|16`): ejemplos reales SOLO de train, elegidos por
   KNN balanceado (mitad mora, mitad pago) en el espacio de las 29 variables.
4. **Thinking**: el razonamiento nativo completo se guarda por caso — es el
   insumo de la comparación explicaciones vs. SHAP (PLAN §2.4).
5. **Cache reanudable**: un JSONL por configuración; se corta y retoma solo.

## Ver el funcionamiento (educativo)

```bash
poetry run python pipeline/05_qwen/predecir.py --demo
# imprime: prompt completo -> thinking -> respuesta -> probabilidad parseada
```

## Correr

```bash
poetry run python pipeline/05_qwen/predecir.py --perfil tu_form --shots 0   # una config
poetry run python pipeline/05_qwen/predecir.py                              # las 6
poetry run python pipeline/monitoreo.py                                     # consolidar y medir
```

## Registro JSONL (uno por caso)

```json
{"evaluation_id": "4a6401ec067b", "modelo": "qwen3:8b", "perfil": "tu_form",
 "shots": 0, "set": "test", "segmento": "denso", "y_true": 0,
 "prompt_variant": "minimum", "probabilidad": 0.65, "valida": true,
 "thinking": "Okay, let's tackle this credit risk assessment...",
 "respuesta": "PROBABILIDAD_DE_MORA: 65", "eval_count": 812}
```

Si cambia el contrato (variables, prompt, split): borrar los JSONL de
`data/pipeline/razonamientos/` antes de volver a correr. Contienen datos de
casos: NO se publican.
