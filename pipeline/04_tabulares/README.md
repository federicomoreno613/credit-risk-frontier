# 04_tabulares: Modelos Tabulares Preentrenados

Modelo #3 del PLAN §2.2: transformers tabulares preentrenados para inferencia directa sobre variables numéricas.

## Candidatos

- **TabPFN v2** (Hollmann et al. 2025): prior functions network, sin hiperparámetros, ~100ms/predicción.
- **TabFM** (Google Research): in-context learning, API scikit-learn, ~100 filas de contexto por defecto. Licencia: **no comercial** (citar `github.com/google-research/tabfm`).

## Plan de Integración

```python
import contrato as C

# Cargar y dividir
variables = C.cargar_variables()
train, val, test = C.dividir(variables)

# Preparar numéricas
X_train = C.preparar_numerico(train[C.FEATURES_29])
y_train = train['target']
# (id. para val, test)

# Entrenar/inferir con TabPFN o TabFM
# ... modelo específico aquí ...

# Guardar predicciones
df_predicciones = ...  # [modelo, configuracion, credito_id_anon, segmento, y_true, probabilidad, valida]
C.guardar_predicciones("tabulares", df_predicciones)
```

## Referencias

- Esbozo previo: `nbs/06_tabpfn.py`
- TabPFN: https://github.com/PriorLabs/TabPFN
- TabFM: https://github.com/google-research/tabfm (licencia no comercial)
- Datos: `data/pipeline/02_variables.parquet`, `03_humanizado.parquet`

## Estado

Implementado en `predecir.py` (TabPFN v2, device MPS/CPU, corrida 2026-08-24):
AUC test 0,7656 (denso 0,7707 / esparso 0,7299) con las 2.940 filas de train
como contexto in-context, sin entrenamiento ni hiperparámetros.

Nota de dependencia: se fija `tabpfn = ^2.2` (release abierto de TabPFN v2,
pesos públicos en HuggingFace). Las versiones 8.x de la librería exigen
aceptar licencia con cuenta PriorLabs (`TABPFN_TOKEN`); si se quiere migrar,
obtener el token en https://ux.priorlabs.ai/account. TabFM sigue pendiente
(flag `--tabfm`, licencia no comercial, instalación manual).
