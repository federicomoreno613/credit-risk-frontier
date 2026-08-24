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

Pendiente implementación y validación de desempeño.
