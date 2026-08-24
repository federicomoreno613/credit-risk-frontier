# 08_lendingclub: Benchmark LendingClub

Segundo dataset del PLAN §2.3: replicación del flujo completo (preprocesamiento → humanización → modelos → monitoreo) sobre LendingClub (Kaggle, público).

## Dataset

- Fuente: Kaggle LendingClub + Feng et al. (2023)
- Ubicación: `data/lendingclub/`
- Target: default/no-default; variables: préstamo, FICO, ingresos, etc.
- Preprocesamiento: mismo pipeline que datos Feng (split temporal, normalización)

## Punto Crítico: Contaminación de Preentrenamiento

LendingClub es **dataset público** y probable que Qwen/GPT lo hayan visto durante preentrenamiento. Implicaciones:
- Qwen/GPT pueden memorizar patrones → inflación artificial de desempeño
- Necesario documentar y, de ser posible, medir:
  - Comparar desempeño zero-shot relativo: Feng vs. LendingClub
  - Tests de memorización (p. ej. paráfrasis de variables)
  - Mencionar explícitamente el riesgo en conclusiones

## Plan

1. **Preprocesamiento** (`01_prep.py`): subset Feng et al., split temporal, validación
2. **Humanización** (`02_humanizar.py`): serializar perfiles igual que Feng
3. **Modelos** (`03_*.py`): TabPFN, Qwen, GPT, etc. con mismas configs
4. **Monitoreo** (`04_reporte.py`): métricas, comparativo vs. Feng

## Referencias

- Esbozos: `nbs/10_lendingclub_prep.py`, `nbs/11_lendingclub_benchmark.py`
- Feng et al. (2023): baseline comparativo
- Datos: `data/lendingclub/`
- Contrato: C.cargar_variables(), C.dividir(), C.guardar_predicciones()

## Estado

Pendiente: preprocesamiento, humanización, análisis de contaminación y comparativo.
