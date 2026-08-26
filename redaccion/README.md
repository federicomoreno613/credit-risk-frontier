# Redacción de la tesis — método de trabajo humano + IA

Base para escribir la tesis final por unidades, manteniendo la voz de Federico.
La IA NO redacta el texto final: prepara evidencia, estructura y crítica.

## Regla de oro (humanidad del texto)

1. **Federico escribe la primera prosa de cada unidad.** Aunque sea mala, corta
   o en borrador de voz. La IA nunca genera el primer borrador de un párrafo
   argumentativo.
2. La IA puede: armar el esqueleto de la unidad, traer cifras verificadas del
   monitoreo, proponer el orden de los argumentos, señalar huecos lógicos,
   sugerir dónde va cada cita, y marcar (no reescribir) frases confusas.
3. La IA puede reescribir SOLO cuando Federico lo pide sobre un párrafo suyo
   ya escrito, y mostrando el diff para que él acepte frase por frase.
4. Toda cifra en el texto sale de un archivo del repo (ver "Trazabilidad") —
   nunca de memoria de la IA ni de un chat anterior.
5. Estilo: español académico-formal, mínimo anglicismo, lenguaje llano sobre
   jerga (feedback explícito de la profesora — ver `memory/` y AGENTS.md).

## La unidad de trabajo

Cada sesión de escritura ataca UNA unidad = una fila de `00_mapa_tesis.md`:

```
unidad = capítulo/sección + resultado que sostiene + figura(s) + cita(s)
```

Flujo por unidad (60–90 min):
1. **Preparación (IA, 5 min):** la IA arma la "ficha de unidad": pregunta que
   responde la sección, cifras exactas con su archivo fuente, figuras
   disponibles, citas candidatas del `.bib`, y 3 preguntas que el lector se
   va a hacer.
2. **Escritura (Federico, 30–45 min):** prosa propia, sin IA, con la ficha a
   la vista. Vale escribir "acá va la figura X" y seguir.
3. **Crítica (IA, 10 min):** la IA verifica cifras contra el repo, marca
   huecos de argumento, citas faltantes y frases opacas. NO reescribe.
4. **Revisión (Federico, 15 min):** corrige con las marcas. Si pide
   reescritura de un párrafo, es con diff y decisión frase por frase.
5. **Cierre:** actualizar el estado de la unidad en `00_mapa_tesis.md`.

## Trazabilidad de cifras (fuente única)

| Qué | Archivo |
|---|---|
| Métricas por modelo/segmento (AUC/Gini/KS/Brier) | `data/pipeline/monitoreo/metricas.csv` |
| Matrices de confusión y costos por umbral | `data/pipeline/monitoreo/matrices_confusion.csv` |
| Resumen del experimento | `data/pipeline/monitoreo/resumen.json` |
| LendingClub: contaminación | `data/lendingclub/salidas/contaminacion.json` |
| LendingClub: Qwen zero/few-shot | `data/lendingclub/salidas/metricas_qwen.csv` |
| Figuras EDA y modelos | `figures/tesis/`, `figures/pipeline/` |
| Citas | `bibliografia/references.bib` (clave obligatoria en el texto) |
| Decisiones metodológicas y su porqué | `docs/DECISIONES_EXPERIMENTO_FINAL.md` |

Regla: si una cifra del texto no está en uno de estos archivos, no se publica
hasta regenerarla con el pipeline.

## Rol de Claude for Science / AI for Science

- **Claude Science (app)** hoy está orientada a biología/biomedicina; para esta
  tesis el equivalente funcional es este repo + Claude Code: pipeline
  reproducible, figuras con su código, y este método de redacción.
- **AI for Science (créditos API)**: programa de Anthropic con hasta USD 20.000
  en créditos API, evaluación mensual, foco en biología pero abierto a otros
  dominios. Si los costos de GPT/API pesan, se puede aplicar con el proyecto de
  la tesis (formulario en support.claude.com, artículo 11199177). Los créditos
  son solo API, no Claude.ai.

## Archivos de esta carpeta

- `00_mapa_tesis.md` — el mapa: unidades por capítulo con evidencia/figura/cita/estado.
- `ejercicios_escritura.md` — ejercicios para entrenar la escritura conjunta.
- `fichas/` — una ficha por unidad (las genera la IA en el paso 1; descartables).
