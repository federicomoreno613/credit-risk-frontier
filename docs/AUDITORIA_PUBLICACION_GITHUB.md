# Auditoría del repositorio Kedro para publicación en GitHub

**Fecha:** 15 de julio de 2026  
**Alcance:** estructura Kedro, estado de Git, datos, privacidad y reproducción desde un clon sin archivos locales.

## Veredicto

El proyecto local está organizado y ejecuta correctamente el experimento
vigente, pero **el repositorio todavía no está listo para compartirse como
repositorio público**. El bloqueo principal no está en los modelos: está en la
publicación de datos individuales, el historial ya visible en GitHub y la falta
de un recorrido que funcione desde un clon sin los datos privados.

No se modificó la visibilidad del repositorio, no se reescribió su historial y
no se realizó ningún commit ni envío al remoto durante esta auditoría.

## 1. Situación crítica del remoto

El repositorio `federicomoreno613/credit-risk-frontier` es público. La rama
`master` contiene `data/dataset_tesis.csv`, un archivo de 3.261.631 bytes con
5.351 filas y 102 columnas. No corresponde a la cohorte analítica vigente de
4.201 casos.

El archivo remoto contiene registros individuales, fechas, identificadores
seudonimizados y campos de texto. La columna `descripcion_negocio` reúne 4.592
valores no vacíos y 3.987 descripciones distintas. La presencia de un
identificador transformado no elimina el riesgo de identificación indirecta
cuando se combina con edad, actividad, fechas, montos y descripciones libres.

La política actual del repositorio establece que los datos individuales y los
textos no deben publicarse. Por lo tanto, el contenido del remoto contradice la
política vigente.

Eliminar el archivo en un commit nuevo no alcanza: el objeto continúa accesible
en el historial. Antes de compartir el enlace conviene contener la exposición y
crear una historia pública limpia.

## 2. Estado de la organización Kedro

### Aspectos correctamente resueltos

- El proyecto usa Poetry, Python 3.11 y Kedro 1.2.
- `poetry check` no encuentra inconsistencias en la configuración.
- El registro expone `data_processing`, `intermediate_classics`, `intermediate_delivery` y `__default__`.
- Las rutas físicas están concentradas en `conf/base/catalog.yml`.
- La definición del desenlace, la separación temporal, las variables y las semillas están centralizadas en parámetros.
- La inferencia extensa de Qwen está aislada del recorrido predeterminado y utiliza archivos recuperables declarados en el catálogo.
- El ejecutable de Qwen carga y guarda mediante nombres del Data Catalog.
- La ejecución local de `intermediate_delivery` completó sus 19 tareas.
- Las 19 pruebas locales aprobaron.
- El catálogo OKF fue validado sin errores ni advertencias.
- No se detectaron patrones de secretos en el árbol activo, fuera de los datos deliberadamente excluidos del análisis de texto.

### Aspectos que impiden considerarlo reproducible desde GitHub

El recorrido predeterminado es seguro respecto de Qwen, pero necesita
`data/05_model_input/model_input_table.parquet`. Ese archivo está correctamente
ignorado porque contiene datos individuales. En consecuencia, un clon público
no puede ejecutar `kedro run` sin recibir previamente datos privados.

Se construyó una copia temporal del árbol actual sin `data/` y se ejecutaron las
pruebas con el mismo entorno instalado. El resultado fue:

```text
11 failed, 8 passed
```

Las fallas corresponden a pruebas que abren directamente las solicitudes, la
cohorte, el diccionario local o resultados persistidos. En esa misma copia,
`kedro run` se detuvo al intentar cargar `model_input_table.parquet`.

Esto no invalida la reproducción privada ya comprobada. Indica que deben
distinguirse dos niveles:

1. reproducción completa con acceso autorizado a los datos;
2. comprobación pública del código, los contratos y los resultados agregados mediante datos sintéticos.

## 3. Estado de Git

El árbol de trabajo contiene 215 entradas sin consolidar:

| Estado | Cantidad |
|---|---:|
| Modificados | 13 |
| Eliminados | 161 |
| Nuevos | 41 |
| Preparados para commit | 0 |

La rama local `fm/tesis-v16-target-fix` está doce commits por delante de
`origin/master`, pero su seguimiento está configurado contra el `master` local,
no contra una rama remota propia. Además, el rediseño vigente permanece en gran
parte sin commit. El remoto público, por lo tanto, no representa el estado que
produjo la entrega actual.

## 4. Qué puede publicarse

La publicación debería construirse con una lista explícita. No conviene decidir
por extensión ni publicar una carpeta completa.

| Candidato público | Condición |
|---|---|
| Código de `src/` y ejecutables vigentes | Después de revisar la eliminación de antecedentes fuera del alcance. |
| Configuración base sin credenciales | Manteniendo cualquier reemplazo privado en `conf/local/`. |
| Pruebas unitarias y datos sintéticos | Deben funcionar sin datos reales. |
| Diez figuras de la entrega | Son resultados agregados y fueron inspeccionadas. |
| `resultados_principales.csv` | Tabla agregada de seis configuraciones. |
| `paired_qwen_description_delta_auc.csv` | Comparación agregada y pareada. |
| `metrics.csv` | Puede publicarse como tabla agregada si se conserva la nota sobre casos válidos. |
| `resumen_campos_textuales.csv` | Publicable porque resume cobertura y extensión, sin copiar textos. |
| Contrato de variables sin registros | Publicable como documentación del experimento. |
| Resumen de la ejecución de Qwen | Publicable sin rutas locales ni mensajes por crédito. |
| Muestra completamente sintética | Debe reemplazar al ejemplo real en las pruebas públicas. |

En este momento, `.gitignore` excluye también las tablas agregadas bajo
`data/08_reporting/`. Por eso no se incorporarían a un commit normal. Conviene
crear un destino público específico, por ejemplo `reports/public/`, o habilitar
solamente nombres de archivos previamente aprobados.

## 5. Qué debe permanecer privado

| Archivo o grupo | Motivo |
|---|---|
| `data/01_raw/` | Solicitudes, pagos, atributos personales y comportamiento individual. |
| `data/02_intermediate/` | Puente entre identificadores seudonimizados. |
| `data/03_primary/` | Referencia individual y reconstrucción del desenlace. |
| `data/05_model_input/` | Cohorte completa con variables, textos, etiqueta y partición. |
| `data/07_model_output/` | Predicciones por crédito. |
| Cuatro archivos recuperables de Qwen | Incluyen identificador de evaluación, etiqueta, probabilidad y mensaje. |
| `combined_predictions_private.parquet` | Consolida predicciones individuales. |
| `split_manifest.json` | Contiene listas y huellas de identificadores por partición. |
| `ejemplo_serializacion_real.json` | Copia valores y texto de un registro real, aunque esté seudonimizado. |
| Registros de ejecución | Pueden contener rutas locales, mensajes o detalles operativos. |

El diccionario literal de TransUnion debe quedar privado hasta confirmar que su
licencia o autorización permite redistribuirlo. La entrega puede explicar el
significado de los atributos sin asumir que el archivo fuente completo es de
libre publicación.

## 6. Documentación todavía no portable

El README y varios documentos contienen rutas absolutas bajo
`/Users/federicomoreno/`, referencias a archivos de Downloads, adjuntos locales
y carpetas archivadas que no existen para otra persona. El Markdown y el DOCX
finales también están fuera del repositorio Kedro.

Antes de publicar deben utilizarse rutas relativas o una nota explícita de
disponibilidad. El informe local de validación puede conservar rutas privadas,
pero no debería ser el documento de entrada del repositorio público.

Tampoco existen todavía `LICENSE`, `CITATION.cff` ni una declaración específica
de disponibilidad y uso de datos. La licencia del código debe distinguirse de
la autorización sobre los datos, que no se presume.

## 7. Identidad de la ejecución de Qwen

La configuración conserva el nombre `qwen3:8b`, la temperatura, la ventana de
contexto y los límites de respuesta. La copia local ejecutada tiene el
identificador abreviado `500a1f067a9f`, pero ese identificador no está guardado
en los contratos publicados. Como una etiqueta local puede apuntar a otra
copia en el futuro, deben registrarse el identificador completo del modelo, la
cuantización y la versión de Ollama usada.

## 8. Telemetría

El archivo `.telemetry` está marcado para eliminarse y, durante las ejecuciones,
Kedro informó que enviaba telemetría anónima. Para que la decisión sea explícita
y consistente en un repositorio académico, conviene conservar una configuración
con `consent: false` o definir `KEDRO_DISABLE_TELEMETRY=1` en los comandos de
reproducción y en la integración continua.

## 9. Secuencia recomendada antes de publicar

1. Volver privado temporalmente el remoto actual o dejar de compartir su enlace mientras se define la limpieza.
2. Elegir entre un repositorio público nuevo —opción más simple— y una reescritura completa del historial existente.
3. Aprobar una lista de archivos públicos que excluya todo registro individual y todo texto original.
4. Consolidar el rediseño vigente en una rama de publicación limpia, sin antecedentes fuera del alcance.
5. Incorporar resultados agregados en una ruta pública declarada por Kedro.
6. Separar pruebas públicas con datos sintéticos de pruebas privadas de integración.
7. Definir un recorrido público predeterminado que funcione desde un clon sin datos confidenciales.
8. Incorporar el documento final o indicar formalmente dónde se distribuye.
9. Reemplazar rutas absolutas y agregar `LICENSE`, `CITATION.cff` y una declaración de disponibilidad de datos.
10. Registrar la identidad exacta de Qwen y desactivar explícitamente la telemetría.
11. Repetir en un clon limpio: instalación, registro, catálogo, recorrido público, pruebas y escaneo de secretos.

## 10. Comandos de comprobación utilizados

```bash
git status -sb
git status --porcelain=v1 -uall
git ls-files data figures models results lineage notebooks docs knowledge
git rev-list --left-right --count HEAD...origin/master
gh repo view federicomoreno613/credit-risk-frontier --json visibility,isPrivate,defaultBranchRef,url
gh api repos/federicomoreno613/credit-risk-frontier/contents/data/dataset_tesis.csv?ref=master
poetry check
poetry run kedro --version
poetry run kedro registry list
poetry run kedro catalog describe-datasets --pipeline intermediate_delivery
poetry run pytest -q
python3 /Users/federicomoreno/.codex/plugins/cache/personal/knowledge-catalog/0.1.0/scripts/validate_okf.py knowledge
```

También se ejecutaron `pytest -q` y `kedro run` sobre una copia temporal sin la
carpeta de datos para comprobar el comportamiento real de un clon público.
