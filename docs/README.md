# Documentación vigente de la entrega intermedia

Esta carpeta contiene únicamente las notas que describen el experimento activo.

El mapa navegable de fuentes, conceptos y procesos está en el
[`catálogo OKF`](../knowledge/index.md).

1. [`DISENO_ENTREGA_INTERMEDIA.md`](DISENO_ENTREGA_INTERMEDIA.md): pregunta, comparaciones y límites.
2. [`INVENTARIO_VARIABLES_ENTREGA_INTERMEDIA.md`](INVENTARIO_VARIABLES_ENTREGA_INTERMEDIA.md): las 20 variables de TransUnion, las 9 variables directas del formulario y el único campo libre.
3. [`FUENTE_DE_VERDAD_Y_REPRODUCCION.md`](FUENTE_DE_VERDAD_Y_REPRODUCCION.md): archivos, comandos Kedro, caches y controles automáticos.
4. [`ARCHIVO_HISTORICO_20260714.md`](ARCHIVO_HISTORICO_20260714.md): qué se retiró del repositorio activo y dónde quedó preservado.
5. [`INFORME_VALIDACION_ENTREGA_INTERMEDIA.md`](INFORME_VALIDACION_ENTREGA_INTERMEDIA.md): fuentes, comandos, pruebas, cifras verificadas y asuntos pendientes.
6. [`AUDITORIA_PUBLICACION_GITHUB.md`](AUDITORIA_PUBLICACION_GITHUB.md): evaluación de privacidad, datos publicables y reproducción desde un clon limpio.

La definición ejecutable de las variables se encuentra en
`conf/base/parameters_intermediate_delivery.yml`. Kedro materializa una copia
validada en `data/08_reporting/intermedia_20260714_redesign/feature_contract.json`.
