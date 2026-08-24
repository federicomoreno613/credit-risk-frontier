# Configuración vigente

`base/catalog.yml` declara las entradas y salidas del procesamiento de la
cohorte, de los dos brazos clásicos y de la consolidación con Qwen3-8B.

`base/parameters.yml` define el resultado observado, la partición temporal y los
controles de la cohorte. `base/parameters_intermediate_delivery.yml` es la fuente
legible del contrato de 20 variables TransUnion más 9 declaraciones directas y
de los parámetros que identifican las cuatro inferencias recuperables de Qwen.

`local/` queda reservado para credenciales o reemplazos propios de una máquina.
Su contenido, salvo `.gitkeep`, está excluido del control de versiones.
