# Análisis del texto de los créditos (cohorte v2)

Generado por `pipeline/01_preprocesamiento/04_analisis_texto.py` sobre `02_variables.parquet` (4617 créditos; train 3232 / test 686). Todo lo entrenado usa solo train.

## 1. Cobertura

| columna | global | train | val | test | esparso | denso | mora (train) | no mora (train) |
|---|---|---|---|---|---|---|---|---|
| descripcion_negocio | 0.843 | 0.780 | 0.999 | 0.978 | 0.796 | 0.968 | 0.742 | 0.865 |
| subcategoria_texto | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tipo_credito | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| otra_categoria_negocio | 0.275 | 0.376 | 0.082 | 0.000 | 0.300 | 0.208 | 0.353 | 0.426 |
| objetivo_credito | 0.495 | 0.405 | 0.481 | 0.934 | 0.444 | 0.633 | 0.329 | 0.571 |

## 2. Longitud, cardinalidad y plantillas

| columna | n con texto | únicos | % repetido | palabras (mediana) | palabras (p90) | caracteres (mediana) |
|---|---|---|---|---|---|---|
| descripcion_negocio | 3891 | 3746 | 0.037 | 7.000 | 23.000 | 48.000 |
| subcategoria_texto | 4617 | 38 | 0.992 | 3.000 | 6.000 | 29.000 |
| tipo_credito | 4617 | 2 | 1.000 | 2.000 | 2.000 | 14.000 |
| otra_categoria_negocio | 1271 | 813 | 0.360 | 3.000 | 11.000 | 23.000 |
| objetivo_credito | 2285 | 1498 | 0.344 | 4.000 | 21.000 | 25.000 |

Descripciones más repetidas (¿plantillas?):

- (10×) «calzado»
- (9×) «ropa»
- (7×) «ropa deportiva»
- (7×) «comidas rapidas»
- (6×) «comidas rápidas»
- (6×) «zapatos»
- (5×) «licor»
- (5×) «frutas y verduras»

## 3. Tasa de mora por rubro (train)

Tasa de mora base en train: 0.687 (rubros con n ≥ 50).

| subcategoria_texto | n | tasa_mora |
|---|---|---|
| esteticas spa | 181 | 0.796 |
| comidas rapidas | 443 | 0.783 |
| peluqueria y manicuria | 73 | 0.712 |
| venta de productos para el hogar | 1500 | 0.712 |
| arreglos | 162 | 0.698 |
| tienda abarrotes | 178 | 0.652 |
| viveros | 77 | 0.623 |
| construccion | 93 | 0.602 |
| reparacion electrodomesticos | 90 | 0.578 |
| venta productos cuidado personal belleza | 61 | 0.459 |
| venta de accesorios bolsos y o bisuteria | 52 | 0.385 |

## 4. Tokens asociados a mora — descripción del negocio (train)

| token | n | log_odds_mora |
|---|---|---|
| negocio | 28 | 1.573 |
| carne | 31 | 1.357 |
| soy | 33 | 0.951 |
| hamburguesas | 36 | 0.864 |
| queso | 25 | 0.857 |
| almuerzos | 60 | 0.803 |
| uñas | 55 | 0.698 |
| relacionado | 78 | 0.613 |
| verduras | 29 | 0.608 |
| gaseosa | 26 | 0.474 |
| niños | 36 | -0.519 |
| general | 39 | -0.575 |
| muebles | 51 | -0.662 |
| camisetas | 25 | -0.698 |
| hogar | 50 | -0.701 |
| fabricación | 32 | -0.742 |
| bolsos | 39 | -0.771 |
| cuidado | 29 | -0.818 |
| alimentos | 26 | -0.912 |
| confección | 38 | -1.030 |

## 5. Señal predictiva del texto solo (TF-IDF + LogReg, train→test)

| fuente | features | AUC test | AUC esparso | AUC denso |
|---|---|---|---|---|
| descripcion_negocio | 611 | 0.482 | 0.465 | 0.493 |
| objetivo_credito | 138 | 0.391 | 0.388 | 0.405 |
| rubro + tipo | 107 | 0.512 | 0.543 | 0.482 |
| todo el texto | 1009 | 0.458 | 0.446 | 0.458 |

## 6. La ausencia de texto como señal

| columna | AUC test (solo indicador tiene/no tiene) | % mora con texto (train) | % mora sin texto (train) |
|---|---|---|---|
| descripcion_negocio | 0.502 | 0.653 | 0.807 |
| objetivo_credito | 0.544 | 0.558 | 0.774 |
