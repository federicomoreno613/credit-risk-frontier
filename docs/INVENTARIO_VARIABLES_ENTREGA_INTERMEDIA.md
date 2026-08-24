# Inventario de variables de la entrega intermedia

## Contrato resumido

- **20 variables de TransUnion:** fuente externa al formulario.
- **9 variables directas del formulario:** declaraciones disponibles antes de decidir.
- **1 campo libre:** `descripcion_negocio`, agregado solamente en dos perfiles de Qwen.
- **Perfil estructurado común:** 29 variables para Regresión Logística, XGBoost y Qwen.

El orden de las listas es parte del contrato. La definición ejecutable está en
`conf/base/parameters_intermediate_delivery.yml`.

## Atributos de TransUnion

Las descripciones siguientes provienen de la columna oficial
`definicion_oficial_CreditVision` del diccionario del proyecto. No se infieren a
partir de la abreviatura.

| Orden | Código | Uso | Definición oficial completa |
|---:|---|---|---|
| 1 | `AGG308` | Mora | Monto en mora agregado de obligaciones no hipotecarias en créditos financieros al mes M = 08 |
| 2 | `WD81` | Mora | Mora ponderada en créditos financieros en el mes M = 01 |
| 3 | `AGG2503` | Plazo Inferido | Plazo Inferido (relación de saldo sobre la cuota minima) agregado en el mes M=03 |
| 4 | `UTLMAG04` | Utilización | Magnitud de utilización de obligaciones retail en los últimos 24 meses (índice que mide la tendencia de 0 a 600) |
| 5 | `DUEMAG01` | Otro | Magnitud total de todas las obligaciones en los últimos 24 meses (índice que mide la tendencia de 0 a 600) |
| 6 | `AEPMAG01` | Pago Inferido | Magnitud del exceso de Pago Inferido agregado no hipotecario en los últimos 24 meses (índice que mide la tendencia de 0 a 600) |
| 7 | `BI21S` | Apertura | Meses desde la más reciente apertura bancaria en instalamentos |
| 8 | `LMD34S` | Utilización | Utilización de obligaciones bancarias vigentes sin garantía de mediano plazo reportadas en los últimos 12 meses |
| 9 | `RI27S` | Cuentas | Número de obligaciones actualmente vigentes y al día de retail instalamentos con 24 meses o más de antigüedad |
| 10 | `RLE904` | Pago Inferido | Exceso de Pago Inferido en cuentas de hipotecario en los últimos 6 meses |
| 11 | `TEL32S` | Saldo | Saldo máximo en obligaciones vigentes de telecomunicaciones reportadas en los últimos 12 meses |
| 12 | `TRANBAL09` | Saldo | Saldo asignado a obligaciones identificadas como transactor al mes 9 |
| 13 | `AT104S` | Apertura | Porcentaje de obligaciones aperturadas en los últimos 24 meses sobre el total de obligaciones |
| 14 | `SA21S` | Apertura | Meses desde la más reciente cuenta de ahorros aperturada |
| 15 | `AT103S` | Cuentas | Porcentaje de obligaciones vigentes y al día del total de obligaciones |
| 16 | `TEL03S` | Cuentas | Número de obligaciones vigentes al día de telecomunicaciones |
| 17 | `AT34AF` | Utilización | Utilización de obligaciones vigentes reportadas en los últimos 12 meses en créditos financieros |
| 18 | `G051S` | Mora | Porcentaje de obligaciones que alguna vez estuvo en mora |
| 19 | `AGG9316` | Mora | Monto agregado en mora al mes M = 16 |
| 20 | `WD03` | Mora | Mora ponderada en las obligaciones en el mes M = 06 |

Los valores negativos de estas columnas son códigos de ausencia. No representan
montos negativos ni menor riesgo. Los modelos clásicos los reciben como faltantes
y Qwen los lee como “sin historial”.

## Variables directas del formulario

| Orden | Columna | Significado | Motivo de inclusión |
|---:|---|---|---|
| 21 | `appusers_age` | Edad de la persona solicitante, en años | Dato directo de la persona, disponible antes de decidir |
| 22 | `credits_dependants_amount` | Cantidad de personas económicamente a cargo | Aproxima obligaciones del hogar sin usar una razón derivada |
| 23 | `credits_family_expenses` | Gastos familiares mensuales | Monto declarado relacionado con egresos del hogar |
| 24 | `shops_monthly_incomes` | Ingresos mensuales del negocio | Medida directa de actividad declarada |
| 25 | `shops_monthly_outcomes` | Egresos mensuales del negocio | Medida directa de costos declarados |
| 26 | `shops_daily_incomes` | Ingresos diarios del negocio | Medida complementaria con horizonte más corto |
| 27 | `shops_initial_capital` | Capital inicial del negocio | Declaración sobre el origen y escala inicial de la actividad |
| 28 | `shops_rent_amount` | Arriendo mensual del negocio | Compromiso mensual declarado de la actividad |
| 29 | `shops_shop_age` | Antigüedad del negocio, en años | Medida directa de permanencia de la actividad |

Las nueve variables son declaraciones, no estados contables auditados. Pueden
contener faltantes y valores extremos. Se conservaron porque su significado es
directo y explicable; no se eligieron según su asociación con el desenlace de
prueba.

Las nueve variables ingresan a los tres modelos. La Regresión Logística las
completa y estandariza junto con TransUnion; XGBoost conserva sus faltantes;
Qwen las recibe como parte de la serialización legible.

## Variables que no ingresan

No se usan condiciones finales del crédito, puntajes internos, identificadores,
fecha, partición, segmento ni etiqueta. Tampoco ingresan categorías codificadas,
canales, alianzas, género, educación u objetivo del crédito.

Se excluyen además `estimated_income`, `free_cash_flow`, `cost_ingress_ratio`,
`debts_savings` y `relacion_edad_deuda`. Son relaciones o estimaciones derivadas,
no respuestas directas independientes. Mantenerlas afuera facilita explicar qué
información adicional se entrega al pasar de TransUnion a TransUnion más formulario.
