# Escenarios de demanda — documentación

**Versión de parámetros:** 2 · **Ajustado el:** 2026-09-08
**Código:** `src/pipeline/` (módulos, `reports/` y `cli/`); los lectores de inputs crudos están en `src/core/inputs.py`

Este documento describe el procedimiento **vigente**. Reemplaza por completo la versión
anterior, que documentaba un código distinto del que corría (ver §9).

---

## 1. Por qué se rehizo

El procedimiento anterior tenía cuatro defectos que invalidaban sus escenarios para el
paper:

1. **Cero correlación espacial y cero factor común.** Los píxeles se muestreaban
   independientes, así que la varianza de la demanda agregada se diluía como ~1/√161.
   El modelo decide dónde abrir satélites y cuánta capacidad instalar mirando la
   demanda agregada de un área de servicio; sin correlación, esa demanda casi no varía
   y la solución SAA parece más robusta de lo que es. Medido: el CV de la demanda
   agregada por período pasa de **0.0254** (independiente, sin factor común) a
   **0.1164** con la estructura ajustada — un factor **4.6×** de riesgo agregado que
   antes el modelo no veía.
2. **Johnson SU se ajustaba y no se usaba.** `fit_distribution_params.py` producía 24
   distribuciones JSU; el generador las cargaba, las pasaba a las funciones de reporte
   y nunca las referenciaba. El muestreo real era ruido log-normal sobre medias
   empíricas. Los dos HTML publicados afirmaban "drop modelado por Johnson SU".
3. **σ sin calibrar.** `SIGMA_STOPS = 0.90` implica un CV log-normal de 1.117 contra un
   CV interanual observado de mediana **0.124**: ~7× de más. Se veía en el output
   (`B-106` con `stop = [4,3,4,…]` en un escenario contra `[12,11,12,…]` en el
   esperado).
4. **El escenario "expected" no era ni media ni mediana.** Con `drop = mu·exp(ξ)` y σ
   variable por mes, la media quedaba 13% sobre `mu` en meses planos y **79% en
   diciembre**, y la mediana en 0.67×. Usarlo como benchmark determinista sesgaba toda
   comparación.

---

## 2. Data de entrada

| Archivo | Rol | Versionado |
|---|---|---|
| `data/raw_demand/base_customers_all_years_z7.csv` | eventos de entrega diarios | no (192 MB) |
| `data/raw_pixel/customer_pixel_layer.csv` | crosswalk cliente → `(layer, pixel)` | sí |
| `data/raw_pixel/grid_pixels.geojson` | geometría de la grilla | sí |
| `data/raw_pixel/input_pixels.xlsx` | píxeles del modelo: `area_surface`, `speed_intra_stop` | sí |
| `data/scenarios/shape_params.json` | **parámetros ajustados: la entrada reproducible** | sí |
| `data/scenarios/panel_monthly.csv` | panel derivado (caché) | no |
| `data/scenarios/generated/<versión>/<régimen>/<set>/` | escenarios generados versionados | no |

### 2.1 El fan-out del archivo crudo

`base_customers_all_years_z7.csv` tiene 2,768,835 filas, de las cuales **63.2% son
duplicados exactos**. No son paquetes repetidos: es fan-out de un join. La evidencia es
concluyente —

- la multiplicidad de las filas de cada cliente **iguala exactamente su cantidad de años
  activos, en el 100% de los 13,964 clientes**;
- el techo de multiplicidad es duro en 3 (= cantidad de años), sin cola hacia 4+;
- sobrevive una columna `_merge == "both"` constante.

`drop_duplicates()` sobre las 8 columnas deja **1,019,698 filas** y la demanda total pasa
de 10,694,636 a **3,924,851**. `src/pipeline/demand_panel.py` aborta si la proporción de
fan-out se aparta de lo verificado, en lugar de seguir con totales inflados 2.7×.

### 2.2 Otras correcciones de data

- **`cluster_targets` es H3 resolución 8, no geohash z7** — el nombre del archivo miente.
  Además H3 **no anida** en la grilla de píxeles (una celda H3 toca hasta 5 celdas), así
  que no se usa como clave espacial.
- **`c_ter_act`** es un código de territorio que varía en el tiempo (9,995 clientes
  cambian): inservible como clave espacial. Se descarta.
- **Negativos** (2.4% de las filas, devoluciones): se **netean** contra las entregas del
  mismo cliente-día en lugar de descartarse; los cliente-día que quedan ≤ 0 se eliminan.
- **2021-02 se excluye**: tiene 23,401 unidades de demanda contra 76,680 de mediana de
  febrero, con cobertura de fechas normal. Es un hueco de extracción, no estacionalidad.
- **Domingos y lunes**: faltan 154 de 156 domingos y los lunes están ~35% por debajo de
  Mar-Sáb. La agregación diario→mensual usa **fechas de entrega reales**, no días
  calendario.

---

## 3. Construcción de los píxeles y el `layer`

Los 161 `(layer, pixel)` fueron construidos **a mano**: los clientes se separan por clase
de drop (`layer`: A = drop bajo, B = drop alto), se agregan sobre la grilla de ~1 km², y
las celdas con menos de 5 clientes promedio por período se fusionan con una celda
adyacente con demanda formando un rectángulo, recalculando drop, demanda y clientes sobre
el área fusionada.

Verificaciones sobre esa construcción:

- **`area_surface` == cantidad de celdas de 1 km² fusionadas: 100% de coincidencia** en
  los 161 keys (122 con 1 celda, 32 con 2, 6 con 3, 1 con 4). Confirma que `area` está en
  km², que es lo que la CA asume al calcular `density = stop / area`.
- **La grilla es regular**: 16 × 19 = 304 celdas con `pixel = row*16 + col`, verificado en
  las 304. Da adyacencia exacta y distancias entre centroides sin estimar nada.
- Los `lon/lat` de `raw_pixels.csv` **no** son centroides geométricos: son puntos
  representativos ponderados por demanda, y algunos caen fuera de su propia celda. Para
  geometría se usa el geojson; esos `lon/lat` son el punto de servicio.
- La regla de ≥5 clientes quedó **incompleta en el trabajo manual**: 17 de 161 keys
  estaban bajo 5 clientes promedio (mínimo 2.25) y 13 de ellos sin fusionar. Con el panel
  nuevo (todos los clientes, no solo el crosswalk) bajan a **2 de 161**.

### 3.1 Imputación del `layer` — el supuesto principal

El archivo crudo no trae `layer`, y el crosswalk manual solo cubre **3,969 de 13,964
clientes = 51.3% de la demanda**. Para el resto:

- el `pixel` se asigna por la celda de la grilla, mapeada a la huella fusionada que la
  contiene (con fallback al píxel más cercano del mismo layer cuando la celda no
  pertenece a ninguna huella de ese layer: 91 casos);
- el `layer` se imputa con el umbral **drop ≥ 11.62 → B**.

Ese umbral es el mejor umbral global y reproduce las etiquetas manuales con **92.1% de
acierto**. **El `layer` no es una función determinista del drop**: el umbral óptimo
*por píxel* — la cota superior de cualquier regla local — llega solo a **95.2%**, y de los
58 píxeles con ambos layers, **48 tienen rangos de drop solapados** entre A y B. Hay un
componente exógeno del trabajo manual que no se recupera.

Consecuencia: ~8% de los clientes imputados quedan en el layer equivocado. Cada fila del
panel lleva `crosswalk_share`, la fracción de sus items con layer conocido (media
**0.553**), y el reporte de validación tiene una sección de sensibilidad. **66 de 161
píxeles dependen de la imputación para más de la mitad de sus items.**

Solo **24 clientes** (0.1% de la demanda) quedan fuera de la grilla y se descartan.

---

## 4. El modelo generador

### 4.1 Descomposición del panel

Para `stop` (clientes activos) y `drop` (items por visita), en logs:

```
log q[j, y, m] = level[j] + trend[y] + season[layer(j), m] + residual
residual[j, y, m] = common[y, m] + deviation[j, y, m]
```

Normalización: `trend[2022] = 0` (el nivel es el del año base) y la estacionalidad de cada
layer promedia 0. `common` es el shock del período compartido por todos los píxeles;
`deviation` es lo específico de cada píxel, y es lo que la cópula correlaciona.

La estacionalidad se estima *pooled por layer* y la dispersión se agrupa por **clase de
tamaño de píxel** (cuartiles de clientes promedio), porque cada celda `(pixel, mes)` tiene
solo 3 observaciones anuales.

Valores ajustados:

| | σ_común | σ_desviación (mediana) | tendencia 2020 / 2021 / 2022 |
|---|---|---|---|
| `stop` | 0.1063 | 0.1714 (rango 0.167–0.195) | −0.503 / −0.118 / 0 |
| `drop` | 0.0539 | 0.1901 (rango 0.152–0.331) | −0.330 / −0.295 / 0 |

Correlación entre los shocks comunes de `stop` y `drop`: **−0.315**. Se generan
independientes; modelarla es una extensión pendiente.

### 4.2 Estructura espacial

Cópula gaussiana con correlación latente

```
corr(h) = plateau + (1 − nugget − plateau) · exp(−h / ρ)      h > 0
corr(0) = 1
```

ajustada por mínimos cuadrados ponderados contra el correlograma empírico de las
desviaciones. Se comparan dos modelos y gana el de mejor R² ponderado:

| modelo | ρ (km) | nugget | plateau | R² ponderado |
|---|---|---|---|---|
| exponencial | 0.913 | 0.274 | — | 0.639 |
| **exponencial + plateau** | **0.671** | **0.000** | **0.0274** | **0.884** |

El correlograma empírico **no es monótono**: decae en los primeros km y se estabiliza en
lugar de ir a cero, por eso la exponencial pura subajusta. La lectura: la dependencia es
de rango muy corto (ρ ≈ 0.67 km, del orden de una celda) más un componente city-wide
uniforme.

El correlograma usa las desviaciones **balanceadas y recentradas** sobre los 148 píxeles
con historia completa. Esto importa: centrar sobre los 161 y correlacionar solo los
completos deja las sumas de columna distintas de cero e **infla el plateau de 0.027 a
0.047**.

Como las marginales son log-normales, la cópula gaussiana se reduce exactamente a una
log-normal multivariada: no hace falta calibración NORTA para la parte continua. La
discretización entra solo por el redondeo de `stop`.

### 4.3 Muestreo

Por período, con todos los factores **mean-preserving** (`exp(ξ − σ²/2)`):

```
F_t         ~ shock común del período
Z ~ N(0, Σ) ~ campo espacial;  dev_j = exp(σ_j · Z_j − σ_j²/2)
stop[j,t]   = max(1, round( E_stop[j,t] · multiplicador^0.70 · F_t · dev_j ))
drop[j,t]   = E_drop[j,t] · multiplicador^0.30 · G_t · dev'_j
demand[j,t] = stop[j,t] · drop[j,t]
```

Que los factores preserven la media es la corrección del defecto 4: antes amplificar σ en
los meses pico también movía su media, duplicando la estacionalidad.

**Semillas:** `np.random.SeedSequence(SEED_BASE).spawn(N)`, una por escenario, en lugar de
`SEED_BASE + i`. Los draws se indexan por `id_pixel` ordenado y no por orden de iteración
de diccionario, así que agregar o reordenar un píxel no desplaza los draws de los demás.

**El piso `max(1, ·)` no liga nunca** con los σ ajustados (0.000% de las celdas), medido y
registrado en el manifest de cada régimen. Eso importa porque el piso sesga `stop` hacia
arriba en los píxeles chicos.

### 4.4 Escenarios deterministas

Se emite **`expected`**: todos los factores en su **media** (= 1), con 12 períodos. Es
el escenario de valor esperado, el que corresponde usar para EV/VSS. Su agregado
`annual_expected` se guarda por separado como promedio anual descriptivo.

### 4.5 Sets, versiones, semillas e identificadores

Cada corrida se guarda sin mezclar propósitos en:

```
data/scenarios/generated/<versión>/<régimen>/<set>/
```

Para cada régimen se generan cuatro sets: `optimization` (30 escenarios simulados),
`validation` (100 simulados independientes), `expected` (un escenario de 12 períodos
con todos los shocks en su media) y `annual_expected` (un único período que es el
promedio de los 12 períodos de `expected`).

Los IDs son canónicos y estables, por ejemplo `v3-normal-optimization-001`; el orden
de uso está explícitamente en `manifest.json`, nunca se infiere del orden de archivos.
Los sets simulados usan `SeedSequence([seed_base, set_code]).spawn(index)`, con códigos
30 y 100 para optimización y validación. Así los sets no comparten draws, y el mismo
índice de los tres regímenes sí comparte el shock base para permitir comparaciones.

Cada manifest guarda la versión, los IDs, la semilla, el esquema de semillas, los
factores de régimen y el SHA-256 de `shape_params.json`. Para repetir una corrida se
usan el mismo `--version`, `--seed-base`, parámetros y tamaños; para no sobrescribir un
artefacto publicado se debe usar una etiqueta de versión nueva.

`annual_expected` es un artefacto descriptivo de un período. Sus `stop` pueden ser
fraccionarios al ser promedios y **no** pertenece al contrato de optimización ni puede
pasarse al modelo, que exige exactamente 12 períodos.

---

## 5. Regímenes de demanda

Un régimen reparte un escalar calibrado entre **cantidad de visitas** (exponente 0.70)
y **tamaño por visita** (exponente 0.30), para que la demanda esperada por período
iguale un objetivo. Los exponentes suman uno: antes del redondeo de `stop`, el producto
de ambos factores equivale al multiplicador del régimen.

Los objetivos son los **cuantiles crudos p10 / p50 / p90 de los totales por período
observados** en los 35 meses usables:

| régimen | objetivo | multiplicador | realizado |
|---|---|---|---|
| `low` | 20,465 | 0.4553 | 20,465 |
| `normal` | 32,678 | 0.7270 | 32,679 |
| `high` | 48,320 | 1.0749 | 48,320 |

La calibración consume **las mismas semillas** que la generación, así que el realizado da
en el objetivo en lugar de quedar a una muestra de distancia (sin eso quedaba ~2% abajo).
El multiplicador depende por tanto de `N` y `seed_base`, y ambos quedan en el manifest.

### 5.1 Unidades — la trampa a no repetir

Los objetivos están en **demanda del modelo**, `Σ_j (stop_j × drop_j)`, que representa
**una ronda de reparto representativa del período**. Eso **no** son los items mensuales
crudos: la demanda del modelo promedia ~45,100 por período en 2022 mientras los items
mensuales promedian ~149,200, porque un cliente se atiende varias veces al mes. Mezclar
las dos escalas infla todos los costos de ruteo ~3×.

### 5.2 Limitación declarada: los regímenes incorporan la tendencia

Anclar a cuantiles crudos significa que el nivel de cada régimen mezcla riesgo con
crecimiento. Con 2020 → 2021 = +24.4% y 2021 → 2022 = +50.9%, el régimen `normal`
(32,678) queda **por debajo del promedio por período de 2022 (45,103)**: representa un
negocio más chico que el último año observado, y `low` se parece a 2020.

La alternativa destendenciada — descomponer en tendencia × estacionalidad × shock y tomar
los cuantiles del **shock** — daría multiplicadores relativos de **0.71 / 0.95 / 1.34**
sobre el nivel del año base, y "alta" significaría un mes anormalmente fuerte *dado su mes
y su año*. Queda documentada como variante de sensibilidad; no está implementada.

---

## 6. El contrato de escenarios

El loader (`src/core/inputs.py`) lee **solo** `data["pixels"]`, y de cada píxel cuatro campos:

| campo | tipo | largo |
|---|---|---|
| `id_pixel` | `str`, formato `f"{LAYER}-{pixel}"` | — |
| `stop` | `list[int]`, **≥ 1** | 12 |
| `drop` | `list[float]`, **> 0** | 12 |
| `demand` | `list[float]`, `== stop × drop` | 12 |

`id_scenario` y `type` se escriben y se conservan en el manifest del set. El optimizador
lee esos IDs canónicos desde el manifest, no desde el orden de archivos. `k`, `lon`, `lat` y `area_surface` no son parte del contrato — vienen de
`input_pixels.xlsx`.

**Por qué los invariantes son duros:** la CA solo escribe claves de costo cuando
`demand > 0`, y `BaseSAAModel._obj_routing_facilities` las indexa directo → un solo píxel-período en
cero es un `KeyError` en el solve. Además `drop` es divisor (`capacity / drop`) y `density`
va bajo un `sqrt` en un divisor → un cero ahí es `ZeroDivisionError` antes de construir el
modelo. Y el conjunto de píxeles debe coincidir con `input_pixels.xlsx`: un píxel que no
esté ahí se **descarta en silencio**.

---

## 7. Reproducir

```bash
poetry install
poetry shell

python -m src.pipeline.cli.build_panel          # raw -> panel mensual
python -m src.pipeline.cli.fit_params --n 50    # panel -> shape_params.json
python -m src.pipeline.cli.generate --all --version v3
python -m src.pipeline.cli.analyze              # reporte de validación
python -m src.pipeline.cli.explore              # explorador comparativo
python -m src.optimization.cli.verify_end_to_end --n 3   # CA + Gurobi
```

Si solo cambia la política de régimen y no se dispone del panel histórico, se puede
recalibrar desde el `shape_params.json` persistido:

```bash
python -m src.pipeline.cli.recalibrate_regimes --validation-n 100
python -m src.pipeline.cli.generate --all --version v3
python -m src.pipeline.cli.analyze
python -m src.pipeline.cli.explore
```

Con `shape_params.json` y los raws versionados, `generate` reproduce los escenarios sin
necesitar el CSV de 192 MB ni el panel.

El reporte queda en `results/analysis/scenario_validation.html`. `analyze` sale con código
distinto de cero si algún invariante del contrato falla, así que sirve de compuerta.

---

## 8. Validación y limitación conocida

`verify_end_to_end` confirma que la CA y el modelo Gurobi corren sin cambios y que el
objetivo ordena los regímenes (con N=3, distancias euclidianas): low 519,562 · normal
827,951 · high 1,222,065, los tres OPTIMAL.

**Limitación abierta — el plateau no se reproduce.** El reporte incluye un *round-trip*:
sintetiza un panel del tamaño del real desde una Σ conocida y lo pasa por el estimador
completo. Recupera **ρ = 0.65 km contra 0.67 de entrada** (bien) pero **plateau = 0.000
contra 0.027** (no). La causa está identificada: el generador aplica un único factor común
idéntico a todos los píxeles, así que el centrado por período del estimador lo elimina
entero; en la data real el plateau sobrevive a ese centrado, lo que implica que viene de
**cargas heterogéneas** sobre el shock city-wide (píxeles que responden con distinta
intensidad), y eso el modelo no lo tiene.

El impacto está acotado: sobre el CV de la demanda agregada por período el plateau aporta
**+3.0%**, mientras que el modelo completo lleva ese CV de 0.0254 a 0.1164 (**4.6×**). El
mecanismo dominante de riesgo agregado sí está reproducido. La extensión natural es un
`β_j` por píxel sobre el factor común.

El estimador está verificado como insesgado: con `plateau = 0` verdadero recupera
`plateau = 0.000` y el promedio ponderado del correlograma da −0.0061 ≈ −1/160, lo que
predice la teoría. No fabrica estructura.

---

## 9. Divergencias con la documentación anterior

La versión previa de este archivo describía un código distinto del que corría:

| Documentaba | El código hacía |
|---|---|
| `SIGMA_BASE = 0.35`, `BULLWHIP_ALPHA = 2.5` | `0.50` / `4.0` |
| no mencionaba `SIGMA_STOPS` ni `SIGMA_PIXEL_DROP` | ambos `0.90` |
| índice estacional = media analítica JSU | media empírica ponderada por `base_stops` |
| `mu_t` = media analítica JSU | media empírica por píxel-mes; **JSU sin usar** |
| `stop` fijo al año 2022, 12 valores iguales | `stop` variaba por mes **y** por escenario |
| bullwhip "sin mover el nivel esperado" | sí movía la media (hasta +79% en diciembre) |
| `n_stops` fijo entre escenarios `[descartado]` | estaba aleatorizado con σ = 0.9 |

Los comentarios inline también se contradecían: `exp(ξ − σ²/2)` etiquetado como
"median-preserving" cuando preserva la **media** (su mediana es 0.667×).
