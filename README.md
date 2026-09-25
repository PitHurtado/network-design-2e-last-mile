# Diseño estocástico de redes de última milla con dos escalones

Este repositorio estudia el diseño de una red de distribución de dos escalones:

```text
Centro de distribución (DC) ──► Satélites ──► Clústeres de clientes (píxeles)
```

El modelo decide dónde instalar satélites, qué capacidad contratar y cómo operar la
red en distintos períodos y escenarios de demanda. Los costos de ruteo se estiman
con *Continuous Approximation* (CA), por lo que el modelo no necesita enumerar rutas.

## Requisitos y preparación

- Python 3.12 o superior
- [Poetry](https://python-poetry.org/)
- Una licencia activa de Gurobi para ejecutar los modelos de optimización

Desde la raíz del repositorio:

```bash
poetry lock && poetry install        # instala también los comandos `scenarios`, `optimize`, `calibrate`
poetry install --with notebooks      # opcional: dependencias de los notebooks históricos
```

Los comandos se ejecutan desde la raíz con `poetry run <comando>`. Sin reinstalar, los
mismos comandos están disponibles como `poetry run python -m src.scenarios ...`,
`python -m src.optimization ...` y `python -m src.calibration ...`.

## Mapa del repositorio

```text
.
├── data/
│   ├── raw_*/                        # Entradas crudas: demanda, grilla, satélites, distancias
│   │   └── raw_facility/tariffs.json # Tarifas por nivel (OPEX, instalación, extrapolación, α)
│   ├── interim/panel_monthly.csv     # Panel mensual (caché) + panel_source.json con sus raws
│   ├── params/p<N>/                  # Parámetros ajustados oficiales (shape_params.json)
│   ├── scenarios/v<N>/<régimen>/<set>/  # Versiones oficiales de escenarios
│   ├── facilities/f<N>/              # Tablas oficiales de capacidad por satélite (capacity.json)
│   ├── sandbox/                      # Candidatas descartables (cp-*, cv-*, cf-*, cc-*)
│   └── _archive/                     # Artefactos previos al versionado
├── results/
│   ├── runs/r<N>/                    # Corridas oficiales de optimización (+ reports/)
│   ├── sandbox/runs/cr-*/            # Corridas candidatas
│   └── _archive/                     # Resultados previos al versionado
├── src/
│   ├── tools/          # Genérico: JSON, hashing, rutas, logging, artefactos, manifests, promoción, CLI
│   ├── core/           # Dominio compartido: constantes, entidades, lectores de raws, grilla, contrato en disco
│   ├── scenarios/      # Demanda histórica → parámetros → escenarios (fitting/, generation/, validation/, stages.py)
│   ├── optimization/   # Escenarios → CA → Gurobi → resultados (routing/, models/, experiments/, metrics/, stages.py)
│   ├── calibration/    # Tamaño de muestra SAA (esqueleto, no implementado)
│   └── visualization/  # Toolkit HTML/Plotly y un renderer por reporte
├── tests/              # Goldens de equivalencia, ciclo de vida de artefactos, arquitectura
├── docs/               # Artículo y presentaciones (local, no versionado)
├── OLD/                # Implementación previa (local, no versionada): referencia de lo no portado
└── pyproject.toml
```

Las dependencias entre paquetes van en un solo sentido, y `tests/test_architecture.py`
lo verifica:

```text
tools ← core ← { scenarios , optimization ← calibration }        visualization ← { tools, core }
```

`scenarios` y `optimization` nunca se importan entre sí: se encuentran en disco, a través
de `src/core/contract.py`. Los productores (`scenarios`, `optimization`) calculan y
llaman a `visualization` para renderizar.

## Versiones: candidata → validar → promover

Cada etapa produce un artefacto con versión propia: parámetros `p<N>`, escenarios `v<N>`,
tablas de capacidad `f<N>` y corridas `r<N>`. El linaje vigente es
`p1 → v1 → v2 (+ set capacity) → f1`. Todo comando escribe una **candidata** descartable
en `sandbox/`; sólo `promote` crea una versión oficial, y la versión oficial no se
modifica nunca.

- `manifest.json` registra todo lo necesario para reproducir el artefacto: el comando y
  su configuración, el sha256 de cada raw, el artefacto padre, las semillas, el commit
  de git y las versiones de librerías.
- `validate <ref>` corre los chequeos de la etapa y escribe `validation.json` (y el
  reporte en `reports/`).
- `promote <ref>` se niega si la candidata no pasó la validación, si cambió desde
  entonces, si algún padre no es oficial, si el árbol git tiene cambios o si HEAD avanzó
  desde que se generó la candidata. En parámetros, escenarios y tablas de capacidad además
  **re-ejecuta el comando y exige bytes idénticos**. Las corridas de Gurobi no se
  re-ejecutan (un MIP con límite de tiempo no es determinista): una hoja ya resuelta en
  una corrida oficial con las mismas entradas y parámetros se copia en vez de resolverse
  de nuevo.

## Flujos de ejecución

### 1. Usar las versiones oficiales

```bash
poetry run optimize verify --scenarios v2 --n 3     # CA + Gurobi sobre v2
poetry run scenarios explore v2                     # data/scenarios/v2/reports/explore.html
poetry run optimize report f1                       # data/facilities/f1/reports/capacity.html
```

Los archivos `scenario_*.json` no se versionan en git: se regeneran de forma exacta con
`scenarios generate --params p1` (y `promote` confirma que coinciden). `v2` es `v1` más
el set `capacity`: los 396 escenarios comunes son idénticos.

### 2. Nuevos parámetros y escenarios

Usa este flujo cuando cambien los eventos de demanda, el crosswalk, la grilla o el
ajuste estadístico. `panel build` requiere el archivo histórico de `data/raw_demand/`.

```bash
poetry run scenarios panel build                 # eventos → data/interim/panel_monthly.csv
poetry run scenarios params fit --n 100          # → cp-*: marginales, correlograma, multiplicadores de régimen
poetry run scenarios validate cp-...             # píxeles, objetivos ±1%, round-trip de ρ, R²
poetry run scenarios promote cp-...              # → p<N>

poetry run scenarios generate --params p2        # → cv-*: optimization (30), validation (100), capacity (100), expected, annual_expected
poetry run scenarios validate cv-...             # contrato + reports/validation.html
poetry run scenarios promote cv-...              # → v<N>
```

Si sólo cambia la política de regímenes: `scenarios params recalibrate --from p1`.
`scenarios list` muestra versiones y candidatas; `scenarios show <ref>` imprime un manifest.

Para comparar el baseline independiente con los dos enfoques conjuntos, con las mismas
marginales y 100 escenarios pareados por método:

```bash
poetry run scenarios compare --params p1 --regimes normal --n 100
open data/sandbox/comparisons/cc-.../reports/normal/demand_comparison.html
```

Compara `independent` (muestreo independiente), `spatial_joint` (cópula espacial
calibrada) y `historical_bootstrap` (remuestreo de vectores históricos conjuntos por
mes), y escribe junto al reporte `comparison_long.csv`, `metrics.csv`,
`pixel_period_metrics.csv`, `neighbor_metrics.csv`, `distance_metrics.csv` y
`summary.json`. Las comparaciones son exploratorias: quedan en sandbox y no se promueven.
El baseline `independent` no comparte ningún shock entre píxeles; la varianza del shock
común histórico se incorpora a la dispersión individual para mantener comparables las
marginales.

### 3. Capacidad por satélite

Los modelos con capacidad (`capacitated`, `flex`) no leen niveles ni costos desde
`input_facilities.xlsx` —que da a los 9 satélites la misma grilla 0–12 y los mismos
costos—, sino desde una tabla de capacidad versionada `f<N>`:

```bash
poetry run optimize capacity analyze --scenarios v2 [--levels percentiles-a]   # → cf-*
poetry run optimize validate cf-...      # todos los satélites, niveles 0<q1<…, nivel máximo cubre ≥95% de los picos, costos crecientes
poetry run optimize promote cf-...       # re-ejecuta y compara bytes → f<N>
```

El análisis asigna cada píxel al satélite más cercano, calcula con la CA la flota de vans
que necesita cada satélite en cada período de los escenarios del set `capacity`, y toma
su **flota pico** (máximo de los 12 períodos), juntando los tres regímenes. Los niveles
salen de esa distribución con un método configurable:

| `--levels` | Niveles instalables (siempre se agrega el 0) |
|---|---|
| `percentiles-a` (defecto) | De P25−2 a P95+2, de a 2 |
| `percentiles-b` | Los que cubren el P50, el P90 y el máximo |
| `fixed-grid` | 2 a 12 para todos (la grilla del Excel) |

Los costos salen de `data/raw_facility/tariffs.json` (OPEX e instalación por nivel, con
extrapolación por encima del mayor conocido): el costo de operación del nivel q en el
mes t es `OPEX(q) × [α + (1−α) × s(t)]`, con α = 0.70 y `s(t)` la estacionalidad de la
demanda de los píxeles del satélite en el escenario esperado.

### 4. Experimento de flexibilidad

Compara tres políticas de operación de capacidad bajo los tres regímenes y las tres
fuentes de decisión de primera etapa (`annual_expected`, `expected`, `optimization`).

```bash
poetry run optimize flexibility --scenarios v2 --facilities f1 --time-limit 600 --mip-gap 0   # → cr-*
poetry run optimize report cr-...        # reports/flexibility_comparison.html + summary.json
poetry run optimize validate cr-...      # sin hojas en ERROR; TIME_LIMIT queda como advertencia
poetry run optimize promote cr-...       # → r<N>
```

| Política | Decisión operacional permitida |
|---|---|
| `fixed_operation` | Siempre opera a la capacidad instalada. |
| `on_off_installed` | Puede apagarse o usar la capacidad instalada. |
| `up_to_installed` | Puede apagarse o elegir cualquier nivel menor o igual al instalado. |

Para un piloto acotado: `--regimes normal --flexibilities up_to_installed --cases optimization`.
Para solves reproducibles: `--threads 1 --seed 0`. `--model capacitated` corre el modelo
sin operación flexible (capacidad = nivel instalado, sin costo de operación); sus hojas
van bajo la política `none`. Sin `--facilities`, los modelos con capacidad fallan.

### 5. Evaluación fuera de muestra y VSS

Fija las decisiones de instalación de cada caso de una corrida y reoptimiza operación y
ruteo sobre los escenarios de validación. El *Value of the Stochastic Solution* (VSS)
positivo indica que la decisión estocástica redujo el costo medio de validación frente
al caso base.

```bash
poetry run optimize evaluate --run r1 --time-limit 600      # → cr-* con flexibility_evaluation/ (usa la f<N> de r1)
poetry run optimize benchmark --scenarios v2 --facilities f1 --time-limit 600   # RP_100 para el VSS teórico
poetry run optimize report cr-<evaluación> --benchmark cr-<benchmark>
```

Si `RP_100` alcanza `TIME_LIMIT`, el reporte entrega un intervalo de VSS basado en el
incumbente y la cota del solver; sólo muestra un VSS puntual cuando el benchmark es óptimo.

## Referencia de comandos

| Comando | Propósito | Produce |
|---|---|---|
| `scenarios panel build` | Panel mensual desde los eventos históricos. | `data/interim/panel_monthly.csv` |
| `scenarios params fit` / `recalibrate` | Ajuste estadístico y multiplicadores de régimen. | candidata `cp-*` |
| `scenarios generate --params <p>` | Los cinco sets por régimen y sus manifests. | candidata `cv-*` |
| `scenarios compare --params <p>` | Comparación de modelos de dependencia. | `cc-*` (sandbox) |
| `scenarios explore <v>` | Explorador comparativo low / normal / high. | `<v>/reports/explore.html` |
| `optimize capacity analyze --scenarios <v>` | Niveles y costos por satélite. | candidata `cf-*` |
| `optimize flexibility --scenarios <v> --facilities <f>` | Experimento régimen × política × caso. | candidata `cr-*` |
| `optimize evaluate --run <r>` | Recourse con Y fija sobre validation. | candidata `cr-*` |
| `optimize benchmark --scenarios <v> --facilities <f>` | `RP_100` para el VSS teórico. | candidata `cr-*` |
| `optimize report <r o f>` | Reportes HTML de una corrida o de una tabla de capacidad. | `<ref>/reports/` |
| `optimize verify --scenarios <v>` | Smoke test CA + Gurobi. | consola; falla si se rompe el contrato |
| `scenarios` / `optimize` `validate`, `promote`, `list`, `show` | Ciclo de vida de los artefactos. | `validation.json`, versión oficial |
| `calibrate sample-size` | Elección de N (planificado, no implementado). | — |

Cada comando acepta `--help`. Verificación: `poetry run python -m unittest discover -s tests -t .`
(o `pytest`); `GOLDEN_SKIP_SOLVE=1` omite los solves de Gurobi.

## Contrato de escenarios

Cada versión tiene las carpetas siguientes por régimen:

```text
data/scenarios/v<N>/<low|normal|high>/
├── optimization/     # 30 escenarios simulados; usados para optimizar
├── validation/       # 100 escenarios independientes; usados para evaluar
├── expected/         # Un escenario de demanda esperada en 12 períodos
├── annual_expected/  # Un período promedio anual (caso de un período del experimento)
└── capacity/         # 100 escenarios con streams propios; sólo para dimensionar satélites (v2 en adelante)
```

Cada conjunto incluye un manifiesto con IDs canónicos, semilla y el SHA-256 de los
parámetros. `annual_expected` no se puede usar como horizonte de 12 períodos en el
optimizador; sus costos variables se escalan por 12. El set `capacity` no se usa para
optimizar ni para evaluar, así el VSS sobre `validation` queda fuera de muestra.

## Modelo

Las variantes anidan (`Uncapacitated ⊂ Capacitated ⊂ Flex`) y cada una es una subclase
de la anterior que sólo agrega bloques de variables, objetivo y restricciones:

| Modelo (`--model`) | Archivo | Agrega |
|---|---|---|
| `uncapacitated` | `src/optimization/models/uncapacitated.py` | Asignación `X`, `W` y costos de ruteo (la base). |
| `capacitated` | `src/optimization/models/capacitated.py` | Instalación `Y[i,q]`, su costo, un nivel por satélite, capacidad sobre `Y`. |
| `flex` | `src/optimization/models/flex.py` | Operación `Z[i,q,t,n]`, su costo, una política `Z`~`Y`, capacidad sobre `Z`. |

Las políticas de `flex` son clases en `src/optimization/models/policies.py`. Para una
variante nueva: subclase de la más cercana, `NAME` (queda registrada y seleccionable con
`--model`), un dataclass `Features` con sus flags, `BLOCKS` con los bloques que introduce
(`before=` fija su posición) y los métodos de esos bloques; para cambiar la formulación
de un bloque existente se sobrescribe su método. Ablaciones sin clase nueva:
`features=replace(Modelo.DEFAULT_FEATURES, disabled_blocks=frozenset({"capacity"}))`.
Cada bloque declara qué variables usa (`USES`): apagar un bloque de variables que otro
bloque activo necesita falla al construir el modelo con un mensaje que dice qué más apagar.
Por ejemplo, `{"operation_cost"}` quita sólo el costo de operación y conserva Z, mientras
`{"operation"}` (las variables Z) exige apagar también sus restricciones.

Los niveles de capacidad y sus costos de instalación y operación vienen de la tabla
`f<N>` (§3); `input_facilities.xlsx` sólo aporta ubicación y costo de sourcing de cada
satélite. La función objetivo minimiza instalación más el costo esperado de operación y ruteo.
En cada período y escenario, cada píxel se atiende exactamente desde un satélite o
directamente desde el DC.

## Documentación y entregables

```bash
# Abrir la presentación migrada a PowerPoint
open docs/presentations/research_seminar_presentation.pptx

# Abrir el artículo compilado
open docs/paper/main.pdf
```

`OLD/` conserva código anterior como referencia. No es parte del flujo reproducible
descrito arriba.
