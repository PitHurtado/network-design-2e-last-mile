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
│   ├── interim/panel_monthly.csv     # Panel mensual (caché) + panel_source.json con sus raws
│   ├── params/p<N>/                  # Parámetros ajustados oficiales (shape_params.json)
│   ├── scenarios/v<N>/<régimen>/<set>/  # Versiones oficiales de escenarios
│   ├── sandbox/                      # Candidatas descartables (cp-*, cv-*, cc-*)
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
├── docs/               # Artículo y presentaciones
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
corridas `r<N>`. Todo comando escribe una **candidata** descartable en `sandbox/`; sólo
`promote` crea una versión oficial, y la versión oficial no se modifica nunca.

- `manifest.json` registra todo lo necesario para reproducir el artefacto: el comando y
  su configuración, el sha256 de cada raw, el artefacto padre, las semillas, el commit
  de git y las versiones de librerías.
- `validate <ref>` corre los chequeos de la etapa y escribe `validation.json` (y el
  reporte en `reports/`).
- `promote <ref>` se niega si la candidata no pasó la validación, si cambió desde
  entonces, si algún padre no es oficial o si el árbol git tiene cambios. En parámetros y
  escenarios además **re-ejecuta el comando y exige bytes idénticos**. Las corridas de
  Gurobi no se re-ejecutan (un MIP con límite de tiempo no es determinista): una hoja ya
  resuelta en una corrida oficial con las mismas entradas y parámetros se copia en vez de
  resolverse de nuevo.

## Flujos de ejecución

### 1. Usar las versiones oficiales

```bash
poetry run optimize verify --scenarios v1 --n 3     # CA + Gurobi sobre v1
poetry run scenarios explore v1                     # data/scenarios/v1/reports/explore.html
open data/scenarios/v1/reports/validation.html      # reporte de validación de v1
```

Los archivos `scenario_*.json` no se versionan en git: se regeneran de forma exacta con
`scenarios generate --params p1` (y `promote` confirma que coinciden).

### 2. Nuevos parámetros y escenarios

Usa este flujo cuando cambien los eventos de demanda, el crosswalk, la grilla o el
ajuste estadístico. `panel build` requiere el archivo histórico de `data/raw_demand/`.

```bash
poetry run scenarios panel build                 # eventos → data/interim/panel_monthly.csv
poetry run scenarios params fit --n 100          # → cp-*: marginales, correlograma, multiplicadores de régimen
poetry run scenarios validate cp-...             # píxeles, objetivos ±1%, round-trip de ρ, R²
poetry run scenarios promote cp-...              # → p<N>

poetry run scenarios generate --params p2        # → cv-*: optimization (30), validation (100), expected, annual_expected
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

### 3. Experimento de flexibilidad

Compara tres políticas de operación de capacidad bajo los tres regímenes y las tres
fuentes de decisión de primera etapa (`annual_expected`, `expected`, `optimization`).

```bash
poetry run optimize flexibility --scenarios v1 --time-limit 600 --mip-gap 0   # → cr-*
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
Para solves reproducibles: `--threads 1 --seed 0`.

### 4. Evaluación fuera de muestra y VSS

Fija las decisiones de instalación de cada caso de una corrida y reoptimiza operación y
ruteo sobre los escenarios de validación. El *Value of the Stochastic Solution* (VSS)
positivo indica que la decisión estocástica redujo el costo medio de validación frente
al caso base.

```bash
poetry run optimize evaluate --run r1 --time-limit 600      # → cr-* con flexibility_evaluation/
poetry run optimize benchmark --scenarios v1 --time-limit 600   # RP_100 para el VSS teórico
poetry run optimize report cr-<evaluación> --benchmark cr-<benchmark>
```

Si `RP_100` alcanza `TIME_LIMIT`, el reporte entrega un intervalo de VSS basado en el
incumbente y la cota del solver; sólo muestra un VSS puntual cuando el benchmark es óptimo.

## Referencia de comandos

| Comando | Propósito | Produce |
|---|---|---|
| `scenarios panel build` | Panel mensual desde los eventos históricos. | `data/interim/panel_monthly.csv` |
| `scenarios params fit` / `recalibrate` | Ajuste estadístico y multiplicadores de régimen. | candidata `cp-*` |
| `scenarios generate --params <p>` | Los cuatro sets por régimen y sus manifests. | candidata `cv-*` |
| `scenarios compare --params <p>` | Comparación de modelos de dependencia. | `cc-*` (sandbox) |
| `scenarios explore <v>` | Explorador comparativo low / normal / high. | `<v>/reports/explore.html` |
| `optimize flexibility --scenarios <v>` | Experimento régimen × política × caso. | candidata `cr-*` |
| `optimize evaluate --run <r>` | Recourse con Y fija sobre validation. | candidata `cr-*` |
| `optimize benchmark --scenarios <v>` | `RP_100` para el VSS teórico. | candidata `cr-*` |
| `optimize report <r>` | Reportes HTML de una corrida. | `<r>/reports/` |
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
└── annual_expected/  # Un período promedio anual; sólo análisis descriptivo
```

Cada conjunto incluye un manifiesto con IDs canónicos, semilla y el SHA-256 de los
parámetros. `annual_expected` no se puede usar como horizonte de 12 períodos en el
optimizador.

## Modelo

La formulación comparte una base y activa bloques por variante:

| Modelo | Archivo | Característica |
|---|---|---|
| Sin capacidad | `src/optimization/models/uncapacitated.py` | Asignación y costo de ruteo. |
| Flexible | `src/optimization/models/flex.py` | Instalación `Y`, operación `Z`, costos y restricciones de capacidad. |

La función objetivo minimiza instalación más el costo esperado de operación y ruteo.
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
