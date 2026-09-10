# Diseño estocástico de redes de última milla con dos escalones

Este repositorio estudia el diseño de una red de distribución de dos escalones:

```text
Centro de distribución (DC) ──► Satélites ──► Clústeres de clientes (píxeles)
```

El modelo decide dónde instalar satélites, qué capacidad contratar y cómo operar la
red en distintos períodos y escenarios de demanda. Los costos de ruteo se estiman
con *Continuous Approximation* (CA), por lo que el modelo no necesita enumerar rutas.

## Requisitos y preparación

- Python 3.10 o superior
- [Poetry](https://python-poetry.org/)
- Una licencia activa de Gurobi para ejecutar los modelos de optimización

Desde la raíz del repositorio:

```bash
poetry install
```

Para incluir las dependencias usadas sólo por notebooks históricos:

```bash
poetry install --with notebooks
```

Todos los comandos que siguen se ejecutan desde la raíz y usan `poetry run`.

## Mapa del repositorio

```text
.
├── data/                                      # Datos de entrada y escenarios
│   ├── raw_demand/                             # Eventos históricos de entrega
│   ├── raw_pixel/                              # Grilla, píxeles y crosswalk cliente→píxel
│   ├── raw_facility/                           # Ubicaciones, capacidades y costos de satélites
│   ├── raw_distance/                           # Matrices DC↔satélite y satélite↔píxel
│   ├── raw_location/                           # Coordenadas de apoyo
│   └── scenarios/                              # Panel, parámetros y escenarios versionados
│       ├── panel_monthly.csv                   # Panel mensual construido desde la demanda
│       ├── shape_params.json                   # Ajuste estadístico reproducible
│       └── generated/<versión>/<régimen>/<set>/ # Escenarios y manifiestos
├── src/
│   ├── core/                                  # Entidades, constantes, rutas, lectores y logging
│   ├── pipeline/                              # Demanda histórica → escenarios
│   │   ├── cli/                               # build_panel, fit_params, generate, analyze, explore
│   │   ├── reports/                           # Reportes HTML de validación y exploración
│   │   └── {crosswalk,marginals,spatial,...}.py
│   └── optimization/                          # Escenarios → CA → Gurobi → resultados
│       ├── routing/continuous_approximation.py # Costos analíticos de ruteo
│       ├── models/                            # Formulación base, sin capacidad y flexible
│       ├── experiments/                       # Experimentos de flexibilidad y evaluación VSS
│       ├── reports/                           # Reportes HTML de resultados
│       └── cli/                               # Smoke test, experimentos y reportes
├── results/                                   # HTML, JSON y resultados de las corridas
│   ├── analysis/<versión>/                    # Validación de escenarios
│   ├── flexibility/<versión>/                 # Comparación de políticas de flexibilidad
│   └── flexibility_evaluation/<versión>/      # Evaluación fuera de muestra y VSS
├── docs/
│   ├── paper/                                 # Artículo LaTeX y PDFs compilados
│   └── presentations/                         # Presentación Keynote y copia PowerPoint
├── OLD/                                       # Implementación previa; referencia, no flujo principal
├── pyproject.toml                             # Dependencias y configuración de herramientas
└── README.md
```

La dependencia entre módulos es intencional y unidireccional:

```text
datos históricos → pipeline → escenarios en disco → optimización → resultados/reportes
                         │                               │
                         └────────── src/core ───────────┘
```

Comprueba que las capas no se crucen:

```bash
rg '^from src\.optimization' src/pipeline      # no debe devolver resultados
rg '^from src\.pipeline' src/optimization      # no debe devolver resultados
```

## Flujos de ejecución

### 1. Usar los escenarios ya disponibles

El repositorio incluye escenarios `v3`. Este es el camino más rápido para verificar
el modelo y regenerar los reportes, sin reprocesar la demanda histórica.

```bash
# Prueba de integración: escenarios → CA → Gurobi
poetry run python -m src.optimization.cli.verify_end_to_end --n 3

# Validar y explorar los escenarios v3
poetry run python -m src.pipeline.cli.analyze --version v3
poetry run python -m src.pipeline.cli.explore --version v3
open results/analysis/v3/scenario_validation.html
open results/v3/explore_scenarios.html
```

### 2. Reconstruir los escenarios desde datos históricos

Usa este flujo sólo cuando cambien los eventos de demanda, el crosswalk, la grilla o
el ajuste estadístico. `build_panel` requiere el archivo histórico de
`data/raw_demand/`.

```bash
# Eventos históricos → panel mensual
poetry run python -m src.pipeline.cli.build_panel

# Panel → parámetros marginales, espaciales y de régimen
poetry run python -m src.pipeline.cli.fit_params --n 50

# Parámetros → conjuntos de escenarios para low, normal y high
poetry run python -m src.pipeline.cli.generate --all --version v4

# Chequeos de contrato y explorador comparativo
poetry run python -m src.pipeline.cli.analyze --version v4
poetry run python -m src.pipeline.cli.explore --version v4
```

Para revisar el panel sin sobrescribir `data/scenarios/panel_monthly.csv`:

```bash
poetry run python -m src.pipeline.cli.build_panel --dry-run
```

### 3. Recalibrar regímenes sin refitar la historia

Cuando cambie la política de demanda baja/normal/alta, pero se mantengan los
parámetros marginales y espaciales, recalibra `shape_params.json` y genera una nueva
versión de escenarios.

```bash
poetry run python -m src.pipeline.cli.recalibrate_regimes --validation-n 100
poetry run python -m src.pipeline.cli.generate --all --version v4
poetry run python -m src.pipeline.cli.analyze --version v4
poetry run python -m src.pipeline.cli.explore --version v4
```

No uses `--overwrite` salvo que quieras reemplazar deliberadamente una versión ya
generada; el generador protege los directorios existentes.

### 4. Ejecutar el experimento de flexibilidad

Compara tres políticas de operación de capacidad bajo los tres regímenes y las tres
fuentes de decisión de primera etapa (`annual_expected`, `expected`, `optimization`).

```bash
poetry run python -m src.optimization.cli.run_flexibility_experiment \
  --version v3 --time-limit 300 --mip-gap 0

poetry run python -m src.optimization.cli.report_flexibility_experiment --version v3
open results/flexibility/v3/flexibility_comparison.html
```

Las políticas son:

| Política | Decisión operacional permitida |
|---|---|
| `fixed_operation` | Siempre opera a la capacidad instalada. |
| `on_off_installed` | Puede apagarse o usar la capacidad instalada. |
| `up_to_installed` | Puede apagarse o elegir cualquier nivel menor o igual al instalado. |

Para un piloto acotado, limita régimen, política y caso:

```bash
poetry run python -m src.optimization.cli.run_flexibility_experiment \
  --version v3 --regimes normal --flexibilities up_to_installed \
  --cases optimization --time-limit 300 --mip-gap 0
```

### 5. Evaluación fuera de muestra y VSS

Fija las decisiones de instalación de cada caso y reoptimiza operación y ruteo sobre
los escenarios de validación. El reporte calcula el *Value of the Stochastic
Solution* (VSS): un valor positivo indica que la decisión estocástica redujo el costo
medio de validación frente al caso base.

```bash
poetry run python -m src.optimization.cli.evaluate_flexibility_experiment \
  --version v3 --time-limit 600

poetry run python -m src.optimization.cli.report_flexibility_evaluation --version v3
open results/flexibility_evaluation/v3/vss_comparison.html
```

## Referencia de comandos

| Comando | Propósito | Salida principal |
|---|---|---|
| `src.pipeline.cli.build_panel` | Construye el panel mensual desde eventos históricos. | `data/scenarios/panel_monthly.csv` |
| `src.pipeline.cli.fit_params` | Ajusta marginales, estructura espacial y regímenes. | `data/scenarios/shape_params.json` |
| `src.pipeline.cli.recalibrate_regimes` | Actualiza sólo multiplicadores de régimen. | `shape_params.json` actualizado |
| `src.pipeline.cli.generate` | Genera escenarios inmutables y manifiestos. | `data/scenarios/generated/...` |
| `src.pipeline.cli.analyze` | Valida invariantes de los escenarios. | `results/analysis/<versión>/scenario_validation.html` |
| `src.pipeline.cli.explore` | Compara low, normal y high espacialmente. | `results/<versión>/explore_scenarios.html` |
| `src.optimization.cli.verify_end_to_end` | Smoke test de CA, escenarios y Gurobi. | Salida de consola; falla si el contrato se rompe. |
| `src.optimization.cli.run_flexibility_experiment` | Corre las configuraciones de flexibilidad. | `results/flexibility/<versión>/.../result.json` |
| `src.optimization.cli.report_flexibility_experiment` | Construye la comparación de flexibilidad. | `flexibility_comparison.html`, `summary.json` |
| `src.optimization.cli.evaluate_flexibility_experiment` | Evalúa decisiones fijas sobre validación. | `results/flexibility_evaluation/<versión>/.../evaluation.json` |
| `src.optimization.cli.report_flexibility_evaluation` | Construye el reporte VSS. | `vss_comparison.html`, `vss_summary.json` |

Para ver argumentos disponibles de cualquier comando:

```bash
poetry run python -m src.optimization.cli.run_flexibility_experiment --help
```

## Contrato de escenarios

Cada versión tiene las carpetas siguientes por régimen:

```text
generated/<versión>/<low|normal|high>/
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
