# Changelog

Todos los cambios relevantes de este proyecto se registran en este archivo.
El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el
proyecto usa [versionado semántico](https://semver.org/lang/es/) para el código.
Los artefactos de datos (`p<N>`, `v<N>`, `f<N>`, `r<N>`) tienen su propio versionado,
descrito en el README.

## [1.0.0] — 2026-09-25

Primera versión con la estructura orientada a objetos, el versionado de artefactos y las
pruebas de equivalencia numérica. Entró a `main` con el PR #6
(`feature/refactor-and-cleaning`).

### Versiones oficiales de datos

| Artefacto | Contenido | Origen |
|---|---|---|
| `p1` | Parámetros ajustados (`shape_params.json`) y la copia del panel sobre el que se ajustaron | `scenarios params fit --n 100` |
| `v1` | Escenarios por régimen: `optimization` (30), `validation` (100), `capacity` (100), `expected`, `annual_expected` | `scenarios generate --params p1` |
| `f1` | Tabla de capacidad por satélite: niveles, costos de instalación y de operación | `optimize capacity analyze --scenarios v1` |

- El panel reconstruido desde la demanda cruda es idéntico byte a byte al panel anterior.
- Los valores de `p1` son idénticos a los de los parámetros v3 anteriores.
- Los sets `optimization`, `validation`, `expected` y `annual_expected` de `v1` son idénticos byte a byte a los escenarios v3 anteriores. La única diferencia es que los ids ya no llevan el prefijo de versión.
- Todo lo anterior al versionado (v2, v3, vtest, vcompare y los resultados) quedó archivado en `data/_archive/` y `results/_archive/`, que no se versionan.
- Todavía no hay una corrida oficial de optimización: `r1` está pendiente.

### Agregado

- **Arquitectura en seis paquetes** con dependencias en un solo sentido (lo verifica `tests/test_architecture.py`):
  - `tools`: JSON, hashing, rutas, logging, artefactos, manifests, validación, promoción y CLI.
  - `core`: constantes, entidades, lectores de raws, grilla y contrato en disco.
  - `scenarios`: panel → parámetros → escenarios.
  - `optimization`: CA → modelos → experimentos, y la capacidad por satélite.
  - `calibration`: elección del tamaño de muestra SAA (solo esqueleto).
  - `visualization`: toolkit HTML/Plotly y un renderer por reporte.
- **Versionado de artefactos por etapa:** parámetros `p<N>`, escenarios `v<N>`, capacidad `f<N>` y corridas `r<N>`.
  - Todo comando escribe una candidata descartable (`cp-`, `cv-`, `cf-`, `cr-`, `cc-`).
  - Solo `validate` + `promote` crean una versión oficial, que después no cambia.
  - `manifest.json` registra todo lo necesario para reproducir el artefacto: comando y configuración, sha256 de los raws, artefactos padre, semillas, commit de git y versiones de librerías.
  - `promote` exige que la candidata haya pasado la validación y no haya cambiado desde entonces, que sus padres sean oficiales y que el árbol git esté limpio en el mismo HEAD en que se generó.
  - En parámetros, escenarios y capacidad, `promote` además re-ejecuta el comando y exige bytes idénticos. Las corridas de Gurobi no se re-ejecutan: una hoja con la misma clave ya resuelta en una corrida oficial se copia en lugar de resolverse de nuevo.
- **CLIs:** uno por paquete, `scenarios`, `optimize` y `calibrate`. También se ejecutan como `python -m src.<paquete>`.
- **Generación de escenarios:**
  - `DependenceStrategy`, con una clase por modelo de dependencia: `SpatialJoint`, `Independent`, `HistoricalBootstrap`, `MeanShocks` y `MedianShocks`.
  - `ScenarioSetWriter` y `ShapeParams`, que envuelve los parámetros persistidos.
  - `ParamsFitter` y `RegimeCalibrator` para el ajuste y la calibración de regímenes.
  - Comparación exploratoria de los modelos de dependencia con `scenarios compare`.
- **Set de escenarios `capacity`:** 100 escenarios por régimen con semillas propias (código 60), que se usan solo para dimensionar satélites. Así el set `validation` sigue fuera de muestra.
- **Análisis de capacidad por satélite** (`optimization/capacity`):
  - Cada píxel se asigna al satélite más cercano.
  - La CA da la flota por período, y se toma la flota pico del año, con los tres regímenes juntos.
  - Los niveles salen de un método configurable (`percentiles-a`, que es el predeterminado, `percentiles-b` o `fixed-grid`).
  - Los costos salen de las tarifas versionadas en `data/raw_facility/tariffs.json`: `OPEX(q) × [0.70 + 0.30 × estacionalidad del satélite]`.
  - La validación exige que el nivel máximo cubra al menos el 95% de los picos y que los costos sean crecientes.
- **Familia de modelos como jerarquía de clases:** `Uncapacitated ⊂ Capacitated ⊂ Flex`.
  - Cada variante declara solo sus bloques (`BLOCKS`), sus flags (`Features`) y qué variables usa cada bloque (`USES`), y queda registrada en `MODELS` por su `NAME`.
  - El modelo se elige con `--model`.
  - `CapacitatedSAAModel` es nuevo y porta el modelo capacitado de `OLD/`.
  - Las políticas de operación son clases: `FixedOperation`, `OnOffInstalled` y `UpToInstalled`.
  - Fijar la instalación para una evaluación es un bloque propio (`fix_installation`).
  - El objetivo, `scenario_costs()` y `decisions()` se calculan desde las mismas expresiones por escenario.
- **Optimización:**
  - `InstanceSpec` e `InstanceBuilder`: el modelo de ruteo es reemplazable y la tabla de capacidad se puede inyectar.
  - `ExperimentRunner`, `ResultStore` y `optimization/metrics`.
  - Reportes separados en productores (métricas) y renderers (`visualization`).
- **Pruebas:**
  - Goldens de equivalencia G1–G9: ajuste, recalibración, generación, comparación, espacial, reportes, CA, solves deterministas y capacidad.
  - Pruebas de ciclo de vida de artefactos, de arquitectura y de la familia de modelos.
  - Se corren con `python -m unittest discover -s tests -t .`.

### Cambiado

- `src/pipeline` pasa a `src/scenarios`, con los subpaquetes `fitting/`, `generation/` y `validation/`.
- Los ids de escenario ya no llevan la versión (`normal-optimization-001`).
- `shape_params.json` ya no incluye `generated_on`: esa fecha queda en el manifest.
- El panel en caché se movió a `data/interim/panel_monthly.csv` (con `panel_source.json`).
- `evaluation.json` guarda `source_run`, `source_result` (ruta relativa a esa corrida) y `source_solve`, en lugar de una ruta absoluta.
- Los modelos con capacidad (`capacitated`, `flex`) leen niveles y costos solo de una tabla `f<N>` (`--facilities`); sin ella, fallan. `input_facilities.xlsx` solo aporta ubicación y costo de sourcing.
- En `disabled_blocks`, el bloque de objetivo del costo de operación se llama `operation_cost` (`operation` son las variables Z). Apagar variables que otro bloque activo usa falla con un mensaje explícito.
- Python `^3.12`; `package-mode = true`, con los scripts `scenarios`, `optimize` y `calibrate`; pytest en el grupo dev.

### Corregido

- **Costos de `annual_expected`:** sus componentes no se escalaban por `horizon_weight` (12), así que sumaban 1/12 del valor real y las barras de costo del reporte de flexibilidad estaban mal en ese caso. Las corridas v3 archivadas conservan el error.
- **Semillas de calibración:** `fit` calibraba los multiplicadores de régimen con semillas que ningún set usa y con el generador sin redondear. Ahora usa los streams de `validation` y el generador de los parámetros persistidos, en una sola pasada.
- **`scenario_costs()`:** ignoraba `horizon_weight`. No cambiaba ningún número actual, pero ahora la evaluación verifica que los costos por escenario reproduzcan el objetivo.
- **`pixel_neighbor_pairs(ring>=3)`:** devolvía en silencio el primer anillo.
- **`run_one` sin costo de operación:** fallaba con modelos que no tienen ese término, como el capacitado.

### Eliminado

- Los 14 módulos CLI sueltos de `src/pipeline/cli/` y `src/optimization/cli/`, reemplazados por los CLIs por paquete.
- Código muerto de los reportes: `_pixel_grid_subplot`, `_grid_trace`, `_grid_hover_trace` y `_cell_polygon`.

### Pendiente

- La corrida `r1`: `optimize flexibility --scenarios v1 --facilities f1`, seguida de `evaluate` y `benchmark`.
- Portar desde `OLD/` los experimentos powerset y best-per-size, los mapas de solución y el análisis de factores del CA.
- Implementar `calibration` (tamaño de muestra SAA).
- Regenerar `poetry.lock` para el nuevo `pyproject.toml`.
