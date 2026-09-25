"""Display names and colours, one place for every report."""

REGIME_COLORS = {"low": "#3498db", "normal": "#27ae60", "high": "#e67e22"}
EMPIRICAL_COLOR = "#2c3e50"

REGIME_LABELS = {"low": "Bajo", "normal": "Normal", "high": "Alto"}
METHOD_LABELS = {
    "independent": "Baseline: muestreo independiente",
    "spatial_joint": "Enfoque paramétrico: cópula espacial",
    "historical_bootstrap": "Enfoque empírico: bootstrap histórico conjunto",
}
CASE_LABELS = {"annual_expected": "Promedio anual", "expected": "Escenario esperado", "optimization": "Optimización (30)"}
FLEX_LABELS = {
    "fixed_operation": "Operación fija",
    "on_off_installed": "Encendido / apagado",
    "up_to_installed": "Hasta capacidad",
}
SOLVE_STATUS_LABELS = {"OPTIMAL": "Óptima", "TIME_LIMIT": "Límite de tiempo"}
