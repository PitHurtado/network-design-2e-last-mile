"""Main driver — Capacitated SAA with flexible operational decisions + operational costs."""

import json
import logging
from pathlib import Path
from typing import Optional

from src.constants import TypeOfFlexibility
from src.models.capacitated_flex_model import CapacitatedFlexModel
from src.utils.custom_logger import get_logger
from src.utils.instance import Instance

logger = get_logger("MainCapacitatedFlex")

# Proposal-A capacity levels per satellite
PROPOSAL_A_LEVELS: dict[str, list[int]] = {
    "ABAROA":           [6, 8, 10],
    "ACHACHICALA":      [4, 6, 8, 10],
    "COTA_COTA":        [8, 10, 12, 14],
    "LLOJETA":          [4, 6, 8],
    "LOS_PINOS":        [4, 6, 8, 10],
    "MALLASA":          [2, 4],
    "PERIFERICA":       [12, 14, 16, 18],
    "SOPOCACHI":        [14, 16, 18, 20],
    "ZONA_CEMENTERIO":  [8, 10, 12, 14],
}

# Proposed operational costs — derived from:
#   base OPEX (tariff table) × seasonal factor from raw historical data
#   formula: cost_op[q][t] = OPEX(q) × [0.70 + 0.30 × real_seasonal_factor(sat, t)]
#   level 0 = satellite available but NOT operating → cost = 0
PROPOSED_COST_OPERATION: dict[str, dict[int, list[float]]] = {
    "ABAROA": {
        0:  [0.0]*12,
        6:  [3082.49, 2971.51, 3188.67, 3128.71, 3119.0,  3228.9,  3232.73, 3235.97, 3400.63, 3537.23, 3438.41, 3687.56],
        8:  [3355.78, 3234.95, 3471.37, 3406.1,  3395.52, 3515.17, 3519.34, 3522.86, 3702.12, 3850.83, 3743.25, 4014.49],
        10: [4609.13, 4443.18, 4767.89, 4678.24, 4663.72, 4828.05, 4833.78, 4838.62, 5084.83, 5289.08, 5141.32, 5513.87],
    },
    "ACHACHICALA": {
        0:  [0.0]*12,
        4:  [2183.54, 2117.64, 2269.11, 2227.27, 2218.38, 2304.09, 2301.15, 2311.36, 2424.14, 2527.96, 2458.0,  2641.43],
        6:  [3062.77, 2970.33, 3182.78, 3124.1,  3111.64, 3231.85, 3227.72, 3242.05, 3400.24, 3545.86, 3447.73, 3705.03],
        8:  [3334.31, 3233.67, 3464.96, 3401.08, 3387.51, 3518.37, 3513.89, 3529.49, 3701.7,  3860.23, 3753.4,  4033.51],
        10: [4579.64, 4441.42, 4759.09, 4671.35, 4652.71, 4832.45, 4826.29, 4847.71, 5084.24, 5301.99, 5155.26, 5539.99],
    },
    "COTA_COTA": {
        0:  [0.0]*12,
        8:  [3345.63, 3234.42, 3468.27, 3403.75, 3391.78, 3516.67, 3516.77, 3525.96, 3701.91, 3855.32, 3748.06, 4023.47],
        10: [4595.19, 4442.45, 4763.64, 4675.01, 4658.58, 4830.11, 4830.25, 4842.87, 5084.54, 5295.24, 5147.92, 5526.19],
        12: [5421.03, 5240.83, 5619.75, 5515.2,  5495.81, 5698.16, 5698.34, 5713.22, 5998.32, 6246.89, 6073.1,  6519.35],
        14: [6246.87, 6039.22, 6475.86, 6355.38, 6333.04, 6566.22, 6566.42, 6583.57, 6912.1,  7198.54, 6998.27, 7512.51],
    },
    "LLOJETA": {
        0:  [0.0]*12,
        4:  [2188.23, 2117.92, 2270.51, 2228.39, 2220.13, 2303.39, 2302.34, 2309.89, 2424.21, 2525.93, 2455.76, 2637.31],
        6:  [3069.34, 2970.72, 3184.74, 3125.67, 3114.09, 3230.86, 3229.39, 3239.99, 3400.34, 3543.02, 3444.59, 3699.24],
        8:  [3341.46, 3234.1,  3467.1,  3402.78, 3390.18, 3517.31, 3515.7,  3527.24, 3701.8,  3857.13, 3749.98, 4027.21],
    },
    "LOS_PINOS": {
        0:  [0.0]*12,
        4:  [2194.95, 2118.34, 2272.46, 2229.93, 2222.65, 2302.41, 2304.09, 2307.86, 2424.35, 2522.99, 2452.61, 2631.36],
        6:  [3078.76, 2971.31, 3187.49, 3127.83, 3117.62, 3229.49, 3231.85, 3237.15, 3400.53, 3538.89, 3440.18, 3690.9],
        8:  [3351.72, 3234.74, 3470.09, 3405.14, 3394.02, 3515.81, 3518.37, 3524.14, 3702.02, 3852.65, 3745.17, 4018.13],
        10: [4603.56, 4442.89, 4766.13, 4676.92, 4661.66, 4828.93, 4832.45, 4840.38, 5084.68, 5291.57, 5143.96, 5518.86],
    },
    "MALLASA": {
        0:  [0.0]*12,
        2:  [1376.58, 1328.92, 1425.52, 1398.88, 1394.22, 1444.48, 1445.36, 1447.99, 1520.93, 1583.0,  1538.89, 1651.2],
        4:  [2194.25, 2118.27, 2272.25, 2229.79, 2222.37, 2302.48, 2303.88, 2308.07, 2424.35, 2523.27, 2452.96, 2631.99],
    },
    "PERIFERICA": {
        0:  [0.0]*12,
        12: [5414.63, 5240.31, 5618.02, 5513.64, 5493.39, 5699.2,  5696.78, 5715.3,  5998.15, 6249.66, 6076.04, 6525.06],
        14: [6239.49, 6038.62, 6473.87, 6353.58, 6330.25, 6567.42, 6564.62, 6585.97, 6911.9,  7201.73, 7001.66, 7519.09],
        16: [7064.35, 6836.93, 7329.71, 7193.53, 7167.11, 7435.63, 7432.47, 7456.63, 7825.66, 8153.8,  7927.29, 8513.11],
        18: [7889.21, 7635.24, 8185.56, 8033.48, 8003.97, 8303.85, 8300.32, 8327.3,  8739.41, 9105.87, 8852.91, 9507.14],
    },
    "SOPOCACHI": {
        0:  [0.0]*12,
        14: [6272.6,  6040.62, 6483.44, 6361.36, 6342.61, 6562.63, 6573.0,  6575.79, 6912.5,  7187.17, 6986.1,  7489.77],
        16: [7101.84, 6839.19, 7340.55, 7202.34, 7181.11, 7430.21, 7441.95, 7445.12, 7826.33, 8137.32, 7909.67, 8479.92],
        18: [7931.08, 7637.76, 8197.67, 8043.31, 8019.61, 8297.79, 8310.91, 8314.44, 8740.17, 9087.46, 8833.23, 9470.07],
        20: [8760.32, 8436.33, 9054.78, 8884.29, 8858.1,  9165.37, 9179.86, 9183.76, 9654.0,  10037.61,9756.8,  10460.21],
    },
    "ZONA_CEMENTERIO": {
        0:  [0.0]*12,
        8:  [3354.71, 3234.85, 3471.05, 3405.88, 3395.2,  3515.38, 3519.12, 3523.18, 3702.12, 3851.36, 3743.79, 4015.45],
        10: [4607.66, 4443.03, 4767.45, 4677.95, 4663.28, 4828.35, 4833.48, 4839.06, 5084.83, 5289.81, 5142.06, 5515.19],
        12: [5435.74, 5241.53, 5624.25, 5518.66, 5501.35, 5696.09, 5702.14, 5708.72, 5998.67, 6240.49, 6066.17, 6506.37],
        14: [6263.82, 6040.02, 6481.05, 6359.37, 6339.42, 6563.83, 6570.81, 6578.39, 6912.5,  7191.16, 6990.29, 7497.55],
    },
}


class MainCapacitatedFlex:
    """Solve Capacitated SAA with flex operational decisions."""

    def __init__(
        self,
        id_instance: str,
        folder_path: Path,
        configuration: tuple,
        max_run_time: int,
        is_evaluation: bool = False,
        id_sampling: Optional[int] = None,
        use_euclidean_distance: bool = False,
        facilities_subset: Optional[list] = None,
    ):
        self.folder_path  = folder_path
        self.configuration = configuration
        self.max_run_time  = max_run_time
        N, is_cont_x = configuration

        self.instance = Instance(
            id_instance=id_instance,
            is_continuous_var_x=is_cont_x,
            type_of_flexibility=TypeOfFlexibility.FLEX_CAPACITY.value,
            periods=12,
            N=N,
            is_evaluation=is_evaluation,
            id_sampling=id_sampling,
            use_euclidean_distance=use_euclidean_distance,
            facilities_subset=facilities_subset,
        )

    def solve(self) -> Path:
        label = self.instance.id_instance
        logger.info(f"Solving flex-operational for: {label}")

        solver = CapacitatedFlexModel(
            instance=self.instance,
            capacity_levels=PROPOSAL_A_LEVELS,
            cost_operation=PROPOSED_COST_OPERATION,
        )
        solver.build()
        solver.set_params({"TimeLimit": self.max_run_time})

        logger.disabled = True
        logging.disable(logging.CRITICAL)
        metrics = solver.solve()
        logger.disabled = False
        logging.disable(logging.NOTSET)

        results = self._build_results(solver, metrics)
        N, is_cont_x = self.configuration
        out = self.folder_path / f"capflex_{label}_N{N}.json"
        out.write_text(json.dumps(results, indent=4))
        logger.info(f"Saved → {out.name}")
        return out

    def _build_results(self, solver: CapacitatedFlexModel, metrics: dict) -> dict:
        N, is_cont_x = self.configuration
        cap = {}
        for i in self.instance.facilities:
            for q, (veh, _) in solver._cap[i].items():
                if round(solver.vars.Y[(i, q)].X) == 1:
                    cap[i] = {"level": q, "vehicles": veh}
                    break

        # Operational level chosen per satellite (mode across periods/scenarios)
        op_levels: dict[str, dict] = {}
        for i in self.instance.facilities:
            counts: dict[int, int] = {}
            for q in solver._cap[i]:
                for t in range(self.instance.periods):
                    for n in self.instance.scenarios:
                        if round(solver.vars.Z[(i, q, t, n)].X) == 1:
                            counts[q] = counts.get(q, 0) + 1
            op_levels[i] = counts

        return {
            "objective":                       solver.obj.cost_total.getValue(),
            "cost_installation":               solver.obj.cost_installation.getValue(),
            "cost_operation":                  solver.obj.cost_operation.getValue(),
            "cost_served_from_facilities":     solver.obj.cost_served_from_facilities.getValue(),
            "cost_served_from_dc":             solver.obj.cost_served_from_dc.getValue(),
            "scenarios":                       self.instance.scenarios_ids,
            "solver_info":                     metrics,
            "Y":                               {str(k): int(round(v.X)) for k, v in solver.model._Y.items()},
            "X":                               {str(k): round(v.X, 6) for k, v in solver.model._X.items() if v.X > 1e-6},
            "W":                               {str(k): round(v.X, 6) for k, v in solver.model._W.items() if v.X > 1e-6},
            "chosen_capacity":                 cap,
            "operational_level_counts":        op_levels,
            "pixel_info":                      self._pixel_info(),
            "facility_info":                   self._facility_info(),
            "configuration":                   self.instance.id_instance,
            "N":                               N,
            "is_continuous_x":                 is_cont_x,
            "periods":                         12,
            "max_run_time":                    self.max_run_time,
            "capacity_proposal":               "A",
            "cost_op_source":                  "historical_seasonal_70_30",
        }

    def _pixel_info(self) -> dict:
        first = next(iter(self.instance.scenarios.values()))
        return {k: {"lon": p.geo_point.lon, "lat": p.geo_point.lat} for k, p in first.pixels.items()}

    def _facility_info(self) -> dict:
        return {i: {"lon": f.geo_point.lon, "lat": f.geo_point.lat}
                for i, f in self.instance.facilities.items()}
