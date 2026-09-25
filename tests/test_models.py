"""The model family: how variants extend it, and that the capacitated variant is the right formulation."""

import os
import unittest
from dataclasses import replace

from src.optimization.models import MODELS, POLICIES, CapacitatedSAAModel, FlexSAAModel, OperationPolicy
from src.optimization.models.base import Block
from tests.golden.producers import build_instance
from tests.golden.support import Workspace

DETERMINISTIC = {"Threads": 1, "Seed": 0, "MIPGap": 0.0, "OutputFlag": 0, "TimeLimit": 600}


class ExtensionTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(sorted(MODELS), ["capacitated", "flex", "uncapacitated"])

    def test_flex_build_order_is_stable(self):
        # The order Gurobi sees; with a work or time limit it changes the result.
        order = {kind: [b.name for b in FlexSAAModel.blocks(kind)] for kind in ("var", "obj", "constr")}
        self.assertEqual(order["var"], ["assignment", "install", "operation"])
        self.assertEqual(order["obj"], ["routing_facilities", "routing_dc", "installation", "operation_cost"])
        self.assertEqual(
            order["constr"],
            ["demand", "one_install_level", "fix_installation", "one_operation_level", "operation_le_install", "capacity"],
        )

    def test_a_variant_cannot_redeclare_a_block(self):
        with self.assertRaises(TypeError):

            class Broken(CapacitatedSAAModel):  # pylint: disable=unused-variable
                BLOCKS = (Block("capacity", "constr", "_constr_capacity"),)

    def test_disabling_variables_still_in_use_is_explicit(self):
        instance = type(
            "FakeInstance",
            (),
            {
                "config": type("C", (), {"is_continuous_var_x": False, "type_of_flexibility": "up_to_installed"})(),
                "facilities": {},
            },
        )()
        features = replace(FlexSAAModel.DEFAULT_FEATURES, disabled_blocks=frozenset({"operation"}))
        with self.assertRaisesRegex(ValueError, "variables 'operation' are off but .*operation_cost"):
            FlexSAAModel(instance, features=features)
        # Dropping only the operating cost keeps Z and builds.
        FlexSAAModel(instance, features=replace(FlexSAAModel.DEFAULT_FEATURES, disabled_blocks=frozenset({"operation_cost"})))
        self.assertEqual(FlexSAAModel.REQUIREMENTS["capacity"], frozenset({"assignment", "install", "operation"}))

    def test_capacitated_models_require_a_capacity_table(self):
        from src.optimization.stages import RunInputs
        from src.tools.artifacts import ArtifactStore

        for model in ("capacitated", "flex"):
            with self.assertRaisesRegex(ValueError, "--facilities"):
                RunInputs.resolve(ArtifactStore(), "v1", None, model)

    def test_a_new_policy_registers_itself(self):
        class AlwaysOn(OperationPolicy):  # pylint: disable=unused-variable
            name = "test_always_on"

            def link(self, model, facility_id, levels, period, scenario_id):
                pass

        self.assertIn("test_always_on", POLICIES)
        del POLICIES["test_always_on"]


@unittest.skipIf(os.environ.get("GOLDEN_SKIP_SOLVE"), "GOLDEN_SKIP_SOLVE set")
class CapacitatedFormulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def _solve(self, model):
        model.set_params(DETERMINISTIC)
        model.build()
        return model.solve()

    def test_capacitated_equals_fixed_flex_without_operating_cost(self):
        """Z == Y and no operating cost is exactly the capacitated model: same optimum."""
        instance = build_instance(
            self.ws, flexibility="fixed_operation", N=1, scenario_set="expected", use_euclidean_distance=True
        )
        capacitated = self._solve(CapacitatedSAAModel(instance))
        flex_features = replace(FlexSAAModel.DEFAULT_FEATURES, disabled_blocks=frozenset({"operation_cost"}))
        flex = self._solve(FlexSAAModel(instance, features=flex_features))
        self.assertTrue(capacitated["is_optimal"] and flex["is_optimal"])
        self.assertAlmostEqual(capacitated["objective_value"], flex["objective_value"], places=3)

    def test_scenario_costs_reproduce_the_objective(self):
        instance = build_instance(self.ws, flexibility="none", N=2, scenario_set="optimization", use_euclidean_distance=True)
        model = CapacitatedSAAModel(instance)
        solve = self._solve(model)
        rows = model.scenario_costs()
        mean_total = sum(row["total_cost"] for row in rows) / len(rows)
        self.assertAlmostEqual(mean_total, solve["objective_value"], delta=0.01)
        self.assertEqual(
            set(rows[0]),
            {"scenario_id", "installation_cost", "routing_facilities_cost", "routing_dc_cost", "second_stage_cost", "total_cost"},
        )

    def test_a_capacitated_leaf_persists_with_no_operating_cost(self):
        from src.core.contract import ScenarioLayout
        from src.optimization.experiments.flexibility import ExperimentRun, run_one
        from src.optimization.experiments.runner import ExperimentRunner
        from src.optimization.experiments.store import ResultStore
        from src.optimization.instance import InstanceBuilder
        from tests.golden.producers import generated_root
        from tests.golden.support import VERSION

        runner = ExperimentRunner(InstanceBuilder(ScenarioLayout(generated_root(self.ws)), VERSION), {"Threads": 1, "Seed": 0})
        store = ResultStore(self.ws.path("results", "flexibility"), "result.json")
        payload = run_one(ExperimentRun(VERSION, "normal", "none", "expected", "capacitated"), store, runner, 600, 0.0, True)
        self.assertNotIn("status", payload, payload.get("error"))
        self.assertIsNone(payload["costs"]["operation_expected"])
        components = sum(value for value in payload["costs"].values() if value is not None)
        self.assertAlmostEqual(components, payload["solve"]["objective_value"], delta=0.01)


if __name__ == "__main__":
    unittest.main()
