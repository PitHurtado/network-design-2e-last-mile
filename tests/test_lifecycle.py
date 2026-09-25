"""Artifact lifecycle: candidate -> validate -> promote, and the conditions that block promotion."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.scenarios.stages import FitConfig, GenerateConfig, ParamsStage, ParamsValidator, ScenarioStage, ScenarioValidator
from src.tools.artifacts import ArtifactKind, ArtifactStore, content_digest
from src.tools.manifest import Manifest
from src.tools.promotion import Promoter, PromotionError
from tests.golden.support import FIXTURES


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Path(tempfile.mkdtemp(prefix="lifecycle-"))
        cls.store = ArtifactStore.under(cls.base)
        # The tests run on a working tree that may be dirty; code provenance is tested separately.
        cls.promoter = Promoter(cls.store, require_clean_tree=False)
        with mock.patch("src.scenarios.stages.PATH_PANEL_MONTHLY", FIXTURES / "panel_monthly.csv"):
            cls.candidate = ParamsStage().fit(cls.store, FitConfig(n=5, validation_n=5))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, ignore_errors=True)

    def test_1_candidate_is_self_contained(self):
        self.assertTrue(self.candidate.id.startswith("cp-"))
        self.assertEqual(
            sorted(p.name for p in self.candidate.path.iterdir()), ["manifest.json", "panel_monthly.csv", "shape_params.json"]
        )
        manifest = Manifest.load(self.candidate.manifest_path)
        self.assertEqual(manifest.content, content_digest(self.candidate.path))
        self.assertNotIn("generated_on", json.loads((self.candidate.path / "shape_params.json").read_text()))

    def test_2_promotion_requires_validation(self):
        with self.assertRaisesRegex(PromotionError, "not been validated"):
            self.promoter.promote(self.candidate, ParamsStage().reproduce(self.store))

    def test_3_validate_and_promote_params_then_scenarios(self):
        result = ParamsValidator().run(self.candidate)
        self.assertTrue(result.passed, [c for c in result.checks if not c.passed])
        p1 = self.promoter.promote(self.candidate, ParamsStage().reproduce(self.store))
        self.assertEqual(p1.id, "p1")
        self.assertEqual(Manifest.load(p1.manifest_path).promoted_from, self.candidate.id)
        with self.assertRaisesRegex(PromotionError, "already official"):
            self.promoter.promote(p1)

        candidate = ScenarioStage().generate(self.store, GenerateConfig("p1", optimization_n=2, validation_n=3))
        self.assertEqual(Manifest.load(candidate.manifest_path).parent("params")["id"], "p1")
        validation = ScenarioValidator(self.store).run(candidate)
        self.assertTrue((candidate.reports_dir / "validation.html").exists())
        # With three validation scenarios the ±1% target check cannot pass: validation blocks promotion.
        self.assertFalse(validation.passed)
        with self.assertRaisesRegex(PromotionError, "failed validation"):
            self.promoter.promote(candidate, ScenarioStage().reproduce(self.store))

    def test_4_tampering_after_validation_blocks_promotion(self):
        candidate = ScenarioStage().generate(
            self.store, GenerateConfig("p1", regimes=("normal",), optimization_n=1, validation_n=2)
        )
        ScenarioValidator(self.store).run(candidate)
        # Pretend it passed, then change the content.
        validation = json.loads(candidate.validation_path.read_text())
        validation["passed"] = True
        candidate.validation_path.write_text(json.dumps(validation))
        (candidate.path / "normal" / "expected" / "manifest.json").write_text("{}")
        with self.assertRaisesRegex(PromotionError, "changed after it was validated"):
            self.promoter.promote(candidate, ScenarioStage().reproduce(self.store))

    def test_5_reproduction_must_be_byte_identical(self):
        candidate = ScenarioStage().generate(
            self.store, GenerateConfig("p1", regimes=("normal",), optimization_n=1, validation_n=2)
        )
        validation = ScenarioValidator(self.store).run(candidate)
        payload = json.loads(candidate.validation_path.read_text())
        payload["passed"] = True
        candidate.validation_path.write_text(json.dumps(payload))

        def drifted(artifact, manifest, out):
            ScenarioStage().reproduce(self.store)(artifact, manifest, out)
            (out / "normal" / "expected" / "manifest.json").write_text("{}")

        with self.assertRaisesRegex(PromotionError, "not byte-identical"):
            self.promoter.promote(candidate, drifted)
        promoted = self.promoter.promote(candidate, ScenarioStage().reproduce(self.store))
        self.assertEqual(promoted.id, "v1")
        self.assertEqual(content_digest(promoted.path)["sha256"], validation.content_sha256)

    def test_6_resolution(self):
        self.assertEqual(self.store.resolve("latest", ArtifactKind.PARAMS).id, "p1")
        self.assertIs(ArtifactKind.of("cv-20260101-000000"), ArtifactKind.SCENARIOS)
        with self.assertRaises(ValueError):
            self.store.resolve("p1", ArtifactKind.SCENARIOS)


if __name__ == "__main__":
    unittest.main()
