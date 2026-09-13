import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase0Foundation(unittest.TestCase):
    """Sanity test suite for Phase 0 directory structure and package imports."""

    def test_project_root_directories_exist(self):
        """Verify essential project directories are present."""
        expected_dirs = [
            "backend",
            "backend/app",
            "backend/app/api",
            "backend/app/core",
            "backend/app/data_processing",
            "backend/app/forecasting",
            "backend/app/explainability",
            "backend/app/decision_intelligence",
            "backend/app/agents",
            "backend/app/orchestration",
            "frontend",
            "data",
            "data/raw",
            "data/processed",
            "models",
            "tests",
            "tests/unit",
            "tests/integration",
            "docs",
        ]
        for dir_path in expected_dirs:
            full_path = PROJECT_ROOT / dir_path
            self.assertTrue(full_path.exists(), f"Expected directory missing: {dir_path}")
            self.assertTrue(full_path.is_dir(), f"Expected path is not a directory: {dir_path}")

    def test_backend_package_import(self):
        """Verify backend packages are importable."""
        import backend.app
        from backend.app.core.config import settings

        self.assertEqual(backend.app.__version__, "0.1.0-alpha")
        self.assertEqual(settings.app_name, "GlassBox-BI")
        self.assertEqual(settings.backend_port, 8000)

    def test_modular_package_boundaries(self):
        """Verify modular domain packages are importable without circular dependencies."""
        import backend.app.data_processing
        import backend.app.forecasting
        import backend.app.explainability
        import backend.app.decision_intelligence
        import backend.app.agents
        import backend.app.orchestration

        self.assertIsNotNone(backend.app.data_processing)
        self.assertIsNotNone(backend.app.forecasting)
        self.assertIsNotNone(backend.app.explainability)
        self.assertIsNotNone(backend.app.decision_intelligence)
        self.assertIsNotNone(backend.app.agents)
        self.assertIsNotNone(backend.app.orchestration)


if __name__ == "__main__":
    unittest.main()

