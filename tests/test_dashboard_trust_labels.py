"""Regression checks for dashboard trust boundaries and unavailable states."""
import inspect
import unittest

from dashboard.pages import candidate_comparison, layout_prediction, model_training, simulation_studio


class DashboardTrustLabelTests(unittest.TestCase):
    def test_dashboard_pages_do_not_load_quarantined_classifier(self):
        for module in (candidate_comparison, layout_prediction, simulation_studio):
            with self.subTest(module=module.__name__):
                self.assertNotIn('load_model("layout_verification_classifier")', inspect.getsource(module))

    def test_unavailable_and_provenance_states_are_explicit(self):
        layout_src = inspect.getsource(layout_prediction)
        candidate_src = inspect.getsource(candidate_comparison)
        sim_src = inspect.getsource(simulation_studio)
        self.assertIn("No fallback prediction is provided", layout_src)
        self.assertIn("DRC status: **Not verified.**", layout_src)
        self.assertIn("drc_status", candidate_src)
        self.assertIn("Synthetic sample data; not simulated or measured", candidate_src)
        self.assertIn("not a foundry-qualified PDK", sim_src)
        self.assertIn("not extracted parasitics", sim_src)

    def test_model_training_page_shows_policy_status_and_limits(self):
        source = inspect.getsource(model_training)
        self.assertIn("get_model_status", source)
        self.assertIn("evaluation_limitations", source)
        self.assertIn("DRC classifier is quarantined", source)


if __name__ == "__main__":
    unittest.main()
