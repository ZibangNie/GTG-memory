import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_experiment_report import collect_results
from egoper_utils import load_json
from experiment_metrics import parse_log_dir, validate_metrics


class ExperimentMetricTests(unittest.TestCase):
    def test_parses_preserved_baseline_log(self):
        log_dir = REPO_ROOT / "reports/source_logs/EgoPER/baseline/tea"
        metrics = parse_log_dir(log_dir)
        validate_metrics(metrics, log_dir)

        self.assertEqual(metrics["tas_f1_050"], 70.1)
        self.assertEqual(metrics["ed_f1_050"], 44.2)
        self.assertEqual(metrics["omit_oiou"], 38.6)
        self.assertEqual(metrics["er_wf1_000"], 19.9)

    def test_parses_partial_erm_log_without_treating_it_as_five_task(self):
        log_dir = REPO_ROOT / "reports/source_logs/EgoPER/soft_erm_v1/tea"
        metrics = parse_log_dir(log_dir)
        validate_metrics(metrics, log_dir)

        self.assertEqual(metrics["ed_f1_050"], 9.7)
        self.assertEqual(metrics["er_wf1_000"], 8.9)

    def test_manifest_rebuilds_expected_five_task_averages(self):
        manifest = load_json(REPO_ROOT / "reports/experiments/egoper_runs.json")
        payload = collect_results(REPO_ROOT, manifest)

        self.assertEqual(payload["manifest"], "reports/experiments/egoper_runs.json")
        self.assertAlmostEqual(payload["variants"]["baseline"]["average"]["tas_f1_050"], 68.72)
        self.assertAlmostEqual(payload["variants"]["visual_memory"]["average"]["omit_oiou"], 49.32)
        self.assertEqual(payload["variants"]["soft_erm_v1"]["task_count"], 1)


if __name__ == "__main__":
    unittest.main()
