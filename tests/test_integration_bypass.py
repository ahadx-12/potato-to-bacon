import unittest, csv, subprocess, sys
from pathlib import Path

class TestIntegration(unittest.TestCase):
    def test_cli_rewrites_csv(self):
        out = Path("sandbox/evidence_adjusted.csv")
        if out.exists():
            out.unlink()
        cmd = [sys.executable, "tools/bypass_rewrite_evidence.py", "--in", "tests/fixtures/evidence_sample.csv", "--out", str(out)]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
        self.assertTrue(out.exists())
        with out.open() as f:
            rows = list(csv.DictReader(f))
        # First two rows have bypass effects
        r1, r2, r3 = rows
        self.assertNotEqual(float(r1["adjusted_cce"]), float(r1["cce"]))
        self.assertIn("bypass_rationale_json", r1)
        self.assertNotEqual(float(r2["adjusted_cce"]), float(r2["cce"]))
        # Third row unchanged
        self.assertAlmostEqual(float(r3["adjusted_cce"]), float(r3["cce"]), places=6)

if __name__ == "__main__":
    unittest.main()
