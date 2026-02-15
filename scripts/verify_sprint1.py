
import sys
import json
import unittest
from pathlib import Path

# Add src to path
SRC_PATH = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

class Sprint1Verification(unittest.TestCase):
    def test_01_models_match_level(self):
        """Verify TariffOverlayResultModel has match_level field."""
        try:
            from potatobacon.tariff.models import TariffOverlayResultModel
            m = TariffOverlayResultModel(
                overlay_name="test", 
                applies=True, 
                additional_rate=0, 
                reason="r", 
                match_level="exact_8digit"
            )
            self.assertEqual(m.match_level, "exact_8digit")
            print("\n[PASS] TariffOverlayResultModel accepts match_level")
        except Exception as e:
            self.fail(f"TariffOverlayResultModel failed: {e}")

    def test_02_adcvd_data(self):
        """Verify AD/CVD full orders file exists and is populated."""
        path = Path(__file__).parents[1] / "data" / "overlays" / "adcvd_orders_full.json"
        if not path.exists():
            self.fail(f"Missing {path}")
        
        data = json.loads(path.read_text(encoding="utf-8"))
        count = len(data.get("orders", []))
        print(f"\n[PASS] adcvd_orders_full.json loaded: {count} orders")
        self.assertGreater(count, 50, "Should have > 50 orders")

    def test_03_section301_data(self):
        """Verify Section 301 subheading file exists and is populated."""
        path = Path(__file__).parents[1] / "data" / "overlays" / "section301_subheading.json"
        if not path.exists():
            self.fail(f"Missing {path}")
            
        data = json.loads(path.read_text(encoding="utf-8"))
        overlays = data.get("overlays", [])
        # Count total HTS prefixes covered
        total_prefixes = sum(len(o.get("hts_prefixes", [])) for o in overlays)
        print(f"\n[PASS] section301_subheading.json loaded: {len(overlays)} lists covering {total_prefixes} HTS codes")
        self.assertGreater(total_prefixes, 100, "Should have > 100 HTS codes covered")

    def test_04_duty_calculator_specific(self):
        """Verify specific duty calculation."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        # Verify params are accepted
        res = compute_total_duty(
            base_rate=0.0,
            hts_code="0000.00.00",
            origin_country="CN",
            weight_kg=100.0
        )
        print("\n[PASS] compute_total_duty accepts weight_kg")
        self.assertIsNotNone(res)

    def test_05_adcvd_registry_keywords(self):
        """Verify ADCVDRegistry logic."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry, ADCVDOrder
        # Check scope_keywords field
        order = ADCVDOrder(
            order_id="test", order_type="AD", product_description="desc",
            hts_prefixes=("1234",), origin_countries=("CN",),
            duty_rate_pct=10.0, effective_date="2020-01-01",
            status="active", case_number="A-123",
            federal_register_citation="FR",
            scope_keywords=("keyword1",)
        )
        self.assertIn("keyword1", order.scope_keywords)
        print("\n[PASS] ADCVDOrder has scope_keywords")

    def test_06_overlays_logic(self):
        """Verify Overlay logic."""
        from potatobacon.tariff.overlays import _OverlayRule
        rule = _OverlayRule(
            overlay_name="test", hts_prefixes=("1234",),
            additional_rate=0.1, reason="r",
            match_level="exact_8digit"
        )
        self.assertEqual(rule.match_level, "exact_8digit")
        print("\n[PASS] _OverlayRule has match_level")
        
    def test_07_fetch_script_runnable(self):
        """Verify fetch_full_hts.py can run (help only)."""
        import subprocess
        import os
        
        script_path = Path(__file__).parents[1] / "scripts" / "fetch_full_hts.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_PATH)
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            env=env
        )
        if result.returncode != 0:
            print(f"\n[FAIL] fetch_full_hts.py failed: {result.stderr}")
        self.assertEqual(result.returncode, 0)
        print("\n[PASS] fetch_full_hts.py is runnable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
