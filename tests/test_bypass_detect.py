import unittest
from potatobacon.extract.bypass import detect_bypass

class TestBypassDetect(unittest.TestCase):
    def test_detects_unless_threshold(self):
        s = "… shall not be required if the Company's liquidity shall fall below $300M."
        hit = detect_bypass(s)
        self.assertTrue(hit.is_bypass)
        self.assertTrue(hit.has_threshold)
        self.assertIn("liquidity", " ".join(hit.metrics))
        self.assertGreater(hit.strength, 0.5)

    def test_no_bypass(self):
        s = "Borrower shall deliver monthly statements."
        hit = detect_bypass(s)
        self.assertFalse(hit.is_bypass)

if __name__ == "__main__":
    unittest.main()
