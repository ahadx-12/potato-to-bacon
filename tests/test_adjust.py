import unittest
from potatobacon.cale.cce_adjust import adjust_cce

class TestAdjust(unittest.TestCase):
    def test_adjust_reduces_and_may_flip(self):
        base = 0.8
        new_cce, delta = adjust_cce(base, bypass_strength=0.9, link_score=0.9, obligation_polarity=+1)
        self.assertLess(new_cce, base)
        # strong carve-out can flip sign
        self.assertTrue(new_cce <= 0)

if __name__ == "__main__":
    unittest.main()
