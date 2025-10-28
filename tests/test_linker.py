import unittest
from potatobacon.text.sentence_split import split_sentences
from potatobacon.extract.linker import link_bypass_to_obligation

class TestLinker(unittest.TestCase):
    def test_links_nearest_obligation(self):
        text = "A. Borrower shall maintain liquidity. B. However, not required if liquidity < $300M. C. Borrower shall provide reports."
        sents = split_sentences(text)
        ob_idx, link_score, polarity = link_bypass_to_obligation(sents, 1, window=2)
        self.assertIsNotNone(ob_idx)
        self.assertGreater(link_score, 0.2)

if __name__ == "__main__":
    unittest.main()
