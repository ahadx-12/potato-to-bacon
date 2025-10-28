from collections import deque

from potatobacon.cale.finance import dedup


def test_duplicate_detection_recall_precision():
    cache = set()
    order = deque()
    unique_sentences = []
    for i in range(170):
        token = f"metric{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}"
        unique_sentences.append(
            f"The borrower shall maintain {token} leverage ratio of {3 + i/100:.2f} to 1.00 at all times."
        )
    near_duplicates = [
        sentence.replace("shall", "will") if idx % 2 == 0 else sentence + " This requirement remains in effect."
        for idx, sentence in enumerate(unique_sentences[:30])
    ]

    false_positives = 0
    for sentence in unique_sentences:
        if dedup.is_duplicate(sentence, cache, order):
            false_positives += 1
    unique_retained = len(unique_sentences) - false_positives
    assert unique_retained / len(unique_sentences) >= 0.99

    detected = sum(1 for sentence in near_duplicates if dedup.is_duplicate(sentence, cache, order))
    assert detected / len(near_duplicates) >= 0.95
