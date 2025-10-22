from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from potatobacon.cale.runtime import bootstrap
from potatobacon.cale.train import CALETrainer, LegalConflictDataset


def test_trainer_runs_and_saves(tmp_path) -> None:
    services = bootstrap()
    dataset = LegalConflictDataset(
        csv_path=Path("data/cale/expert_labels.csv"),
        corpus=services.corpus,
        symbolic=services.symbolic,
    )
    trainer = CALETrainer(len(services.corpus[0].feature_vector))
    initial = trainer.weights.detach().clone()

    save_path = tmp_path / "cale_weights.pt"
    history = trainer.train(
        dataset,
        symbolic=services.symbolic,
        corpus=services.corpus,
        num_epochs=1,
        use_ssl=True,
        use_graph=True,
        save_path=save_path,
    )

    assert save_path.exists()
    assert not torch.allclose(initial, trainer.weights.detach())
    assert set(history) == {"supervised", "ssl", "graph"}
    assert len(history["supervised"]) == 1
