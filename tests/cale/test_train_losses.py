from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from potatobacon.cale.runtime import bootstrap
from potatobacon.cale.train import CALELoss, LegalConflictDataset


def test_dataset_loads_demo_csv() -> None:
    services = bootstrap()
    dataset = LegalConflictDataset(
        csv_path=Path("data/cale/expert_labels.csv"),
        corpus=services.corpus,
        symbolic=services.symbolic,
    )
    assert len(dataset) >= 5
    rule1, rule2, label, ci = dataset[0]
    assert rule1.id
    assert rule2.id
    assert 0.0 <= label <= 1.0
    assert 0.0 <= ci <= 1.0


def test_loss_components_behave() -> None:
    loss = CALELoss()
    supervised = loss.compute_supervised_loss(torch.tensor(0.8), torch.tensor(1.0))
    assert supervised.item() > 0

    ssl = loss.compute_ssl_loss(
        torch.tensor(0.6), torch.tensor(0.4), torch.tensor(0.8)
    )
    assert ssl.item() >= 0

    l1 = loss.compute_l1_loss(torch.tensor([0.1, -0.2]), torch.tensor([0.5, 0.5]))
    assert torch.isclose(l1, torch.tensor(0.1 + 0.2 + 0.5 + 0.5))

    laplacian = torch.tensor([[1.0, -0.5], [-0.5, 1.0]])
    ccs_vec = torch.tensor([0.2, 0.4])
    graph_loss = loss.compute_graph_loss(ccs_vec, laplacian)
    assert graph_loss.item() >= 0
