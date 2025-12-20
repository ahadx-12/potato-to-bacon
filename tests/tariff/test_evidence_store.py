from pathlib import Path

from potatobacon.tariff.evidence_store import EvidenceStore


def test_evidence_store_deduplicates_by_hash(tmp_path: Path):
    index_path = tmp_path / "evidence_index.jsonl"
    data_dir = tmp_path / "blobs"
    store = EvidenceStore(index_path=index_path, data_dir=data_dir)

    payload = b"contract evidence payload"
    record = store.save(payload, filename="contract.pdf", content_type="application/pdf")
    duplicate = store.save(payload, filename="duplicate.pdf", content_type="application/pdf")

    assert record.evidence_id == duplicate.evidence_id
    assert record.sha256 == duplicate.sha256
    assert record.byte_length == len(payload)
    assert (data_dir / record.evidence_id).read_bytes() == payload

    reload = EvidenceStore(index_path=index_path, data_dir=data_dir)
    fetched = reload.get(record.evidence_id)
    assert fetched is not None
    assert fetched.original_filename == record.original_filename
    assert fetched.sha256 == record.sha256

    lines = index_path.read_text().strip().splitlines()
    assert len(lines) == 1
