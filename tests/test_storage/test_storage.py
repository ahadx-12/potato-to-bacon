import importlib


def test_storage_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))

    import potatobacon.storage as storage

    storage = importlib.reload(storage)

    schema_sha = storage.save_schema({"type": "object", "title": "Demo"})
    assert storage.load_schema(schema_sha)["title"] == "Demo"

    code_sha = storage.save_code("def compute():\n    return 42\n")
    saved_code = (tmp_path / "code" / f"{code_sha}.py").read_text()
    assert "return 42" in saved_code

    manifest_sha = storage.save_manifest(
        {
            "canonical": "E = 1/2*m*v**2",
            "domain": "classical",
            "schema_digest": schema_sha,
            "code_digest": code_sha,
            "checks_report": {"ok": True},
        }
    )
    manifest = storage.load_manifest(manifest_sha)
    assert manifest["code_digest"] == code_sha

    index_path = tmp_path / "index" / "manifests.jsonl"
    assert index_path.exists()
    index_contents = index_path.read_text().strip().splitlines()
    assert any(manifest_sha in line for line in index_contents)
