import importlib


def test_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))

    import potatobacon.storage as storage_mod
    import potatobacon.manifest.store as store_mod

    storage_mod = importlib.reload(storage_mod)
    store_mod = importlib.reload(store_mod)

    man = store_mod.ComputationManifest(
        version="1.0",
        canonical="E=1/2 m v^2",
        domain="classical",
        units={"m": "kg", "v": "m/s", "E": "J"},
        constraints={"m": {"positive": True}},
        checks_report={"ok": True},
        schema_digest="deadbeef",
        code_digest="cafebabe",
    )

    code_digest = store_mod.persist_code("def compute():\n    return 0.0\n")
    manifest_hash = store_mod.persist_manifest(man)

    assert code_digest
    loaded = store_mod.load_manifest(manifest_hash)
    assert loaded["canonical"] == man.canonical
    assert loaded["schema_digest"] == man.schema_digest
