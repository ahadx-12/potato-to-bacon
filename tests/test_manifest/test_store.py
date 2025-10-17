from potatobacon.manifest.store import ComputationManifest, persist_manifest, load_manifest, persist_code

def test_manifest_roundtrip(tmp_path, monkeypatch):
    # Redirect artifacts root to tmp
    from potatobacon.manifest import store as S
    monkeypatch.setattr(S, "ART_ROOT", tmp_path)
    monkeypatch.setattr(S, "ART_MANIFEST", tmp_path / "manifests")
    monkeypatch.setattr(S, "ART_CODE", tmp_path / "code")

    man = ComputationManifest(
        version="1.0",
        canonical="E=1/2 m v^2",
        domain="classical",
        units={"m":"kg", "v":"m/s", "E":"J"},
        constraints={"m":{"positive": True}},
        checks_report={"ok": True},
        schema_digest="deadbeef",
        code_digest="cafebabe",
    )
    h = persist_manifest(man)
    loaded = load_manifest(h)
    assert loaded["canonical"] == man.canonical
