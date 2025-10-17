from fastapi.testclient import TestClient
from potatobacon.api.app import app

client = TestClient(app)

def test_health():
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_translate_validate_codegen_manifest():
    dsl = "@equation(domain='classical')\n" \
          "def kinetic_energy(m: Mass(kg), v: Speed(m_per_s)) -> Energy(J):\n" \
          "    return 0.5*m*v**2\n"

    r = client.post("/v1/translate", json={"dsl": dsl})
    assert r.status_code == 200
    assert r.json()["success"] is True

    r = client.post("/v1/validate", json={"dsl": dsl, "domain":"classical",
                                          "units":{"m":"kg","v":"m/s","E":"J"}})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = client.post("/v1/codegen", json={"dsl": dsl, "name":"ke"})
    assert r.status_code == 200
    assert "def ke(" in r.json()["code"]

    r = client.post("/v1/manifest", json={"dsl": dsl, "domain":"classical",
                                          "units":{"m":"kg","v":"m/s","E":"J"}})
    assert r.status_code == 200
    assert "manifest_hash" in r.json()
