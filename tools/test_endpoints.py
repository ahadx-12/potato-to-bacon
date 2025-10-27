import os, json, requests

BASE = os.environ.get("API_BASE","http://127.0.0.1:8000")

payload = {
  "rule1": {"text":"An organization MUST obtain consent before collecting personal data.","jurisdiction":"Canada","statute":"PIPEDA","section":"7(3)","enactment_year":2000},
  "rule2": {"text":"A government agency MAY collect personal data without consent in cases of national security.","jurisdiction":"Canada","statute":"Anti-Terrorism Act","section":"83.28","enactment_year":2001}
}
r = requests.post(f"{BASE}/v1/law/analyze", json=payload, timeout=30)
print("Status:", r.status_code)
print("Body keys:", list(r.json().keys()))
