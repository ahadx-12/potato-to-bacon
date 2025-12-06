# CALE-LAW REPORT: #hunt_7729_x9 — Staking-IP-Arbitrage

## 1) System Header
- Status: ONLINE
- Version: 0.2.0
- Manifest: 7bc17967bce6ed0bbcfb841b9f1f71954ca153b7f2fbd4e766ee64c5b757d668
- Job ID: cc7a8438-1e27-4f0a-94ec-933ab7179f36
- Jurisdictions: ["KY","IE","US"]
- Seed: 7729

## 2) Executive Metrics (table)
| Metric | Value | Status | Meaning |
|---|---|---|---|
| Regulatory Alpha (R_α) | 366.67 | 🟢 | >1.5=strong |
| Net Value (Simulated) | $45,833,333.33 | — | post-tax |
| Effective Tax Rate | 8.33% | — | blended |
| Contradiction Risk (P_c) | 0.0 | 🟢 | empirical |
| Entropy (H) | 0.05 | — | ambiguity |

## 3) Golden Scenario (bullet list)
- Strategy title: Cayman service CIGA → Irish ManCo trading → US FTC shield
- Cayman step: SERVICE classification via uptime/slashing CIGA; outsourcing permitted to satisfy economic substance
- Ireland step: TRADING treatment for active ManCo operations at 12.5% vs passive Case IV 25%
- US step: FTC offsets GILTI 12.6% to minimize residual inclusion

## 4) Risk Analysis
- Risk A: Cayman ES misclassification (probability low); Mitigation: maintain local MSP contract documenting uptime/slashing controls
- Risk B: Ireland passive recharacterization (probability low); Mitigation: evidence of active validation risk management and board minutes in IE

## 5) Provenance Chain (min 3)
```json
[
  {
    "step": 1,
    "jurisdiction": "KY",
    "rule_id": "KY_ES_TEST",
    "urn": "urn:law:ky:ky_es_test:s0",
    "citations": ["KY_ES_TEST"],
    "summary": "OBLIGE classify income as SERVICE if CIGA includes uptime/slashing mitigation; PROHIBIT HIGH_RISK_IP when passive; PERMIT outsourcing"
  },
  {
    "step": 2,
    "jurisdiction": "IE",
    "rule_id": "IE_TRADING_VS_PASSIVE",
    "urn": "urn:law:ie:ie_trading_vs_passive:s0",
    "citations": ["IE_TRADING_VS_PASSIVE"],
    "summary": "OBLIGE treat ManCo income as TRADING when actively managing validation ops; PROHIBIT passive receipt at Case IV; rates 12.5% vs 25%"
  },
  {
    "step": 3,
    "jurisdiction": "US",
    "rule_id": "US_GILTI_2026",
    "urn": "urn:law:us:us_gilti_2026:s0",
    "citations": ["US_GILTI_2026"],
    "summary": "OBLIGE GILTI_RATE 12.6%; PERMIT FTC; RESULT residual = max(0, GILTI-foreign_rate)"
  }
]
```

## 6) Dependency Graph (sample)
```json
{
  "nodes": [
    {"id": "KY_ES_TEST", "urn": "urn:law:ky:ky_es_test:s0", "citations": ["KY_ES_TEST"]},
    {"id": "IE_TRADING_VS_PASSIVE", "urn": "urn:law:ie:ie_trading_vs_passive:s0", "citations": ["IE_TRADING_VS_PASSIVE"]},
    {"id": "US_GILTI_2026", "urn": "urn:law:us:us_gilti_2026:s0", "citations": ["US_GILTI_2026"]}
  ],
  "edges": [
    {"from_id": "KY_ES_TEST", "to_id": "IE_TRADING_VS_PASSIVE", "relation": "sequence"},
    {"from_id": "IE_TRADING_VS_PASSIVE", "to_id": "US_GILTI_2026", "relation": "sequence"}
  ]
}
```

## 7) Score Components (JSON)
```json
{
  "value_term": 1.0,
  "entropy_term": 0.05,
  "risk_term": 0.9,
  "alpha": 1.0,
  "beta": 1.0,
  "seed": 7729
}
```

## 8) Decision
- RECOMMEND: EXECUTE
- Actionable suggestion: finalize outsourced Cayman MSP uptime/slashing SLA and document Irish ops oversight to preserve trading treatment and FTC eligibility.

## 9) Appendix
- Request payload (compact): jurisdictions KY→IE→US; gross_staking_rewards 50,000,000; cayman_substance service_provider; ireland_role manco_active; outsourced_ciga true; seed 7729
- Response ids: asset af97f434-6f2b-4350-b924-e7ed402ffc9d; job cc7a8438-1e27-4f0a-94ec-933ab7179f36
- Timing: polling completed within 4s
- Notes on auto-fixes: enforced minimum entropy/risk for multi-jurisdiction runs; rebuilt rate limiter to reinitialize per test; added manifest ingest fixture and objective alias handling.
