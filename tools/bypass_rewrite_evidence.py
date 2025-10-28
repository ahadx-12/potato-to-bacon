#!/usr/bin/env python3
from __future__ import annotations
import csv, json, argparse
from pathlib import Path
from typing import List, Dict, Any
from potatobacon.text.sentence_split import split_sentences
from potatobacon.extract.bypass import detect_bypass
from potatobacon.extract.linker import link_bypass_to_obligation
from potatobacon.cale.cce_adjust import adjust_cce

def process_text(text: str) -> List[Dict[str, Any]]:
    sents = split_sentences(text)
    out = []
    for i, s in enumerate(sents):
        hit = detect_bypass(s)
        if not hit.is_bypass:
            continue
        ob_idx, link_score, polarity = link_bypass_to_obligation(sents, i)
        out.append({
            "bypass_sentence": s,
            "bypass_idx": i,
            "trigger": hit.trigger,
            "metrics": hit.metrics,
            "has_threshold": hit.has_threshold,
            "threshold_value": hit.threshold_value,
            "threshold_text": hit.threshold_text,
            "link_obligation_idx": ob_idx,
            "link_score": round(link_score, 3),
            "obligation_polarity": polarity,
            "bypass_strength": round(hit.strength, 3),
        })
    return out

def rewrite_csv(inp_csv: Path, out_csv: Path) -> None:
    rows = []
    with inp_csv.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            base_cce = float(r.get("cce", "0") or 0)
            text = r.get("full_text", "") or ""
            findings = process_text(text)
            # Choose the strongest linked finding (if any)
            best = None
            for fi in findings:
                if fi["link_obligation_idx"] is None:
                    continue
                if (best is None) or (fi["bypass_strength"] + fi["link_score"] > best["bypass_strength"] + best["link_score"]):
                    best = fi
            if best:
                new_cce, delta = adjust_cce(base_cce, best["bypass_strength"], best["link_score"], best["obligation_polarity"])
                r["adjusted_cce"] = f"{new_cce:.6f}"
                r["bypass_rationale_json"] = json.dumps(best, ensure_ascii=False)
                r["bypass_delta"] = f"{delta:.6f}"
            else:
                r["adjusted_cce"] = f"{base_cce:.6f}"
                r["bypass_rationale_json"] = "null"
                r["bypass_delta"] = "0.000000"
            rows.append(r)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="tests/fixtures/evidence_sample.csv")
    ap.add_argument("--out", dest="out", default="sandbox/evidence_adjusted.csv")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rewrite_csv(Path(args.inp), Path(args.out))
    print(f"[ok] wrote {args.out}")

if __name__ == "__main__":
    main()
