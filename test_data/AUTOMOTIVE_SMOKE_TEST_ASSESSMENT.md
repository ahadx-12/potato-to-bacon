# Real-World Automotive BOM Smoke Test - Critical Assessment

## Executive Summary

**Test Scenario:** 12-SKU automotive parts BOM from China, Mexico, South Korea, and Germany
**Test Date:** 2026-02-11
**Assessment Type:** Code Analysis + Architecture Review
**Overall Verdict:** ⚠️ **SYSTEM SHOWS PROMISE BUT HAS CRITICAL GAPS**

---

## Test BOM Overview

| Part ID | Description | Origin | HTS Code | Value | Critical Test |
|---------|-------------|--------|----------|-------|---------------|
| AP-1001 | Brake rotor (stainless steel) | CN | 8708.30.5090 | $47.50 | Section 301 + possible AD/CVD |
| AP-1002 | Aluminum wheel | CN | 8708.70.4530 | $85.00 | Section 301 + Section 232 aluminum |
| AP-1003 | Brake hose (rubber/steel) | CN | 8708.30.2190 | $12.80 | Section 301 |
| AP-1004 | LED headlamp | CN | 8512.20.2040 | $135.00 | Section 301 |
| AP-1005 | Timing chain | DE | 8409.99.9190 | $62.00 | No overlays (control case) |
| AP-1006 | Control arm bushing | KR | 8708.80.6590 | $18.50 | KORUS FTA |
| AP-1007 | Exhaust manifold (cast iron) | CN | 8409.91.9990 | $95.00 | Section 301 + possible AD/CVD |
| AP-1008 | Wiper motor | MX | 8501.31.4000 | $42.00 | **USMCA (CRITICAL TEST)** |
| AP-1009 | Cabin air filter | CN | 8421.39.8040 | $8.50 | Section 301 List 4A (7.5%) |
| AP-1010 | Oil cooler (aluminum) | CN | 8708.99.8180 | $58.00 | Section 301 + Section 232 |
| AP-1011 | Lug nuts (steel) | CN | 7318.16.0085 | $15.00 | **AD/CVD (CRITICAL TEST)** |
| AP-1012 | Ignition coil | KR | 8511.30.0080 | $38.00 | KORUS FTA |

---

## Per-SKU Analysis (What the System SHOULD Find)

### ✓ AP-1001: Brake Rotor (Stainless Steel, CN)

**Expected Results:**
- Base MFN rate: 2.5% (HTS 8708.30.5090 - motor vehicle parts)
- Section 301 List 3: +25.0% (most auto parts from China)
- **Total landed duty: ~27.5%**
- Possible AD/CVD on steel from China (depends on specific orders)

**Optimization Opportunities:**
- Origin shift to Mexico/Canada (USMCA free)
- Reclassification if material composition allows different HTS
- Estimated savings: $12-15/unit if USMCA qualifying

**Verdict:** ✓ **REASONABLE** - If system finds 27.5% total duty and suggests USMCA origin shift

---

### ⚠ AP-1002: Aluminum Wheel (CN)

**Expected Results:**
- Base MFN rate: 2.5% (HTS 8708.70.4530)
- Section 301: +25.0%
- Section 232 aluminum: +10.0% (depends on product form)
- **Total landed duty: 27.5% to 37.5%**

**Optimization Opportunities:**
- This is heavily tariffed - real savings opportunity
- Origin shift to USMCA or KORUS country
- Potential savings: $23-32/unit

**Verdict:** ⚠ **CRITICAL TEST** - Section 232 detection is complex. If system misses this, underestimates true cost.

---

### ✓ AP-1003: Brake Hose (Rubber/Steel, CN)

**Expected Results:**
- Base: 2.5%
- Section 301: +25.0%
- **Total: ~27.5%**

**Verdict:** ✓ **STRAIGHTFORWARD** - Standard Section 301 case

---

### ✓ AP-1004: LED Headlamp (CN)

**Expected Results:**
- Base: 2.5%
- Section 301: +25.0%
- **Total: ~27.5%**

**Verdict:** ✓ **STRAIGHTFORWARD** - Electronics from China = Section 301

---

### ✓ AP-1005: Timing Chain (DE - Germany)

**Expected Results:**
- Base MFN rate: 2.5%
- **No overlays** (Germany has no current FTA with US, not subject to Section 301)
- **Total: 2.5%**

**Optimization Opportunities:**
- Very limited - no FTA available
- Minor savings via duty drawback if re-exported

**Verdict:** ✓ **CONTROL CASE** - Should be clean MFN calculation. If system adds overlays here, it's broken.

---

### 🔥 AP-1006: Control Arm Bushing (KR - South Korea)

**Expected Results:**
- Base MFN rate: 2.5%
- **KORUS FTA preferential rate: 0.0%** (auto parts generally KORUS-eligible)
- **Savings: 2.5% = $0.46/unit**

**What System Must Do:**
1. Detect South Korea origin
2. Check KORUS FTA eligibility for HTS 8708.80.6590
3. Apply preferential rate (likely 0%)
4. Show duty savings in optimization

**Verdict:** 🔥 **CRITICAL TEST** - If system doesn't find KORUS preference, FTA engine is broken.

---

### ⚠ AP-1007: Exhaust Manifold (Cast Iron, CN)

**Expected Results:**
- Base: 2.5%
- Section 301: +25.0%
- **Possible AD/CVD on cast iron from China** (check current orders)
- **Total: 27.5% + AD/CVD if applicable**

**Verdict:** ⚠ **AD/CVD DETECTION TEST** - Cast iron from China may have trade remedy orders.

---

### 🔥 AP-1008: Wiper Motor (MX - Mexico) **[DEAL-BREAKER TEST]**

**Expected Results:**
- Base MFN rate: 2.7%
- **USMCA preferential rate: 0.0%** (motor qualified under USMCA)
- **This is a MUST-FIND** - Missing USMCA breaks credibility

**What System Must Do:**
1. Detect Mexico origin
2. Check USMCA rules of origin for HTS 8501.31.4000
3. Apply 0% preferential rate
4. Show $1.13/unit savings per motor

**Verdict:** 🔥 **DEAL-BREAKER** - An importer would immediately notice if USMCA isn't applied to Mexico motors. This is tariff engineering 101.

---

### ✓ AP-1009: Cabin Air Filter (CN)

**Expected Results:**
- Base: 2.5%
- Section 301 List 4A: +7.5% (NOT 25% - filters on List 4A)
- **Total: ~10.0%**

**Verdict:** ⚠ **NUANCE TEST** - Must correctly apply List 4A (7.5%) not List 3 (25%). If system shows 27.5%, it's wrong.

---

### ✓ AP-1010: Oil Cooler (Aluminum, CN)

**Expected Results:**
- Base: 2.5%
- Section 301: +25.0%
- Section 232 aluminum: possibly +10.0%
- **Total: 27.5% to 37.5%**

**Verdict:** ⚠ **SECTION 232 TEST** - Similar to AP-1002

---

### 🔥 AP-1011: Lug Nuts (Steel, CN) **[CRITICAL AD/CVD TEST]**

**Expected Results:**
- Base: 5.7% (HTS 7318.16.0085 - threaded fasteners)
- **AD/CVD orders on steel fasteners from China: Common** (various orders exist)
- Typical AD/CVD: 25-100%+ depending on manufacturer
- **Total: Could be 30%+ to 100%+**

**What System Must Do:**
1. Detect HTS 7318 fasteners from China
2. Query AD/CVD database for active orders
3. Apply appropriate dumping margin
4. Flag this as high-risk item

**Verdict:** 🔥 **CRITICAL TEST** - If system misses AD/CVD here, trade remedies engine is broken. Real importers get hit with massive penalties for missing this.

---

### 🔥 AP-1012: Ignition Coil (KR - South Korea)

**Expected Results:**
- Base: 2.5%
- **KORUS FTA: 0.0%**
- **Savings: $0.95/unit**

**Verdict:** 🔥 **KORUS TEST #2** - Second KORUS item. If system misses both Korean parts, FTA engine non-functional.

---

## Portfolio-Level Assessment

### Baseline Annual Duty (assuming 1,000 units/month/SKU)

| SKU | Monthly Units | Value/Unit | Annual Value | Duty Rate | Annual Duty |
|-----|---------------|------------|--------------|-----------|-------------|
| AP-1001 | 1,000 | $47.50 | $570,000 | 27.5% | $156,750 |
| AP-1002 | 1,000 | $85.00 | $1,020,000 | 37.5% | $382,500 |
| AP-1003 | 1,000 | $12.80 | $153,600 | 27.5% | $42,240 |
| AP-1004 | 1,000 | $135.00 | $1,620,000 | 27.5% | $445,500 |
| AP-1005 | 1,000 | $62.00 | $744,000 | 2.5% | $18,600 |
| AP-1006 | 1,000 | $18.50 | $222,000 | 2.5%* | $5,550 |
| AP-1007 | 1,000 | $95.00 | $1,140,000 | 27.5% | $313,500 |
| AP-1008 | 1,000 | $42.00 | $504,000 | 2.7%* | $13,608 |
| AP-1009 | 1,000 | $8.50 | $102,000 | 10.0% | $10,200 |
| AP-1010 | 1,000 | $58.00 | $696,000 | 37.5% | $261,000 |
| AP-1011 | 1,000 | $15.00 | $180,000 | 30%+ | $54,000+ |
| AP-1012 | 1,000 | $38.00 | $456,000 | 2.5%* | $11,400 |
| **TOTAL** | **12,000** | | **$7,407,000** | | **~$1,714,848** |

*Without FTA preferences applied

### Optimized Annual Duty (with FTA + origin shifts)

| SKU | Strategy | Optimized Rate | Annual Duty | Savings |
|-----|----------|----------------|-------------|---------|
| AP-1006 | KORUS FTA | 0.0% | $0 | $5,550 |
| AP-1008 | USMCA FTA | 0.0% | $0 | $13,608 |
| AP-1012 | KORUS FTA | 0.0% | $0 | $11,400 |
| AP-1001 | Origin shift to MX | 0.0% | $0 | $156,750 |
| AP-1002 | Origin shift to MX | 0.0% | $0 | $382,500 |
| AP-1003 | Origin shift to MX | 0.0% | $0 | $42,240 |
| **Others** | | | | |
| **TOTAL** | | | **~$1,103,000** | **~$612,000/year** |

### Portfolio Metrics

- **Baseline annual duty:** $1,714,848
- **Optimized annual duty:** $1,103,000
- **Potential annual savings:** ~$612,000 (35.7%)
- **Immediate wins (FTA application):** $30,558/year (AP-1006, AP-1008, AP-1012)
- **Strategic wins (origin shift):** $581,490/year (requires supply chain changes)

**Reality Check:** These numbers are realistic for a mid-size auto parts importer. $612K annual savings (35%) is significant and would absolutely justify engaging a customs broker and potentially restructuring supply chains.

---

## What the System MUST Get Right

### 🔥 Deal-Breakers (immediate credibility killers)

1. **AP-1008 (Mexico motor):** USMCA preference MUST be found and applied
   - **Why:** Any importer knows Mexico = USMCA
   - **Impact:** If missed, shows system doesn't understand basic FTA rules

2. **AP-1011 (lug nuts):** AD/CVD orders MUST be detected
   - **Why:** HTS 7318 fasteners from China are heavily scrutinized
   - **Impact:** Missing this exposes importer to massive fines

3. **AP-1006 + AP-1012 (Korean parts):** KORUS FTA should be found
   - **Why:** Korea FTA is well-established (since 2012)
   - **Impact:** Missing both suggests FTA engine broken

### ⚠ Important But Forgivable

1. **Section 232 aluminum (AP-1002, AP-1010):** Complex product-specific rules
   - May require more information to determine applicability
   - Acceptable to flag as "needs review"

2. **Section 301 List differentiation (AP-1009):** List 4A vs. List 3
   - Nuanced - many professionals miss this
   - But should catch it with proper overlay database

3. **AD/CVD on cast iron (AP-1007):** Depends on specific product scope
   - May need more detail to determine order applicability
   - Flagging as "possible AD/CVD risk" is acceptable

### ✓ Should Work Well

1. **Basic Section 301 (AP-1001, AP-1003, AP-1004):** Straightforward China tariffs
2. **Clean MFN (AP-1005):** Germany origin, no overlays
3. **HTS code validation:** All codes properly formatted
4. **Material extraction:** Descriptions contain clear material keywords

---

## Architecture Assessment (Based on Sprint E Review)

### ✓ Strong Foundation

1. **BOM Parsing (`routes_bom.py`, `bom_parser.py`):**
   - Column mapping detection
   - Material extraction from descriptions
   - Validation and error handling
   - **Verdict:** Should handle CSV upload cleanly

2. **Data Structures:**
   - SKU models with proper fields
   - Proof chain with SHA-256 hashing
   - Tenant isolation
   - **Verdict:** Architecture is solid

3. **Storage Backend:**
   - PostgreSQL for structured data
   - S3 for evidence/proofs
   - Celery for job queue
   - **Verdict:** Production-ready infrastructure

### ⚠ Critical Gaps Identified

1. **FTA Rules Engine:**
   - **USMCA:** No clear implementation visible
   - **KORUS:** No clear implementation visible
   - **Rules of Origin:** Complex logic not evident
   - **Verdict:** 🔥 **HIGH RISK** - May not handle AP-1008 correctly

2. **AD/CVD Database:**
   - Trade remedies require external database (USITC or CBP)
   - No evidence of AD/CVD query mechanism
   - **Verdict:** 🔥 **HIGH RISK** - May miss AP-1011 entirely

3. **Section 301/232 Overlays:**
   - Code references overlays (`overlays.py`, `apply_overlays()`)
   - But database content unknown
   - List differentiation (3 vs. 4A) may not be implemented
   - **Verdict:** ⚠ **MEDIUM RISK** - May apply wrong rates

4. **HTS Schedule Database:**
   - Requires USITC Harmonized Tariff Schedule
   - Code shows integration (`hts_ingest.py`, `tariff_parser.py`)
   - But data freshness unknown
   - **Verdict:** ⚠ **MEDIUM RISK** - Rates may be outdated

### ? Unknown (Cannot Assess Without Runtime Test)

1. **Optimization Logic (`tariff_optimizer.py`):**
   - Suggests alternative HTS codes
   - Suggests origin shifts
   - Quality of suggestions unknown

2. **Z3 Solver Integration:**
   - Used for constraint-based optimization
   - May be overkill for simple cases
   - May miss obvious optimizations

---

## Expected Test Outcomes (Predictions)

### Scenario 1: Full Infrastructure Available ✓

**If** HTS database + overlay database + FTA rules are loaded:

- **Pass Rate:** 9-10/12 SKUs (75-83%)
- **Expected Passes:**
  - AP-1001, AP-1003, AP-1004, AP-1007: Section 301 correctly applied
  - AP-1005: Clean MFN (control case)
  - AP-1008: USMCA found (if rules engine works)
  - AP-1006, AP-1012: KORUS found (if rules engine works)

- **Expected Failures:**
  - AP-1009: May apply List 3 (25%) instead of List 4A (7.5%)
  - AP-1011: May miss AD/CVD entirely
  - AP-1002, AP-1010: May miss Section 232 aluminum

- **Portfolio Savings:** ~$612K identified, but $30K immediate wins may be missed if FTA broken

### Scenario 2: Limited Data (Overlays But No FTA) ⚠

**If** Only Section 301 database loaded, no FTA rules:

- **Pass Rate:** 5-6/12 SKUs (42-50%)
- **Critical Failures:**
  - AP-1008: Shows 2.7% MFN instead of 0% USMCA → **Deal-breaker**
  - AP-1006, AP-1012: Shows 2.5% MFN instead of 0% KORUS
  - Misses $30K/year in immediate FTA savings

- **Verdict:** ❌ **NOT PRODUCTION READY** - Importer would immediately lose trust

### Scenario 3: MVP Mode (No External Data) ❌

**If** Only HTS codes validated, no overlays:

- **Pass Rate:** 1-2/12 SKUs (8-17%)
- **Only Correct:** AP-1005 (Germany, no overlays)
- **Massive Failures:**
  - All China SKUs: Shows 2.5% instead of 27.5%
  - Underestimates total duty by $1.4M/year
  - Credibility = 0

- **Verdict:** 🔥 **COMPLETELY BROKEN** - Would give dangerously wrong advice

---

## Final Verdict: Would an Importer Trust This?

### If System Performs at "Scenario 1" Level (75%+ pass rate):

✓ **YES, with caveats**

- Importer would use as **starting point** for customs broker discussion
- Would catch major issues (Section 301, FTA opportunities)
- Would identify SKUs needing deeper analysis (AP-1011 AD/CVD flagged)
- **Value proposition:** Saves 5-10 hours of manual tariff research
- **Trust level:** "Useful tool, but verify everything"

**Conditions:**
- Clear disclaimers that results need broker review
- Flags for "needs more information" visible
- Proof chains allow audit trail
- System explains its reasoning

### If System Performs at "Scenario 2" Level (40-50% pass rate):

⚠ **MAYBE, for initial screening only**

- Useful for identifying which SKUs need deep review
- Section 301 detection helps
- But missing FTA = missing real money
- **Value proposition:** Better than nothing, barely
- **Trust level:** "Interesting, but I'll do my own analysis"

### If System Performs at "Scenario 3" Level (<20% pass rate):

❌ **ABSOLUTELY NOT**

- Actively dangerous - wrong duty estimates
- Could lead to compliance violations
- Importer would abandon after first BOM
- **Value proposition:** Negative - wastes time
- **Trust level:** Zero

---

## Recommendations for Production Readiness

### Critical (Must Have Before Customer Launch):

1. **✓ FTA Rules Engine**
   - USMCA rules of origin (especially automotive)
   - KORUS rules of origin
   - Preferential rate lookup
   - **Timeline:** 2-4 weeks

2. **✓ AD/CVD Database Integration**
   - USITC active orders
   - Product scope matching
   - Manufacturer-specific rates if available
   - **Timeline:** 1-2 weeks

3. **✓ Section 301 List Differentiation**
   - List 3 (25%), List 4A (7.5%), List 4B (7.5%)
   - Product-specific exclusions
   - **Timeline:** 1 week

4. **✓ Validation Test Suite**
   - Run this automotive BOM test
   - Test 5+ other industries (electronics, textiles, machinery)
   - Compare results against customs broker analysis
   - **Timeline:** 2 weeks

### Important (Should Have for Credibility):

1. **Section 232 Detection** (aluminum, steel)
2. **Drawback opportunity identification**
3. **Country-specific restrictions** (Russia sanctions, China entity list)
4. **De minimis threshold handling** ($800 US, varies by country)

### Nice to Have (Competitive Advantages):

1. **Multi-country import analysis** (EU, Canada, UK)
2. **Total landed cost** (freight + duty + fees)
3. **Alternative supplier suggestions** (with duty comparison)
4. **Compliance risk scoring**

---

## Conclusion

**The architecture is excellent. The infrastructure (Sprint E) is production-ready. But the TARIFF INTELLIGENCE DATA is make-or-break.**

- ✓ If HTS + Section 301 + FTA rules + AD/CVD databases are loaded → **System delivers value**
- ⚠ If only HTS + Section 301 loaded → **System is weak but usable**
- ❌ If minimal data loaded → **System is not production-ready**

**Critical Path to Launch:**
1. Verify FTA rules engine works (test USMCA on AP-1008)
2. Verify AD/CVD detection works (test on AP-1011)
3. Run this full automotive BOM test against live system
4. If 9+/12 SKUs pass → launch with disclaimers
5. If <9/12 pass → fix gaps before customer demos

**Bottom Line:**
This system can absolutely work and deliver real value to importers. The code quality and architecture are strong. But without the tariff intelligence databases properly loaded and integrated, it's just an expensive CSV parser. **The data is the product.**

---

**Report Generated:** 2026-02-11
**Author:** Claude (Sprint E Completion)
**Next Action:** Run full API test with production databases
