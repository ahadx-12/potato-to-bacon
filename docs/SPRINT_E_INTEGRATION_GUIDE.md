# Sprint E Integration Guide

This document provides detailed instructions for completing the remaining Sprint E integrations.

## ✅ Completed Components

- PostgreSQL models (8 tables)
- Alembic migrations
- Redis caching layer with distributed locks
- Celery workers with BOM analysis task
- S3 storage backend
- Migration scripts (JSONL→PostgreSQL, Evidence→S3)
- **routes_bom.py** - Celery integration (feature flag: `PTB_JOB_BACKEND=celery`)
- **routes_jobs.py** - PostgreSQL job storage (feature flag: `PTB_STORAGE_BACKEND=postgres`)
- **solver_z3_cached.py** - Redis-backed Z3 caching

## 🔧 Remaining Integrations

### 1. tenants.py - PostgreSQL Tenant Registry

**Current State:** In-memory TenantRegistry with explicit comment "PostgreSQL for production"

**Integration Pattern:**

```python
# Add at top of file
USE_POSTGRES = os.getenv("PTB_STORAGE_BACKEND", "jsonl").lower() == "postgres"

if USE_POSTGRES:
    from potatobacon.db.models import Tenant as TenantModel, APIKey as APIKeyModel
    from potatobacon.db.session import get_standalone_session

# Add PostgreSQL-backed registry class
class PostgresTenantRegistry:
    """PostgreSQL-backed tenant registry."""

    def resolve(self, api_key: str) -> Optional[Tenant]:
        """Look up tenant by API key from PostgreSQL."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        with get_standalone_session() as session:
            api_key_record = session.query(APIKeyModel).filter_by(
                key_hash=key_hash,
                revoked=False
            ).first()

            if not api_key_record:
                return None

            tenant_model = session.query(TenantModel).filter_by(
                tenant_id=api_key_record.tenant_id
            ).first()

            if not tenant_model:
                return None

            # Convert SQLAlchemy model to dataclass
            return Tenant(
                tenant_id=tenant_model.tenant_id,
                name=tenant_model.name,
                api_keys=[],  # Not needed for resolution
                plan=tenant_model.plan,
                created_at=tenant_model.created_at.isoformat(),
                sku_limit=tenant_model.sku_limit,
                monthly_analyses=tenant_model.monthly_analyses,
                monthly_analysis_limit=tenant_model.monthly_analysis_limit,
                metadata=tenant_model.metadata_json or {},
            )

    def increment_usage(self, tenant_id: str) -> None:
        """Increment monthly analysis counter."""
        with get_standalone_session() as session:
            tenant = session.query(TenantModel).filter_by(tenant_id=tenant_id).first()
            if tenant:
                tenant.monthly_analyses += 1
                session.commit()

# Update get_registry() function
def get_registry() -> TenantRegistry:
    if USE_POSTGRES:
        return PostgresTenantRegistry()
    return _registry  # Existing in-memory registry
```

**Testing:**
```bash
# Test with PostgreSQL
export PTB_STORAGE_BACKEND=postgres
export DATABASE_URL=postgresql://...
python -m pytest tests/test_tenants_postgres.py
```

---

### 2. sku_store.py - PostgreSQL SKU Backend

**Current State:** JSONL-backed SKUStore with threading.Lock

**Integration Pattern:**

```python
# Add at top
USE_POSTGRES = os.getenv("PTB_STORAGE_BACKEND", "jsonl").lower() == "postgres"

if USE_POSTGRES:
    from potatobacon.db.models import SKU as SKUModel
    from potatobacon.db.session import get_standalone_session

class PostgresSKUStore:
    """PostgreSQL-backed SKU store."""

    def upsert(self, sku_id: str, payload: Dict, tenant_id: str = "default") -> SKURecordModel:
        """Insert or update SKU in PostgreSQL."""
        with get_standalone_session() as session:
            existing = session.query(SKUModel).filter_by(
                tenant_id=tenant_id,
                sku_id=sku_id
            ).first()

            if existing:
                # Update existing
                for key, value in payload.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.now(timezone.utc)
            else:
                # Create new
                sku = SKUModel(
                    tenant_id=tenant_id,
                    sku_id=sku_id,
                    **payload,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(sku)

            session.commit()

            # Return as SKURecordModel for compatibility
            return SKURecordModel(
                sku_id=sku_id,
                description=payload.get("description"),
                # ... map all fields
            )

    def get(self, sku_id: str, tenant_id: str = "default") -> Optional[SKURecordModel]:
        """Retrieve SKU from PostgreSQL."""
        with get_standalone_session() as session:
            sku = session.query(SKUModel).filter_by(
                tenant_id=tenant_id,
                sku_id=sku_id
            ).first()

            if not sku:
                return None

            return SKURecordModel(
                sku_id=sku.sku_id,
                description=sku.description,
                current_hts=sku.current_hts,
                # ... map all fields
            )

    def list(self, tenant_id: str = "default", prefix: str = None, limit: int = 50) -> List[SKURecordModel]:
        """List SKUs from PostgreSQL."""
        with get_standalone_session() as session:
            query = session.query(SKUModel).filter_by(tenant_id=tenant_id)

            if prefix:
                query = query.filter(SKUModel.sku_id.startswith(prefix))

            skus = query.order_by(SKUModel.sku_id).limit(limit).all()

            return [
                SKURecordModel(
                    sku_id=sku.sku_id,
                    description=sku.description,
                    # ... map all fields
                )
                for sku in skus
            ]

# Update factory functions
def get_tenant_sku_store(tenant_id: str) -> SKUStore:
    if USE_POSTGRES:
        return PostgresSKUStore()
    return SKUStore(...)  # Existing JSONL store
```

---

### 3. evidence_store.py - S3 Backend

**Current State:** Local filesystem storage with SHA-256 addressing

**Integration Pattern:**

```python
# Add at top
USE_S3 = os.getenv("PTB_EVIDENCE_BACKEND", "local").lower() == "s3"

if USE_S3:
    from potatobacon.storage.s3_backend import get_s3_backend
    from potatobacon.db.models import EvidenceMetadata
    from potatobacon.db.session import get_standalone_session

class S3EvidenceStore:
    """S3-backed evidence store with PostgreSQL metadata."""

    def __init__(self):
        self.s3 = get_s3_backend()

    def store(
        self,
        content: bytes,
        original_filename: str,
        content_type: str,
        tenant_id: str = "default",
        evidence_kind: str = None,
    ) -> EvidenceRecord:
        """Upload evidence to S3 and store metadata in PostgreSQL."""
        # Compute SHA-256 for content addressing
        evidence_id = hashlib.sha256(content).hexdigest()

        # Upload to S3
        s3_key = self.s3.save_evidence(
            tenant_id=tenant_id,
            evidence_id=evidence_id,
            content=content,
            content_type=content_type,
            original_filename=original_filename,
        )

        # Store metadata in PostgreSQL
        with get_standalone_session() as session:
            metadata = EvidenceMetadata(
                evidence_id=evidence_id,
                tenant_id=tenant_id,
                original_filename=original_filename,
                content_type=content_type,
                byte_length=len(content),
                s3_bucket=self.s3.bucket,
                s3_key=s3_key,
                evidence_kind=evidence_kind,
            )
            session.add(metadata)
            session.commit()

        return EvidenceRecord(
            evidence_id=evidence_id,
            original_filename=original_filename,
            content_type=content_type,
            byte_length=len(content),
            sha256=evidence_id,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            evidence_kind=evidence_kind,
        )

    def retrieve(self, evidence_id: str, tenant_id: str = "default") -> Optional[bytes]:
        """Download evidence from S3."""
        return self.s3.get_evidence(tenant_id, evidence_id)

    def get_presigned_url(self, evidence_id: str, tenant_id: str = "default", expiration: int = 3600) -> str:
        """Generate presigned URL for direct download."""
        with get_standalone_session() as session:
            metadata = session.query(EvidenceMetadata).filter_by(evidence_id=evidence_id).first()
            if not metadata:
                raise ValueError(f"Evidence {evidence_id} not found")

            return self.s3.generate_presigned_url(metadata.s3_key, expiration)

# Update factory
def get_default_evidence_store() -> EvidenceStore:
    if USE_S3:
        return S3EvidenceStore()
    return EvidenceStore()  # Local filesystem
```

---

### 4. proofs/store.py - PostgreSQL Metadata + S3 Storage

**Current State:** JSONL append-only ledger with linear scan

**Integration Pattern:**

```python
# Add at top
USE_POSTGRES = os.getenv("PTB_STORAGE_BACKEND", "jsonl").lower() == "postgres"
USE_S3 = os.getenv("PTB_EVIDENCE_BACKEND", "local").lower() == "s3"

if USE_POSTGRES:
    from potatobacon.db.models import Proof as ProofModel
    from potatobacon.db.session import get_standalone_session
    from potatobacon.storage.s3_backend import get_s3_backend

class PostgresProofStore:
    """PostgreSQL metadata + S3 JSON storage."""

    def __init__(self):
        self.s3 = get_s3_backend() if USE_S3 else None

    def store_proof(self, proof: Dict[str, Any], tenant_id: str = "default") -> str:
        """Store proof in S3 and metadata in PostgreSQL."""
        from potatobacon.proofs.canonical import canonical_json

        # Compute proof ID from canonical JSON
        canonical = canonical_json(proof)
        proof_id = hashlib.sha256(canonical.encode()).hexdigest()

        # Upload to S3 if enabled
        if self.s3:
            s3_key = self.s3.save_proof(tenant_id, proof_id, proof)
            s3_bucket = self.s3.bucket
        else:
            # Fallback to local storage
            s3_key = f"local/{tenant_id}/proofs/{proof_id}.json"
            s3_bucket = "local"

        # Store metadata in PostgreSQL
        with get_standalone_session() as session:
            metadata = ProofModel(
                proof_id=proof_id,
                tenant_id=tenant_id,
                s3_bucket=s3_bucket,
                s3_key=s3_key,
                sku_id=proof.get("sku_id"),
                baseline_hts=proof.get("baseline_hts"),
                optimized_hts=proof.get("optimized_hts"),
                duty_savings=proof.get("duty_savings"),
                summary_json=proof.get("summary"),
            )
            session.add(metadata)
            session.commit()

        return proof_id

    def get_proof(self, proof_id: str, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        """Retrieve proof from S3."""
        if self.s3:
            return self.s3.get_proof(tenant_id, proof_id)
        return None

    def list_proofs(self, tenant_id: str = "default", sku_id: str = None, limit: int = 50) -> List[Dict]:
        """List proof metadata from PostgreSQL."""
        with get_standalone_session() as session:
            query = session.query(ProofModel).filter_by(tenant_id=tenant_id)

            if sku_id:
                query = query.filter_by(sku_id=sku_id)

            proofs = query.order_by(ProofModel.created_at.desc()).limit(limit).all()

            return [
                {
                    "proof_id": p.proof_id,
                    "sku_id": p.sku_id,
                    "baseline_hts": p.baseline_hts,
                    "optimized_hts": p.optimized_hts,
                    "duty_savings": float(p.duty_savings) if p.duty_savings else None,
                    "created_at": p.created_at.isoformat(),
                }
                for p in proofs
            ]
```

---

### 5. routes_portfolio.py - PostgreSQL Alert Queries

**Current State:** In-memory `_alerts` dict (MVP comment at line ~40)

**Integration Pattern:**

```python
# Add at top
USE_POSTGRES = os.getenv("PTB_STORAGE_BACKEND", "jsonl").lower() == "postgres"

if USE_POSTGRES:
    from potatobacon.db.models import Alert as AlertModel
    from potatobacon.db.session import get_db_session

# Update list_alerts endpoint
@router.get("/alerts", response_model=List[AlertResponse])
def list_alerts(
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> List[AlertResponse]:
    """List tariff schedule change alerts."""
    if USE_POSTGRES:
        from sqlalchemy.orm import Session

        db_session: Session = next(get_db_session())
        try:
            alerts = db_session.query(AlertModel).filter_by(
                tenant_id=tenant.tenant_id,
                acknowledged=False
            ).order_by(AlertModel.detected_at.desc()).limit(50).all()

            return [
                AlertResponse(
                    alert_id=alert.alert_id,
                    sku_id=alert.sku_id,
                    change_type=alert.change_type,
                    current_hts=alert.current_hts,
                    old_rate=alert.old_rate,
                    new_rate=alert.new_rate,
                    description=alert.description,
                    detected_at=alert.detected_at.isoformat(),
                    acknowledged=alert.acknowledged,
                )
                for alert in alerts
            ]
        finally:
            db_session.close()
    else:
        # In-memory fallback
        with _lock:
            tenant_alerts = [a for a in _alerts.values() if a.tenant_id == tenant.tenant_id]

        return [AlertResponse(...) for a in tenant_alerts]

# Update acknowledge_alert endpoint
@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: str,
    tenant: Tenant = Depends(resolve_tenant_from_request),
):
    """Mark an alert as acknowledged."""
    if USE_POSTGRES:
        from sqlalchemy.orm import Session

        db_session: Session = next(get_db_session())
        try:
            alert = db_session.query(AlertModel).filter_by(alert_id=alert_id).first()
            if not alert or alert.tenant_id != tenant.tenant_id:
                raise HTTPException(status_code=404)

            alert.acknowledged = True
            alert.acknowledged_at = datetime.now(timezone.utc)
            db_session.commit()
        finally:
            db_session.close()
    else:
        # In-memory
        with _lock:
            if alert_id not in _alerts:
                raise HTTPException(status_code=404)
            _alerts[alert_id].acknowledged = True
```

---

## Environment Variables

```bash
# Feature Flags
PTB_STORAGE_BACKEND=postgres    # Use PostgreSQL (default: jsonl)
PTB_JOB_BACKEND=celery          # Use Celery (default: threads)
PTB_EVIDENCE_BACKEND=s3         # Use S3 (default: local)
PTB_REDIS_CACHE=true            # Enable Redis caching (default: true)

# Database
DATABASE_URL=postgresql://user:pass@host:5432/potatobacon

# Redis
REDIS_URL=redis://host:6379/0
CELERY_BROKER_URL=redis://host:6379/1
CELERY_RESULT_BACKEND=db+postgresql://user:pass@host:5432/potatobacon

# S3
S3_BUCKET=potatobacon-production
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

---

## Testing Checklist

### Unit Tests
- [ ] Test PostgreSQL tenant resolution
- [ ] Test SKU CRUD operations with PostgreSQL
- [ ] Test evidence upload/download with S3
- [ ] Test proof storage with S3
- [ ] Test alert CRUD with PostgreSQL

### Integration Tests
- [ ] Run migration scripts (backfill + verify)
- [ ] Test Celery BOM analysis end-to-end
- [ ] Test job status polling with PostgreSQL
- [ ] Test Redis cache hit/miss
- [ ] Test S3 presigned URLs

### Load Tests
- [ ] 100 concurrent BOM analyses
- [ ] Verify Z3 cache hit rate >70%
- [ ] Verify Celery worker scaling
- [ ] Verify PostgreSQL query performance

---

## Deployment Order

1. **Deploy PostgreSQL** - Run `alembic upgrade head`
2. **Migrate Data** - Run migration scripts
3. **Deploy Redis** - ElastiCache or equivalent
4. **Deploy S3** - Create bucket with IAM policy
5. **Deploy Celery Workers** - `celery -A potatobacon.workers.celery_app worker`
6. **Enable Feature Flags** - Gradually roll out (10% → 50% → 100%)
7. **Monitor** - Check logs, metrics, error rates

---

## Rollback Plan

If issues occur:

1. **Disable feature flags**: `PTB_STORAGE_BACKEND=jsonl PTB_JOB_BACKEND=threads`
2. **System falls back to JSONL + threading** (backward compatible)
3. **Investigate issues** in PostgreSQL/Redis/S3 logs
4. **Fix and re-enable** flags when ready

---

## Estimated Effort

| Component | Lines | Effort |
|-----------|-------|--------|
| tenants.py | ~50 | 1 hour |
| sku_store.py | ~80 | 2 hours |
| evidence_store.py | ~60 | 1.5 hours |
| proofs/store.py | ~70 | 2 hours |
| routes_portfolio.py | ~40 | 1 hour |
| **Total** | **~300** | **7-8 hours** |

---

## Support

For questions or issues:
- Review existing integrations in `routes_bom.py` and `routes_jobs.py`
- Check migration scripts for data access patterns
- Refer to SQLAlchemy models in `db/models.py`
- Test with feature flags before full rollout
