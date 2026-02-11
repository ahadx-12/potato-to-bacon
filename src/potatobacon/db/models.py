"""SQLAlchemy models for production PostgreSQL backend.

Replaces JSONL-based stores with relational database tables for:
- Tenants and API keys (multi-tenant isolation)
- SKUs and analyses (tariff classification data)
- Proofs and evidence (audit trail with S3 references)
- Jobs (Celery task tracking)
- Alerts (schedule change notifications)
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Tenant(Base):
    """TEaaS customer account with subscription and usage tracking.

    Replaces in-memory TenantRegistry from api/tenants.py.
    """

    __tablename__ = "tenants"

    tenant_id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    plan = Column(
        Enum("starter", "professional", "enterprise", name="plan_enum"),
        default="starter",
    )
    sku_limit = Column(Integer, default=100)
    monthly_analysis_limit = Column(Integer, default=500)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Stripe fields for Sprint F
    stripe_customer_id = Column(String(255), unique=True, nullable=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    subscription_status = Column(String(32), default="active")  # active | canceled | past_due

    # Usage tracking (reset monthly by Stripe webhook)
    monthly_analyses = Column(Integer, default=0)
    billing_period_start = Column(Date, nullable=True)

    # Flexible metadata storage
    metadata_json = Column(JSONB, default={})

    # Relationships
    api_keys = relationship("APIKey", back_populates="tenant", cascade="all, delete-orphan")
    skus = relationship("SKU", back_populates="tenant", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="tenant", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="tenant", cascade="all, delete-orphan")
    analysis_sessions = relationship("AnalysisSession", back_populates="tenant", cascade="all, delete-orphan")


class APIKey(Base):
    """API key for tenant authentication.

    Supports multiple keys per tenant with revocation.
    """

    __tablename__ = "api_keys"

    key_hash = Column(String(64), primary_key=True)  # SHA-256 of actual key
    tenant_id = Column(String(64), ForeignKey("tenants.tenant_id"), nullable=False)
    name = Column(String(255), nullable=True)  # Human-readable label
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    revoked = Column(Boolean, default=False)

    tenant = relationship("Tenant", back_populates="api_keys")


class SKU(Base):
    """Product SKU with HTS classification and metadata.

    Replaces tariff/sku_store.py JSONL backend.
    """

    __tablename__ = "skus"

    # Composite PK for tenant isolation
    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), ForeignKey("tenants.tenant_id"), nullable=False, index=True)
    sku_id = Column(String(255), nullable=False, index=True)

    # SKURecordModel fields from tariff/sku_models.py
    description = Column(Text, nullable=True)
    current_hts = Column(String(16), nullable=True)
    origin_country = Column(String(2), nullable=True)
    declared_value_per_unit = Column(Numeric(12, 2), nullable=True)
    annual_volume = Column(Integer, nullable=True)
    inferred_category = Column(String(64), nullable=True)
    category_confidence = Column(Numeric(5, 4), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Store full metadata as JSONB for flexibility
    metadata_json = Column(JSONB, default={})

    tenant = relationship("Tenant", back_populates="skus")

    __table_args__ = (
        UniqueConstraint("tenant_id", "sku_id", name="uq_tenant_sku"),
        Index("idx_sku_tenant_hts", "tenant_id", "current_hts"),
    )


class Proof(Base):
    """Cryptographic proof of HTS classification.

    Stores metadata in PostgreSQL, full proof JSON in S3.
    Replaces proofs/store.py JSONL backend.
    """

    __tablename__ = "proofs"

    proof_id = Column(String(64), primary_key=True)  # SHA-256 hash from canonical_json
    tenant_id = Column(String(64), ForeignKey("tenants.tenant_id"), nullable=False, index=True)

    # S3 keys (multi-region support)
    s3_bucket = Column(String(255), nullable=False)
    s3_key = Column(String(512), nullable=False)  # tenants/{tenant_id}/proofs/{proof_id}.json

    # Metadata for quick queries without loading full proof
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    sku_id = Column(String(255), nullable=True, index=True)
    baseline_hts = Column(String(16), nullable=True)
    optimized_hts = Column(String(16), nullable=True)
    duty_savings = Column(Numeric(12, 2), nullable=True)

    # Summary for dashboard (full proof in S3)
    summary_json = Column(JSONB, nullable=True)

    tenant = relationship("Tenant")

    __table_args__ = (Index("idx_proof_tenant_sku", "tenant_id", "sku_id"),)


class AnalysisSession(Base):
    """Interactive analysis session with context and mutations.

    Replaces tariff/analysis_session_store.py JSONL backend.
    """

    __tablename__ = "analysis_sessions"

    session_id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), ForeignKey("tenants.tenant_id"), nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Session state stored as JSONB
    context = Column(JSONB, nullable=False)
    mutations = Column(JSONB, default=[])
    final_classification = Column(String(16), nullable=True)

    tenant = relationship("Tenant", back_populates="analysis_sessions")


class Job(Base):
    """Celery job tracking with progress and results.

    Replaces in-memory job queue from api/routes_jobs.py and law/jobs.py.
    """

    __tablename__ = "jobs"

    job_id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), ForeignKey("tenants.tenant_id"), nullable=False, index=True)
    job_type = Column(String(64), nullable=False)  # bom_analysis | portfolio_scan | arbitrage_hunt

    status = Column(
        Enum("queued", "running", "completed", "failed", name="job_status_enum"),
        default="queued",
    )

    # Celery task ID for tracking
    celery_task_id = Column(String(255), unique=True, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Progress tracking
    total_items = Column(Integer, default=0)
    completed_items = Column(Integer, default=0)
    failed_items = Column(Integer, default=0)

    # Input/output stored as JSONB
    input_params = Column(JSONB, nullable=True)
    result = Column(JSONB, nullable=True)
    errors = Column(JSONB, default=[])

    # For webhook callbacks
    callback_url = Column(String(512), nullable=True)

    tenant = relationship("Tenant", back_populates="jobs")

    __table_args__ = (
        Index("idx_job_tenant_status", "tenant_id", "status"),
        Index("idx_job_created", "created_at"),
    )


class Alert(Base):
    """Tariff schedule change alert for portfolio monitoring.

    Replaces in-memory _alerts dict from api/routes_portfolio.py.
    """

    __tablename__ = "alerts"

    alert_id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), ForeignKey("tenants.tenant_id"), nullable=False, index=True)
    sku_id = Column(String(255), nullable=True, index=True)

    change_type = Column(String(32), nullable=False)  # rate_changed | added | removed
    current_hts = Column(String(16), nullable=True)
    old_rate = Column(String(32), nullable=True)
    new_rate = Column(String(32), nullable=True)
    description = Column(Text, nullable=True)

    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)

    tenant = relationship("Tenant", back_populates="alerts")

    __table_args__ = (Index("idx_alert_tenant_ack", "tenant_id", "acknowledged"),)


class EvidenceMetadata(Base):
    """Evidence blob metadata with S3 reference.

    Content-addressed storage with SHA-256 as primary key.
    Replaces tariff/evidence_store.py local filesystem backend.
    """

    __tablename__ = "evidence_metadata"

    evidence_id = Column(String(64), primary_key=True)  # SHA-256 hash
    tenant_id = Column(String(64), ForeignKey("tenants.tenant_id"), nullable=False, index=True)

    original_filename = Column(String(512), nullable=False)
    content_type = Column(String(128), nullable=False)
    byte_length = Column(Integer, nullable=False)

    # S3 location
    s3_bucket = Column(String(255), nullable=False)
    s3_key = Column(String(512), nullable=False)  # tenants/{tenant_id}/evidence/{evidence_id}

    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    evidence_kind = Column(String(64), nullable=True)  # bom_csv | spec_sheet | cert

    tenant = relationship("Tenant")

    __table_args__ = (Index("idx_evidence_tenant", "tenant_id"),)
