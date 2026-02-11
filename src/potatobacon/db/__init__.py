"""Database layer for TEaaS production PostgreSQL backend.

Provides SQLAlchemy models and session management.
"""

from potatobacon.db.models import (
    Alert,
    AnalysisSession,
    APIKey,
    Base,
    EvidenceMetadata,
    Job,
    Proof,
    SKU,
    Tenant,
)
from potatobacon.db.session import (
    SessionLocal,
    drop_all,
    engine,
    get_db_session,
    get_standalone_session,
    init_db,
)

__all__ = [
    # Models
    "Base",
    "Tenant",
    "APIKey",
    "SKU",
    "Proof",
    "AnalysisSession",
    "Job",
    "Alert",
    "EvidenceMetadata",
    # Session management
    "engine",
    "SessionLocal",
    "get_db_session",
    "get_standalone_session",
    "init_db",
    "drop_all",
]
