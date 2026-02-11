"""Database session factory and connection management.

Provides SQLAlchemy engine and session creation with connection pooling
for production PostgreSQL backend.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/potatobacon",
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,  # Keep 10 connections in pool
    max_overflow=20,  # Allow 20 additional connections
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False,  # Set to True for SQL logging in development
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db_session() -> Session:
    """FastAPI dependency for database sessions.

    Usage:
        @router.get("/endpoint")
        def my_endpoint(session: Session = Depends(get_db_session)):
            tenant = session.query(Tenant).filter_by(tenant_id="default").first()
            ...
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_standalone_session() -> Generator[Session, None, None]:
    """Context manager for standalone database sessions.

    Usage in scripts and Celery tasks:
        with get_standalone_session() as session:
            tenant = session.query(Tenant).filter_by(tenant_id="default").first()
            session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Initialize database tables (for testing only).

    In production, use Alembic migrations:
        alembic upgrade head
    """
    from potatobacon.db.models import Base

    Base.metadata.create_all(bind=engine)


def drop_all():
    """Drop all tables (for testing only)."""
    from potatobacon.db.models import Base

    Base.metadata.drop_all(bind=engine)
