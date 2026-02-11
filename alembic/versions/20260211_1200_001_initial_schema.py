"""Initial schema for production PostgreSQL backend

Revision ID: 001
Revises:
Create Date: 2026-02-11 12:00:00.000000

Creates 8 tables for Sprint E:
- tenants (with Stripe fields for Sprint F)
- api_keys
- skus
- proofs
- analysis_sessions
- jobs
- alerts
- evidence_metadata
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create ENUM types
    op.execute("CREATE TYPE plan_enum AS ENUM ('starter', 'professional', 'enterprise')")
    op.execute("CREATE TYPE job_status_enum AS ENUM ('queued', 'running', 'completed', 'failed')")

    # Create tenants table
    op.create_table(
        'tenants',
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('plan', postgresql.ENUM('starter', 'professional', 'enterprise', name='plan_enum'), nullable=True),
        sa.Column('sku_limit', sa.Integer(), nullable=True),
        sa.Column('monthly_analysis_limit', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('stripe_customer_id', sa.String(length=255), nullable=True),
        sa.Column('stripe_subscription_id', sa.String(length=255), nullable=True),
        sa.Column('subscription_status', sa.String(length=32), nullable=True),
        sa.Column('monthly_analyses', sa.Integer(), nullable=True),
        sa.Column('billing_period_start', sa.Date(), nullable=True),
        sa.Column('metadata_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('tenant_id'),
        sa.UniqueConstraint('stripe_customer_id')
    )

    # Create api_keys table
    op.create_table(
        'api_keys',
        sa.Column('key_hash', sa.String(length=64), nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.tenant_id'], ),
        sa.PrimaryKeyConstraint('key_hash')
    )

    # Create skus table
    op.create_table(
        'skus',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('sku_id', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('current_hts', sa.String(length=16), nullable=True),
        sa.Column('origin_country', sa.String(length=2), nullable=True),
        sa.Column('declared_value_per_unit', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('annual_volume', sa.Integer(), nullable=True),
        sa.Column('inferred_category', sa.String(length=64), nullable=True),
        sa.Column('category_confidence', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.tenant_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'sku_id', name='uq_tenant_sku')
    )
    op.create_index('idx_sku_tenant_hts', 'skus', ['tenant_id', 'current_hts'], unique=False)
    op.create_index(op.f('ix_skus_sku_id'), 'skus', ['sku_id'], unique=False)
    op.create_index(op.f('ix_skus_tenant_id'), 'skus', ['tenant_id'], unique=False)

    # Create proofs table
    op.create_table(
        'proofs',
        sa.Column('proof_id', sa.String(length=64), nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('s3_bucket', sa.String(length=255), nullable=False),
        sa.Column('s3_key', sa.String(length=512), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('sku_id', sa.String(length=255), nullable=True),
        sa.Column('baseline_hts', sa.String(length=16), nullable=True),
        sa.Column('optimized_hts', sa.String(length=16), nullable=True),
        sa.Column('duty_savings', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('summary_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.tenant_id'], ),
        sa.PrimaryKeyConstraint('proof_id')
    )
    op.create_index('idx_proof_tenant_sku', 'proofs', ['tenant_id', 'sku_id'], unique=False)
    op.create_index(op.f('ix_proofs_tenant_id'), 'proofs', ['tenant_id'], unique=False)
    op.create_index(op.f('ix_proofs_sku_id'), 'proofs', ['sku_id'], unique=False)

    # Create analysis_sessions table
    op.create_table(
        'analysis_sessions',
        sa.Column('session_id', sa.String(length=64), nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('context', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('mutations', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('final_classification', sa.String(length=16), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.tenant_id'], ),
        sa.PrimaryKeyConstraint('session_id')
    )
    op.create_index(op.f('ix_analysis_sessions_tenant_id'), 'analysis_sessions', ['tenant_id'], unique=False)

    # Create jobs table
    op.create_table(
        'jobs',
        sa.Column('job_id', sa.String(length=64), nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('job_type', sa.String(length=64), nullable=False),
        sa.Column('status', postgresql.ENUM('queued', 'running', 'completed', 'failed', name='job_status_enum'), nullable=True),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_items', sa.Integer(), nullable=True),
        sa.Column('completed_items', sa.Integer(), nullable=True),
        sa.Column('failed_items', sa.Integer(), nullable=True),
        sa.Column('input_params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('errors', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('callback_url', sa.String(length=512), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.tenant_id'], ),
        sa.PrimaryKeyConstraint('job_id'),
        sa.UniqueConstraint('celery_task_id')
    )
    op.create_index('idx_job_created', 'jobs', ['created_at'], unique=False)
    op.create_index('idx_job_tenant_status', 'jobs', ['tenant_id', 'status'], unique=False)
    op.create_index(op.f('ix_jobs_tenant_id'), 'jobs', ['tenant_id'], unique=False)

    # Create alerts table
    op.create_table(
        'alerts',
        sa.Column('alert_id', sa.String(length=64), nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('sku_id', sa.String(length=255), nullable=True),
        sa.Column('change_type', sa.String(length=32), nullable=False),
        sa.Column('current_hts', sa.String(length=16), nullable=True),
        sa.Column('old_rate', sa.String(length=32), nullable=True),
        sa.Column('new_rate', sa.String(length=32), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('detected_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('acknowledged', sa.Boolean(), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.tenant_id'], ),
        sa.PrimaryKeyConstraint('alert_id')
    )
    op.create_index('idx_alert_tenant_ack', 'alerts', ['tenant_id', 'acknowledged'], unique=False)
    op.create_index(op.f('ix_alerts_sku_id'), 'alerts', ['sku_id'], unique=False)
    op.create_index(op.f('ix_alerts_tenant_id'), 'alerts', ['tenant_id'], unique=False)

    # Create evidence_metadata table
    op.create_table(
        'evidence_metadata',
        sa.Column('evidence_id', sa.String(length=64), nullable=False),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('original_filename', sa.String(length=512), nullable=False),
        sa.Column('content_type', sa.String(length=128), nullable=False),
        sa.Column('byte_length', sa.Integer(), nullable=False),
        sa.Column('s3_bucket', sa.String(length=255), nullable=False),
        sa.Column('s3_key', sa.String(length=512), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('evidence_kind', sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.tenant_id'], ),
        sa.PrimaryKeyConstraint('evidence_id')
    )
    op.create_index('idx_evidence_tenant', 'evidence_metadata', ['tenant_id'], unique=False)
    op.create_index(op.f('ix_evidence_metadata_tenant_id'), 'evidence_metadata', ['tenant_id'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index(op.f('ix_evidence_metadata_tenant_id'), table_name='evidence_metadata')
    op.drop_index('idx_evidence_tenant', table_name='evidence_metadata')
    op.drop_table('evidence_metadata')

    op.drop_index(op.f('ix_alerts_tenant_id'), table_name='alerts')
    op.drop_index(op.f('ix_alerts_sku_id'), table_name='alerts')
    op.drop_index('idx_alert_tenant_ack', table_name='alerts')
    op.drop_table('alerts')

    op.drop_index(op.f('ix_jobs_tenant_id'), table_name='jobs')
    op.drop_index('idx_job_tenant_status', table_name='jobs')
    op.drop_index('idx_job_created', table_name='jobs')
    op.drop_table('jobs')

    op.drop_index(op.f('ix_analysis_sessions_tenant_id'), table_name='analysis_sessions')
    op.drop_table('analysis_sessions')

    op.drop_index(op.f('ix_proofs_sku_id'), table_name='proofs')
    op.drop_index(op.f('ix_proofs_tenant_id'), table_name='proofs')
    op.drop_index('idx_proof_tenant_sku', table_name='proofs')
    op.drop_table('proofs')

    op.drop_index(op.f('ix_skus_tenant_id'), table_name='skus')
    op.drop_index(op.f('ix_skus_sku_id'), table_name='skus')
    op.drop_index('idx_sku_tenant_hts', table_name='skus')
    op.drop_table('skus')

    op.drop_table('api_keys')
    op.drop_table('tenants')

    # Drop ENUM types
    op.execute("DROP TYPE job_status_enum")
    op.execute("DROP TYPE plan_enum")
