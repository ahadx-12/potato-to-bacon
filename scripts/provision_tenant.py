#!/usr/bin/env python3
"""Provision a TEaaS tenant with API key.

Usage:
    python scripts/provision_tenant.py --name "test-importer" --plan professional
    python scripts/provision_tenant.py --name "acme-corp" --plan enterprise

Creates a tenant in the active backend (in-memory or PostgreSQL),
generates an API key, registers it in both the security layer and
the tenant registry, and prints the key.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision a TEaaS tenant")
    parser.add_argument("--name", required=True, help="Tenant display name")
    parser.add_argument(
        "--plan",
        default="professional",
        choices=["starter", "professional", "pro", "enterprise"],
        help="Pricing plan (default: professional)",
    )
    parser.add_argument("--tenant-id", default=None, help="Custom tenant ID (auto-generated if omitted)")
    parser.add_argument("--api-key", default=None, help="Custom API key (auto-generated if omitted)")
    args = parser.parse_args()

    # Normalize plan name
    plan = args.plan
    if plan == "pro":
        plan = "professional"

    # Generate tenant ID and API key
    tenant_id = args.tenant_id or f"tenant_{secrets.token_hex(6)}"
    api_key = args.api_key or f"ptb_{secrets.token_urlsafe(32)}"

    # Register in security layer (CALE_API_KEYS)
    from potatobacon.api.security import register_api_key
    register_api_key(api_key)

    # Register in tenant registry
    from potatobacon.api.tenants import get_registry
    registry = get_registry()
    tenant = registry.register_tenant(
        tenant_id=tenant_id,
        name=args.name,
        api_key=api_key,
        plan=plan,
    )

    print("=" * 60)
    print("TEaaS Tenant Provisioned Successfully")
    print("=" * 60)
    print(f"  Tenant ID:      {tenant.tenant_id}")
    print(f"  Name:           {tenant.name}")
    print(f"  Plan:           {tenant.plan}")
    print(f"  SKU Limit:      {tenant.sku_limit}")
    print(f"  Monthly Limit:  {tenant.monthly_analysis_limit}")
    print(f"  API Key:        {api_key}")
    print("=" * 60)
    print()
    print("Usage:")
    print(f'  curl -H "X-API-Key: {api_key}" http://localhost:8000/v1/health')
    print()
    print("To start the server:")
    print("  uvicorn potatobacon.api.app:app --host 0.0.0.0 --port 8000")
    print()


if __name__ == "__main__":
    main()
