"""S3 storage backend for proofs, evidence, and USITC data.

Replaces local filesystem storage with tenant-isolated S3 buckets
for production scalability and disaster recovery.

Bucket Structure:
    potatobacon-production/
    ├── tenants/{tenant_id}/proofs/{proof_id}.json
    ├── tenants/{tenant_id}/evidence/{sha256_hash}
    ├── tenants/{tenant_id}/bom_uploads/{upload_id}.csv
    └── shared/usitc/editions/{edition_id}.jsonl
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

# S3 configuration
S3_BUCKET = os.getenv("S3_BUCKET", "potatobacon-production")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


class S3Backend:
    """S3 storage abstraction for proofs, evidence, and archives."""

    def __init__(self, bucket: Optional[str] = None, region: Optional[str] = None):
        self.bucket = bucket or S3_BUCKET
        self.region = region or AWS_REGION

        # Create S3 client (uses AWS credentials from environment/IAM role)
        self.client = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    # -------------------------------------------------------------------------
    # Proof Storage
    # -------------------------------------------------------------------------

    def save_proof(
        self, tenant_id: str, proof_id: str, proof_data: Dict[str, Any]
    ) -> str:
        """Upload proof JSON to S3.

        Args:
            tenant_id: Tenant identifier
            proof_id: Proof identifier (SHA-256 hash)
            proof_data: Proof content dict

        Returns:
            S3 key of uploaded proof
        """
        key = f"tenants/{tenant_id}/proofs/{proof_id}.json"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(proof_data, indent=2),
            ContentType="application/json",
            ServerSideEncryption="AES256",
            Metadata={"tenant_id": tenant_id, "proof_id": proof_id},
        )
        return key

    def get_proof(
        self, tenant_id: str, proof_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve proof JSON from S3.

        Args:
            tenant_id: Tenant identifier
            proof_id: Proof identifier

        Returns:
            Proof content dict or None if not found
        """
        key = f"tenants/{tenant_id}/proofs/{proof_id}.json"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def delete_proof(self, tenant_id: str, proof_id: str) -> bool:
        """Delete proof from S3.

        Args:
            tenant_id: Tenant identifier
            proof_id: Proof identifier

        Returns:
            True if deleted, False if not found
        """
        key = f"tenants/{tenant_id}/proofs/{proof_id}.json"
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    # -------------------------------------------------------------------------
    # Evidence Storage
    # -------------------------------------------------------------------------

    def save_evidence(
        self,
        tenant_id: str,
        evidence_id: str,
        content: bytes,
        content_type: str,
        original_filename: Optional[str] = None,
    ) -> str:
        """Upload evidence blob to S3.

        Args:
            tenant_id: Tenant identifier
            evidence_id: Evidence identifier (SHA-256 hash)
            content: Raw bytes
            content_type: MIME type
            original_filename: Original filename (for metadata)

        Returns:
            S3 key of uploaded evidence
        """
        key = f"tenants/{tenant_id}/evidence/{evidence_id}"
        metadata = {"tenant_id": tenant_id, "evidence_id": evidence_id}
        if original_filename:
            metadata["original_filename"] = original_filename

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
            ServerSideEncryption="AES256",
            Metadata=metadata,
        )
        return key

    def get_evidence(self, tenant_id: str, evidence_id: str) -> Optional[bytes]:
        """Retrieve evidence blob from S3.

        Args:
            tenant_id: Tenant identifier
            evidence_id: Evidence identifier

        Returns:
            Raw bytes or None if not found
        """
        key = f"tenants/{tenant_id}/evidence/{evidence_id}"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def delete_evidence(self, tenant_id: str, evidence_id: str) -> bool:
        """Delete evidence from S3.

        Args:
            tenant_id: Tenant identifier
            evidence_id: Evidence identifier

        Returns:
            True if deleted, False if not found
        """
        key = f"tenants/{tenant_id}/evidence/{evidence_id}"
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    # -------------------------------------------------------------------------
    # Presigned URLs
    # -------------------------------------------------------------------------

    def generate_presigned_url(
        self, key: str, expiration: int = 3600
    ) -> str:
        """Generate temporary download URL for evidence/proof.

        Args:
            key: S3 key
            expiration: URL expiration in seconds (default: 1 hour)

        Returns:
            Presigned URL
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expiration,
        )

    def generate_upload_url(
        self, key: str, content_type: str, expiration: int = 3600
    ) -> str:
        """Generate temporary upload URL for client-side uploads.

        Args:
            key: S3 key
            content_type: MIME type
            expiration: URL expiration in seconds

        Returns:
            Presigned POST URL
        """
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expiration,
        )

    # -------------------------------------------------------------------------
    # BOM Uploads
    # -------------------------------------------------------------------------

    def save_bom_upload(
        self, tenant_id: str, upload_id: str, content: bytes, filename: str
    ) -> str:
        """Upload BOM CSV/Excel file to S3 staging area.

        Args:
            tenant_id: Tenant identifier
            upload_id: Upload identifier
            content: Raw bytes
            filename: Original filename

        Returns:
            S3 key of uploaded BOM
        """
        key = f"tenants/{tenant_id}/bom_uploads/{upload_id}"
        content_type = (
            "text/csv"
            if filename.endswith(".csv")
            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
            ServerSideEncryption="AES256",
            Metadata={
                "tenant_id": tenant_id,
                "upload_id": upload_id,
                "original_filename": filename,
            },
        )
        return key

    # -------------------------------------------------------------------------
    # USITC Data Archives
    # -------------------------------------------------------------------------

    def save_usitc_edition(
        self, edition_id: str, content: bytes
    ) -> str:
        """Upload USITC HTS edition to shared archive.

        Args:
            edition_id: Edition identifier (e.g., "2024-01-basic")
            content: JSONL content

        Returns:
            S3 key of uploaded edition
        """
        key = f"shared/usitc/editions/{edition_id}.jsonl"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType="application/jsonl",
            ServerSideEncryption="AES256",
            Metadata={"edition_id": edition_id},
        )
        return key

    def get_usitc_edition(self, edition_id: str) -> Optional[bytes]:
        """Retrieve USITC HTS edition from archive.

        Args:
            edition_id: Edition identifier

        Returns:
            JSONL content or None if not found
        """
        key = f"shared/usitc/editions/{edition_id}.jsonl"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def head_object(self, key: str) -> Optional[Dict[str, Any]]:
        """Get object metadata without downloading.

        Args:
            key: S3 key

        Returns:
            Metadata dict or None if not found
        """
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
            return {
                "content_length": response["ContentLength"],
                "content_type": response["ContentType"],
                "last_modified": response["LastModified"],
                "metadata": response.get("Metadata", {}),
            }
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            raise

    def list_objects(self, prefix: str, max_keys: int = 1000) -> list:
        """List objects with given prefix.

        Args:
            prefix: S3 key prefix
            max_keys: Maximum number of keys to return

        Returns:
            List of object dicts
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix, MaxKeys=max_keys
            )
            return response.get("Contents", [])
        except ClientError:
            return []


# -------------------------------------------------------------------------
# Singleton instance
# -------------------------------------------------------------------------

_s3_backend: Optional[S3Backend] = None


def get_s3_backend() -> S3Backend:
    """Get singleton S3 backend.

    Returns:
        Initialized S3Backend instance
    """
    global _s3_backend
    if _s3_backend is None:
        _s3_backend = S3Backend()
    return _s3_backend
