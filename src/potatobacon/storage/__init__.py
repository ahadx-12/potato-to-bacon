"""Storage backends for TEaaS production infrastructure.

Provides S3-based storage for proofs, evidence, and archives.
"""

from potatobacon.storage.s3_backend import S3Backend, get_s3_backend

__all__ = ["S3Backend", "get_s3_backend"]
