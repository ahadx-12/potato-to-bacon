"""Domain guard registry exports."""

from .registry import get_guard_for_domain
from .base_guard import DomainGuard

__all__ = ["get_guard_for_domain", "DomainGuard"]
