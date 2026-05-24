"""
PRISM Federation — peer-to-peer attack signature sharing over UDP multicast.

Public API:
    FederationManager   — high-level per-node coordinator (start/stop/on_detection)
    SignatureMessage    — serialisable broadcast message
"""
from .manager import FederationManager
from .protocol import SignatureMessage

__all__ = ["FederationManager", "SignatureMessage"]
