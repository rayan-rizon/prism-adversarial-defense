"""
PRISM Federation Protocol — Peer-to-peer attack signature sharing.

Architecture: epidemic (gossip) broadcast over UDP multicast on a local subnet.
Each PRISM node listens for incoming SignatureMessage packets and merges
received signatures into its local ImmuneMemory.  When a local detection
fires at L2 or L3, the node broadcasts the offending persistence diagram so
peers can recognise the same attack pattern without needing to accumulate
the full detection-latency evidence themselves.

Key design properties:
- Zero configuration: multicast on 239.255.0.1:9876 (site-local scope).
- No central coordinator: every node is a symmetric peer.
- TTL-bounded re-broadcast: max_hops prevents infinite loops.
- Graceful degradation: network failures never block the main defend() path.
"""
import hashlib
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Multicast group — site-local scope; does not leave the local network segment.
_MCAST_GROUP = "239.255.0.1"
_MCAST_PORT  = 9876
_MAX_PACKET  = 65_000   # UDP practical limit
_DEFAULT_TTL = 2        # Bounces before packet discarded


@dataclass
class SignatureMessage:
    """
    A serialisable attack-signature broadcast message.

    diagram_h1 is the H1 (loop/cycle) persistence diagram stored as a list of
    [birth, death] float32 pairs.  H0 is omitted because it is dominated by
    the connected-component peak and carries less discriminative information
    for adversarial detection.
    """
    instance_id:    str
    diagram_h1:     List[List[float]]   # [[birth, death], ...]
    attack_type:    str
    response_level: str                 # 'L2' or 'L3'
    confidence:     float
    timestamp:      float = field(default_factory=time.time)
    max_hops:       int = field(default=_DEFAULT_TTL)

    # Derived fingerprint — used to deduplicate re-broadcasts.
    fingerprint:    str = field(default="")

    def __post_init__(self) -> None:
        if not self.fingerprint:
            blob = json.dumps(self.diagram_h1, sort_keys=True).encode()
            self.fingerprint = hashlib.md5(blob).hexdigest()[:12]

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> "SignatureMessage":
        d = json.loads(data.decode("utf-8"))
        return SignatureMessage(**d)


class FederationBroadcaster:
    """Sends SignatureMessage packets to all peers in the multicast group."""

    def __init__(
        self,
        instance_id: str,
        mcast_group: str = _MCAST_GROUP,
        mcast_port:  int = _MCAST_PORT,
    ) -> None:
        self.instance_id = instance_id
        self.mcast_group = mcast_group
        self.mcast_port  = mcast_port

    def broadcast(self, msg: SignatureMessage) -> None:
        """
        Send msg to the multicast group.  Failures are logged but never raised
        so the main inference path is never blocked.
        """
        try:
            payload = msg.to_bytes()
            if len(payload) > _MAX_PACKET:
                logger.warning(
                    "SignatureMessage exceeds UDP limit (%d bytes); truncating "
                    "diagram_h1 to first 500 points.", len(payload)
                )
                msg.diagram_h1 = msg.diagram_h1[:500]
                payload = msg.to_bytes()

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, _DEFAULT_TTL)
            sock.sendto(payload, (self.mcast_group, self.mcast_port))
            sock.close()
            logger.debug(
                "Broadcast: instance=%s level=%s fp=%s hops=%d",
                msg.instance_id, msg.response_level, msg.fingerprint, msg.max_hops,
            )
        except OSError as exc:
            logger.warning("Federation broadcast failed (non-fatal): %s", exc)


class FederationListener:
    """
    Background thread that receives incoming SignatureMessage packets and
    merges them into the local ImmuneMemory.  Also re-broadcasts with
    decremented hop count (epidemic gossip).
    """

    def __init__(
        self,
        instance_id:   str,
        immune_memory,               # ImmuneMemory — typed as Any to avoid circular import
        broadcaster:   FederationBroadcaster,
        mcast_group:   str = _MCAST_GROUP,
        mcast_port:    int = _MCAST_PORT,
    ) -> None:
        self.instance_id   = instance_id
        self.immune_memory = immune_memory
        self.broadcaster   = broadcaster
        self.mcast_group   = mcast_group
        self.mcast_port    = mcast_port

        self._running:          bool              = False
        self._thread:           Optional[threading.Thread] = None
        self._seen_fingerprints: set              = set()  # dedup window
        self._stats = {"received": 0, "merged": 0, "duplicate": 0, "error": 0}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background listener thread (daemon — exits with process)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, name="prism-fed-listener", daemon=True
        )
        self._thread.start()
        logger.info(
            "Federation listener started: %s group=%s port=%d",
            self.instance_id, self.mcast_group, self.mcast_port,
        )

    def stop(self) -> None:
        """Signal the listener thread to exit."""
        self._running = False

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> dict:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        sock = self._make_socket()
        if sock is None:
            logger.warning("Federation listener could not bind; disabling.")
            return

        sock.settimeout(1.0)  # wake up every second to check _running

        while self._running:
            try:
                data, addr = sock.recvfrom(_MAX_PACKET + 1024)
            except socket.timeout:
                continue
            except OSError:
                break

            self._stats["received"] += 1
            try:
                msg = SignatureMessage.from_bytes(data)
            except Exception as exc:
                self._stats["error"] += 1
                logger.debug("Malformed federation packet from %s: %s", addr, exc)
                continue

            # Don't merge your own signatures
            if msg.instance_id == self.instance_id:
                continue

            # Deduplicate: skip if we've seen this fingerprint recently
            if msg.fingerprint in self._seen_fingerprints:
                self._stats["duplicate"] += 1
                continue
            self._seen_fingerprints.add(msg.fingerprint)
            # Keep dedup window bounded
            if len(self._seen_fingerprints) > 5_000:
                self._seen_fingerprints.clear()

            # Merge into local immune memory
            try:
                h1 = np.array(msg.diagram_h1, dtype=np.float64)
                # Reconstitute a full diagram list: [H0_empty, H1]
                diagram = [np.empty((0, 2), dtype=np.float64), h1]
                self.immune_memory.store(
                    diagram=diagram,
                    attack_type=msg.attack_type,
                    response_level=msg.response_level,
                    confidence=msg.confidence * 0.9,  # slight discount for remote sig
                )
                self._stats["merged"] += 1
                logger.info(
                    "Merged remote signature: from=%s level=%s fp=%s",
                    msg.instance_id, msg.response_level, msg.fingerprint,
                )
            except Exception as exc:
                self._stats["error"] += 1
                logger.warning("Failed to merge signature %s: %s", msg.fingerprint, exc)
                continue

            # Re-broadcast with decremented hop count (gossip propagation)
            if msg.max_hops > 1:
                msg.max_hops -= 1
                self.broadcaster.broadcast(msg)

        sock.close()

    def _make_socket(self) -> Optional[socket.socket]:
        """Create and configure a multicast UDP receive socket."""
        try:
            sock = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
            )
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind(("", self.mcast_port))

            # Join multicast group
            mreq = struct.pack(
                "4sL", socket.inet_aton(self.mcast_group), socket.INADDR_ANY
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            return sock
        except OSError as exc:
            logger.warning("Could not create multicast socket: %s", exc)
            return None
