"""
PRISM Federation Manager — High-level API for enabling federated signature sharing.

Usage (inside PRISM):
    manager = FederationManager(instance_id="node-0", immune_memory=memory)
    manager.start()
    # When a detection fires:
    manager.on_detection(diagram, attack_type="PGD", response_level="L3")
"""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from .protocol import FederationBroadcaster, FederationListener, SignatureMessage

logger = logging.getLogger(__name__)


class FederationManager:
    """
    Coordinates broadcast + listening for one PRISM node.

    Thread-safety: `on_detection()` and `get_stats()` are safe to call from
    any thread.  The listener runs in a background daemon thread started by
    `start()`.
    """

    def __init__(
        self,
        instance_id: Optional[str] = None,
        immune_memory: Any = None,
        mcast_group: str = "239.255.0.1",
        mcast_port:  int = 9876,
    ) -> None:
        """
        Args:
            instance_id:   Unique identifier for this PRISM node.  Auto-generated
                           (UUID4) if not provided.
            immune_memory: The local ImmuneMemory instance to merge remote
                           signatures into.  If None, incoming signatures are
                           received and logged but not stored.
            mcast_group:   UDP multicast group address.
            mcast_port:    UDP port.
        """
        self.instance_id   = instance_id or str(uuid.uuid4())[:8]
        self.immune_memory = immune_memory
        self._started      = False
        self._broadcasts_sent = 0

        self.broadcaster = FederationBroadcaster(
            instance_id=self.instance_id,
            mcast_group=mcast_group,
            mcast_port=mcast_port,
        )
        self.listener = FederationListener(
            instance_id=self.instance_id,
            immune_memory=immune_memory,
            broadcaster=self.broadcaster,
            mcast_group=mcast_group,
            mcast_port=mcast_port,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background listener.  Safe to call multiple times."""
        if self._started:
            return
        if self.immune_memory is None:
            logger.warning(
                "FederationManager started without immune_memory — "
                "incoming signatures will be received but not stored."
            )
        self.listener.start()
        self._started = True
        logger.info("FederationManager started: instance_id=%s", self.instance_id)

    def stop(self) -> None:
        """Stop the background listener."""
        self.listener.stop()
        self._started = False

    def is_running(self) -> bool:
        return self._started and self.listener.is_alive()

    # ------------------------------------------------------------------
    # Detection callback
    # ------------------------------------------------------------------

    def on_detection(
        self,
        diagram: List[np.ndarray],
        attack_type: str = "UNKNOWN",
        response_level: str = "L2",
    ) -> None:
        """
        Called by PRISM when a local detection fires at L2 or L3.

        Constructs a SignatureMessage from the H1 persistence diagram and
        broadcasts it to the multicast group.  Network failures are caught
        and logged so the main inference path is never blocked.

        Args:
            diagram:        List of persistence diagrams [H0, H1, ...] returned
                            by TopologicalProfiler.compute_diagram().
            attack_type:    Attack type label, e.g. 'PGD', 'FGSM', 'UNKNOWN'.
            response_level: The PRISM defence level that fired ('L2' or 'L3').
        """
        try:
            # Extract H1 diagram (index 1); fall back to H0 if H1 is missing
            if len(diagram) > 1 and len(diagram[1]) > 0:
                h1 = diagram[1]
            elif len(diagram) > 0 and len(diagram[0]) > 0:
                h1 = diagram[0]
            else:
                logger.debug("Empty diagram; skipping federation broadcast.")
                return

            # Confidence scales with response level
            confidence_map = {"L1": 0.5, "L2": 0.75, "L3": 1.0, "L3_REJECT": 1.0}
            confidence = confidence_map.get(response_level, 0.5)

            msg = SignatureMessage(
                instance_id=self.instance_id,
                diagram_h1=h1.tolist(),
                attack_type=attack_type,
                response_level=response_level,
                confidence=confidence,
                timestamp=time.time(),
            )
            self.broadcaster.broadcast(msg)
            self._broadcasts_sent += 1
            logger.debug(
                "on_detection: broadcast level=%s fp=%s",
                response_level, msg.fingerprint,
            )
        except Exception as exc:
            logger.warning("on_detection failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return metrics for paper reporting / monitoring."""
        listener_stats = self.listener.get_stats()
        return {
            "instance_id":       self.instance_id,
            "running":           self.is_running(),
            "broadcasts_sent":   self._broadcasts_sent,
            "signatures_received": listener_stats.get("received", 0),
            "signatures_merged":   listener_stats.get("merged", 0),
            "duplicates_dropped":  listener_stats.get("duplicate", 0),
            "errors":              listener_stats.get("error", 0),
        }
