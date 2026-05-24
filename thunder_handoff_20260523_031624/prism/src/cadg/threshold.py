"""
CADG: Tiered Threshold Manager
Manages response-level thresholds and provides the escalation logic
for the multi-tier defense system.

This module was listed in the project structure but not implemented in the plan.
"""
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ResponseAction:
    """Describes what to do at each response level."""
    level: str
    description: str
    should_log: bool = True
    should_purify: bool = False
    should_route_expert: bool = False
    should_reject: bool = False


# Default response actions for each tier
DEFAULT_ACTIONS: Dict[str, ResponseAction] = {
    'PASS': ResponseAction(
        level='PASS',
        description='Clean input — normal inference',
        should_log=False,
    ),
    'L1': ResponseAction(
        level='L1',
        description='Suspicious — log and flag, normal inference',
        should_log=True,
    ),
    'L2': ResponseAction(
        level='L2',
        description='Likely adversarial — apply input purification',
        should_log=True,
        should_purify=True,
    ),
    'L3': ResponseAction(
        level='L3',
        description='High-confidence adversarial — route through expert or reject',
        should_log=True,
        should_route_expert=True,
    ),
    'L3_REJECT': ResponseAction(
        level='L3_REJECT',
        description='High-confidence adversarial, no expert available — reject',
        should_log=True,
        should_reject=True,
    ),
}


class TieredThresholdManager:
    """
    Manages the mapping from response levels to actions,
    and provides threshold adjustment during L0 activation.
    """

    def __init__(self, actions: Optional[Dict[str, ResponseAction]] = None):
        self.actions = actions or dict(DEFAULT_ACTIONS)

    def get_action(self, level: str) -> ResponseAction:
        if level not in self.actions:
            raise KeyError(f"Unknown level '{level}'. Available: {list(self.actions)}")
        return self.actions[level]

    def get_severity_order(self):
        """Return levels in order of severity (highest first)."""
        return ['L3_REJECT', 'L3', 'L2', 'L1', 'PASS']
