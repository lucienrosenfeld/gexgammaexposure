"""ES Post-Auction Inventory-Resolution Strategy.

Research framework implementing the consolidated master specification:
a two-stage, sign-constrained, heavily regularised forecast of the
volatility-scaled ES return over the 16:00:15 -> 16:14:30 (NY) window,
with an options block that can delete itself, purged walk-forward
validation, deflated-Sharpe acceptance, and survival-oriented risk
controls.

Nothing in this package hard-codes the 16:15 halt; session times come
from :mod:`espa.sessions`. Nothing uses information after the decision
time; receipt-time discipline lives in :mod:`espa.timestamps`.
"""

from espa.config import SpecConstants, DEFAULT_CONSTANTS

__all__ = ["SpecConstants", "DEFAULT_CONSTANTS"]
__version__ = "0.1.0"
