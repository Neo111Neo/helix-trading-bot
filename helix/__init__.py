"""HELIX: a Jev x Opus paper-trading engine.

Layers (from the stack):
  data.py        Data sources (Kraken public market data, or a synthetic demo market)
  features.py    Data & feature engineering (pure math, no AI)
  state.py       Structured market state sent to Jev (numbers only, < 400 tokens)
  jev.py         Jev decision layer (typed questions, version pinned, deadline enforced)
  policy.py      Code turns Jev's probabilities into an action (thresholds in config.yaml)
  risk.py        Risk engine: hard limits that can veto any order
  broker.py      Paper execution with fees and slippage
  engine.py      The 24/7 loop + fallback ladder + kill switch
  journal.py     Logs every state, decision, trade and outcome (SQLite)
  calibration.py Checks whether Jev's probabilities hold up on your market
  analyst.py     Opus research layer: reviews results and proposes the next experiment
  server.py      API + dashboard
"""

__version__ = "1.0.0"
