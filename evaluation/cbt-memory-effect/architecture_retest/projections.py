"""Reuse the exact production CBT projection prompt and support contract."""
from pathlib import Path
import sys

INTEGRATION = Path(__file__).resolve().parents[3] / "modules/agent-integration"
sys.path.insert(0, str(INTEGRATION))
from cbt_projection_contract import (EXTRACT, VERIFY, FACT_SECTIONS, SECTIONS, VERIFIER_VERSION,
    bind_candidates, canonical, checked_items, fingerprint, request, strict_json)
