"""FraudForge AI: Agentic Payment & Provenance Chain Schema.

Defines typed representations for agentic payment interactions and external
information provenance chains consumed prior to payment authorization.
All representations are strictly research-oriented and synthetic.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse


@dataclass
class ProvenanceSource:
    """Represents a single external evidence source consumed by an AI agent."""
    url: str
    source_type: str = "web_page"
    domain: str = field(init=False)

    def __post_init__(self):
        try:
            parsed = urlparse(self.url if "://" in self.url else f"https://{self.url}")
            self.domain = parsed.netloc.lower() or "unknown"
        except Exception:
            self.domain = "unknown"


@dataclass
class ProvenanceChain:
    """Represents the ordered list of external sources consumed prior to authorization."""
    sources: List[str] = field(default_factory=list)
    provenance_type: str = "benign"  # benign | suspicious | poisoned

    def to_list(self) -> List[str]:
        """Return raw URLs as a list of strings."""
        return list(self.sources)

    @classmethod
    def from_list(cls, urls: List[str], provenance_type: str = "benign") -> "ProvenanceChain":
        """Construct from list of URL strings."""
        return cls(sources=[str(u).strip() for u in urls if str(u).strip()], provenance_type=provenance_type)


@dataclass
class AgenticPaymentRecord:
    """Additive data model for an agentic payment authorization request.
    
    Preserves existing transaction features without modification while attaching
    the external provenance chain consumed by the agent.
    """
    transaction_features: Dict[str, Any]
    provenance_chain: List[str] = field(default_factory=list)
    agent_id: str = "payment_agent_v1"
    session_id: str = "sess_synthetic"

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten record for DataFrame ingestion while keeping provenance as a list object."""
        record = dict(self.transaction_features)
        record["provenance_chain"] = list(self.provenance_chain)
        record["agent_id"] = self.agent_id
        record["session_id"] = self.session_id
        return record
