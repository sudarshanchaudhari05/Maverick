"""FraudForge AI: Provenance Feature Engineering.

Extracts deterministic, explainable numerical and categorical signals
from an agent's external information provenance chain.
All calculations are strictly deterministic and isolated from labels.
"""

from typing import List, Dict, Any, Union, Set, Optional
from urllib.parse import urlparse
import math
import re
import numpy as np
import pandas as pd


# Baseline whitelist of standard merchant, payments, and major partner domains
KNOWN_BENIGN_DOMAINS: Set[str] = {
    "store.merchanta.com",
    "checkout.merchanta.com",
    "merchanta.com",
    "merchantb.com",
    "store.merchantb.com",
    "merchantc.com",
    "supplier-direct.com",
    "shipping-logistics.net",
    "api.globalpay.com",
    "auth.paymentgateway.org",
    "invoicing-hub.net",
    "cloud-procure.io",
}

# Suspicious TLDs and regex patterns
SUSPICIOUS_TLDS: Set[str] = {".xyz", ".top", ".buzz", ".click", ".cam", ".work", ".ru", ".cx"}
IP_REGEX = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$")
DOC_EXTENSIONS: Set[str] = {".pdf", ".docx", ".xlsx", ".csv", ".txt", ".json", ".xml", ".doc"}
INSTRUCTION_KEYWORDS: Set[str] = {
    "instruction",
    "instructions",
    "prompt",
    "override",
    "directive",
    "execute",
    "task",
    "agent_command",
    "payload",
    "system_prompt",
}

PROVENANCE_FEATURE_NAMES: List[str] = [
    "provenance_count",
    "unique_domain_count",
    "unknown_domain_count",
    "provenance_length",
    "document_type_count",
    "instruction_source_present",
    "domain_entropy",
    "repeated_domain_count",
    "external_domain_count",
    "suspicious_source_indicator",
]


def calculate_string_entropy(text: str) -> float:
    """Calculate Shannon entropy for a string, normalized to [0, 1]."""
    if not text:
        return 0.0
    text_lower = text.lower()
    length = len(text_lower)
    counts: Dict[str, int] = {}
    for char in text_lower:
        counts[char] = counts.get(char, 0) + 1
    
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    
    # Normalize by max possible entropy for alphanumeric set (approx log2(36) ~ 5.17)
    max_entropy = math.log2(min(length, 36)) if length > 1 else 1.0
    return float(round(entropy / max(1e-5, max_entropy), 4))


class ProvenanceFeatureExtractor:
    """Extracts 10 deterministic, explainable features from a provenance chain."""

    def __init__(self, known_domains: Optional[Set[str]] = None):
        self.known_domains = known_domains or KNOWN_BENIGN_DOMAINS
        self.feature_names = list(PROVENANCE_FEATURE_NAMES)

    def extract_single(self, chain: Union[List[str], str, None], primary_merchant_domain: str = "merchanta.com") -> Dict[str, float]:
        """Extract features from a single provenance chain (list of URL strings)."""
        if chain is None:
            raw_urls: List[str] = []
        elif isinstance(chain, str):
            raw_urls = [u.strip() for u in chain.split(";") if u.strip()]
        elif isinstance(chain, (list, tuple)):
            raw_urls = [str(u).strip() for u in chain if str(u).strip()]
        else:
            raw_urls = []

        provenance_count = len(raw_urls)
        provenance_length = sum(len(u) for u in raw_urls)

        domains: List[str] = []
        unknown_domain_count = 0
        document_type_count = 0
        instruction_present = 0
        suspicious_indicator = 0
        external_domain_count = 0

        for url in raw_urls:
            url_str = url if "://" in url else f"https://{url}"
            try:
                parsed = urlparse(url_str)
                netloc = parsed.netloc.lower()
                path = parsed.path.lower()
                query = parsed.query.lower()
            except Exception:
                netloc = "unknown"
                path = url.lower()
                query = ""

            domains.append(netloc)

            # Check known vs unknown domain
            if netloc not in self.known_domains:
                unknown_domain_count += 1

            # Check external to primary merchant
            if primary_merchant_domain and not netloc.endswith(primary_merchant_domain):
                external_domain_count += 1

            # Document type detection
            if any(path.endswith(ext) for ext in DOC_EXTENSIONS):
                document_type_count += 1

            # Instruction / Prompt injection keyword detection in path or query
            combined_target = f"{path} {query}"
            if any(kw in combined_target for kw in INSTRUCTION_KEYWORDS):
                instruction_present = 1

            # Suspicious source indicator (raw IP, suspicious TLD, or port specified)
            if IP_REGEX.match(netloc):
                suspicious_indicator = 1
            if any(netloc.endswith(tld) for tld in SUSPICIOUS_TLDS):
                suspicious_indicator = 1
            if ":" in netloc and not netloc.startswith("http"):
                suspicious_indicator = 1

        unique_domains = set(domains)
        unique_domain_count = len(unique_domains)
        repeated_domain_count = max(0, provenance_count - unique_domain_count)

        # Domain entropy
        domain_concat = "".join(unique_domains)
        domain_entropy = calculate_string_entropy(domain_concat) if domain_concat else 0.0

        return {
            "provenance_count": float(provenance_count),
            "unique_domain_count": float(unique_domain_count),
            "unknown_domain_count": float(unknown_domain_count),
            "provenance_length": float(provenance_length),
            "document_type_count": float(document_type_count),
            "instruction_source_present": float(instruction_present),
            "domain_entropy": float(domain_entropy),
            "repeated_domain_count": float(repeated_domain_count),
            "external_domain_count": float(external_domain_count),
            "suspicious_source_indicator": float(suspicious_indicator),
        }

    def transform_series(
        self,
        chains: Union[pd.Series, List[Any]],
        primary_domains: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Extract features for an iterable of provenance chains, returning a DataFrame."""
        rows: List[Dict[str, float]] = []
        for i, chain in enumerate(chains):
            prim = primary_domains[i] if primary_domains and i < len(primary_domains) else "merchanta.com"
            feats = self.extract_single(chain, primary_merchant_domain=prim)
            rows.append(feats)
        return pd.DataFrame(rows, columns=self.feature_names)
