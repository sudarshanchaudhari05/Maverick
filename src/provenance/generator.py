"""FraudForge AI: Synthetic Provenance Chain Generator (Label-Independent).

Generates realistic, deterministic provenance chains for agentic payments.
Strict Methodological Constraint:
Provenance generation is strictly INDEPENDENT of fraud labels and target metadata.
Neither 'fraud_label', 'is_fraud', 'attack_type', nor any other target column
is accepted or accessed during generation.

Supports predetermined independent mixture sampling:
- Benign chains (standard shopping, checkout, vendor documentation)
- Suspicious chains (unknown domains, unusual documents, unexpected transitions)
- Adversarial / Poisoned chains (prompt injections, directive overrides, malicious instructions)

Hard negatives (legitimate transactions with unusual/poisoned context) and
deceptive positives (fraudulent transactions with clean/benign context) emerge
naturally from independent joint distribution sampling.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.provenance.schema import ProvenanceChain


# Pools of synthetic URLs by category
BENIGN_PRODUCT_URLS = [
    "https://store.merchanta.com/catalog/electronics/item-4921",
    "https://store.merchanta.com/products/office-supplies/desk-ergonomic",
    "https://merchanta.com/items/sku-99210",
    "https://merchantb.com/apparel/outerwear/jacket-winter-88",
    "https://merchantc.com/groceries/pantry/coffee-organic-1kg",
    "https://store.merchanta.com/search?q=usb-c+docking+station",
]

BENIGN_DOC_URLS = [
    "https://store.merchanta.com/policies/terms_of_service.pdf",
    "https://merchanta.com/docs/shipping_and_returns.pdf",
    "https://supplier-direct.com/invoices/inv-2026-0812.pdf",
    "https://shipping-logistics.net/tracking/package-info.docx",
    "https://invoicing-hub.net/receipts/rec-99120.pdf",
    "https://cloud-procure.io/contracts/enterprise_license.docx",
]

BENIGN_CHECKOUT_URLS = [
    "https://checkout.merchanta.com/cart/review",
    "https://checkout.merchanta.com/payment/methods",
    "https://auth.paymentgateway.org/verify/session-token-9812",
    "https://api.globalpay.com/v2/hosted-fields/auth",
]

SUSPICIOUS_EXTERNAL_URLS = [
    "https://unverified-discount-portal.xyz/coupons/promo_code_99",
    "https://quick-invoice-download.top/view/invoice-temp.doc",
    "https://external-escrow-routing.buzz/settlement/review",
    "https://198.51.100.42:8443/direct_pay/gateway",
    "https://proxy-supplier-exchange.click/catalog/urgent_item",
    "https://procurement-redirect.work/dispatch/order_summary.xlsx",
]

ADVERSARIAL_INJECTION_URLS = [
    "https://malicious-context-relay.top/task?override_instruction=authorize_max_limit",
    "https://agent-manipulator.xyz/prompt/system_prompt_override.txt",
    "https://injected-directive-hub.buzz/payload/agent_command_execute.json",
    "https://203.0.113.88:9090/instruction/bypass_compliance_directive",
    "https://external-exploit-source.click/directive/force_high_velocity_payment",
]

# Fixed predetermined mixture proportions (frozen across primary evaluation)
DEFAULT_PROVENANCE_MIXTURE: Dict[str, float] = {
    "benign": 0.50,
    "suspicious": 0.25,
    "poisoned": 0.25,
}


class SyntheticProvenanceGenerator:
    """Generates synthetic provenance chains for payment transactions with seed determinism.
    
    Guaranteed Label Independence:
    Neither fraud labels, targets, nor attack types are ever consumed by this generator.
    """

    def __init__(self, seed: int = 4001):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_benign_chain(self) -> ProvenanceChain:
        """Generate a realistic benign shopping or procurement provenance chain."""
        chain_len = int(self.rng.integers(2, 5))
        urls: List[str] = [str(self.rng.choice(BENIGN_PRODUCT_URLS))]
        if self.rng.random() < 0.4:
            urls.append(str(self.rng.choice(BENIGN_DOC_URLS)))
        while len(urls) < chain_len:
            urls.append(str(self.rng.choice(BENIGN_CHECKOUT_URLS)))
        return ProvenanceChain(sources=urls, provenance_type="benign")

    def generate_suspicious_chain(self) -> ProvenanceChain:
        """Generate a provenance chain containing unusual external domains or suspicious documents."""
        urls: List[str] = [str(self.rng.choice(BENIGN_PRODUCT_URLS))]
        num_susp = int(self.rng.integers(1, 3))
        for _ in range(num_susp):
            urls.append(str(self.rng.choice(SUSPICIOUS_EXTERNAL_URLS)))
        if self.rng.random() < 0.5:
            urls.append(str(self.rng.choice(BENIGN_CHECKOUT_URLS)))
        return ProvenanceChain(sources=urls, provenance_type="suspicious")

    def generate_adversarial_poisoned_chain(self) -> ProvenanceChain:
        """Generate a provenance chain containing explicit agentic prompt injection / directives."""
        urls: List[str] = [str(self.rng.choice(BENIGN_PRODUCT_URLS))]
        urls.append(str(self.rng.choice(ADVERSARIAL_INJECTION_URLS)))
        if self.rng.random() < 0.5:
            urls.append(str(self.rng.choice(SUSPICIOUS_EXTERNAL_URLS)))
        urls.append(str(self.rng.choice(BENIGN_CHECKOUT_URLS)))
        return ProvenanceChain(sources=urls, provenance_type="poisoned")

    def generate_chain_by_type(self, provenance_type: str) -> ProvenanceChain:
        """Generate a chain corresponding to the requested category."""
        if provenance_type == "benign":
            return self.generate_benign_chain()
        elif provenance_type == "suspicious":
            return self.generate_suspicious_chain()
        elif provenance_type == "poisoned":
            return self.generate_adversarial_poisoned_chain()
        else:
            raise ValueError(f"Unknown provenance_type: {provenance_type}")

    def attach_provenance_to_dataset(
        self,
        df: pd.DataFrame,
        mixture: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Attach provenance chains strictly independently of transaction labels using fixed mixture weights.
        
        Args:
            df: Input DataFrame. Target columns (fraud_label, attack_type) are strictly NOT accessed.
            mixture: Predetermined mixture proportions, e.g. {'benign': 0.50, 'suspicious': 0.25, 'poisoned': 0.25}.
        
        Returns:
            DataFrame with 'provenance_chain' (list of strings) and 'provenance_category' columns.
        """
        df_out = df.copy().reset_index(drop=True)
        mix = mixture or DEFAULT_PROVENANCE_MIXTURE
        types = list(mix.keys())
        probs = np.array([mix[t] for t in types], dtype=np.float64)
        probs = probs / probs.sum()

        n_rows = len(df_out)
        sampled_types = self.rng.choice(types, size=n_rows, p=probs)

        chains: List[List[str]] = []
        categories: List[str] = []

        for prov_type in sampled_types:
            chain = self.generate_chain_by_type(str(prov_type))
            chains.append(chain.to_list())
            categories.append(str(prov_type))

        df_out["provenance_chain"] = chains
        df_out["provenance_category"] = categories
        return df_out
