from dataclasses import dataclass, field
from typing import Dict, Optional


SUPPORTED_TOKENS = {"FLOP", "DATA", "ATTN"}
SUPPORTED_TX_TYPES = {"TRANSFER", "REGISTER_MODEL", "INFER_CONTRACT", "STAKE_DATA", "STAKE_VALIDATOR", "UNSTAKE_VALIDATOR"}
DEFAULT_FOUNDER_ADDRESS = """-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEYQlPKanrlObeW0lO0TPOENPMenj/
RtKRycB+KHHuXF+CCmV7+31AshaslqeyC32PNY/TP2Wk+xBC07bruRYBDQ==
-----END PUBLIC KEY-----
"""


@dataclass(frozen=True)
class ChainConfig:
    chain_id: str = "cognitive-ledger-localnet-v1"
    difficulty: int = 2
    max_transactions_per_block: int = 10
    max_mempool_size: int = 1000
    max_model_dimension: int = 4096
    max_input_length: int = 4096
    model_registration_fee: float = 10.0
    royalty_rate: float = 0.10
    min_validator_stake: float = 1000.0
    validator_slashing_rate: float = 0.50
    founder_address: Optional[str] = DEFAULT_FOUNDER_ADDRESS
    founder_initial_balances: Dict[str, float] = field(
        default_factory=lambda: {"FLOP": 15000000.0, "DATA": 1500000.0, "ATTN": 1500000.0}
    )
    allow_dev_faucet: bool = False
    dev_initial_balances: Dict[str, float] = field(
        default_factory=lambda: {"FLOP": 10000.0, "DATA": 100.0, "ATTN": 10.0}
    )
