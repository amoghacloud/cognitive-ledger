import math
from typing import Dict, Optional

from config import ChainConfig, SUPPORTED_TOKENS


class TokenomicsManager:
    def __init__(
        self,
        config: Optional[ChainConfig] = None,
        genesis_balances: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.config = config or ChainConfig()
        self.balances = {}
        self.data_equity = {}
        self.model_datasets = {}

        if genesis_balances:
            for address, balances in genesis_balances.items():
                self.set_balance_record(address, balances)

    def set_balance_record(self, address: str, balances: Dict[str, float]):
        if not address:
            raise ValueError("Address cannot be empty")
        self.balances[address] = {token: 0.0 for token in SUPPORTED_TOKENS}
        for token, amount in balances.items():
            self._validate_token(token)
            self._validate_non_negative_amount(amount)
            self.balances[address][token] = float(amount)

    def ensure_balance_record(self, address: str):
        if not address:
            return
        if address not in self.balances:
            if not self.config.allow_dev_faucet:
                self.balances[address] = {token: 0.0 for token in SUPPORTED_TOKENS}
                return
            self.balances[address] = {
                token: float(self.config.dev_initial_balances.get(token, 0.0))
                for token in SUPPORTED_TOKENS
            }

    def get_balance(self, address: str, token_type: str = "FLOP") -> float:
        self._validate_token(token_type)
        self.ensure_balance_record(address)
        return self.balances.get(address, {}).get(token_type, 0.0)

    def transfer(self, sender: str, receiver: str, amount: float, token_type: str = "FLOP") -> bool:
        self._validate_token(token_type)
        if not self._is_positive_amount(amount):
            return False

        self.ensure_balance_record(sender)
        self.ensure_balance_record(receiver)

        if self.balances[sender][token_type] < amount:
            return False

        self.balances[sender][token_type] -= amount
        self.balances[receiver][token_type] += amount
        return True

    def register_dataset_equity(self, dataset_id: str):
        if not dataset_id:
            raise ValueError("Dataset id cannot be empty")
        if dataset_id not in self.data_equity:
            self.data_equity[dataset_id] = {"stakers": {}, "total_staked": 0.0}

    def stake_data_equity(self, address: str, dataset_id: str, amount: float) -> bool:
        if not self._is_positive_amount(amount):
            return False
        self.ensure_balance_record(address)
        self.register_dataset_equity(dataset_id)

        if self.balances[address]["DATA"] < amount:
            return False

        self.balances[address]["DATA"] -= amount
        stakers = self.data_equity[dataset_id]["stakers"]
        stakers[address] = stakers.get(address, 0.0) + amount
        self.data_equity[dataset_id]["total_staked"] += amount
        return True

    def bind_model_to_dataset(self, model_id: str, dataset_id: str):
        if not model_id or not dataset_id:
            raise ValueError("Model id and dataset id are required")
        self.model_datasets[model_id] = dataset_id

    def route_execution_royalties(self, model_id: str, total_gas_paid: float, fee_recipient: str) -> float:
        self._validate_non_negative_amount(total_gas_paid)
        dataset_id = self.model_datasets.get(model_id)
        if not dataset_id or dataset_id not in self.data_equity:
            return total_gas_paid

        equity_pool = self.data_equity[dataset_id]
        total_staked = equity_pool["total_staked"]
        if total_staked <= 0.0:
            return total_gas_paid

        royalty_amount = total_gas_paid * self.config.royalty_rate
        validator_cut = total_gas_paid - royalty_amount

        for staker, staked_val in equity_pool["stakers"].items():
            self.ensure_balance_record(staker)
            share_ratio = staked_val / total_staked
            self.balances[staker]["FLOP"] += royalty_amount * share_ratio

        return validator_cut

    def run_vcg_attention_auction(self, bids: list) -> list:
        valid_bids = []
        for bid in bids:
            addr = bid["address"]
            amount = bid.get("bid", 0.0)
            self.ensure_balance_record(addr)
            if self._is_non_negative_amount(amount) and self.balances[addr]["ATTN"] >= amount:
                valid_bids.append(bid)

        valid_bids.sort(key=lambda x: (-x.get("bid", 0.0), x.get("tx_id", "")))
        return valid_bids

    def _validate_token(self, token_type: str):
        if token_type not in SUPPORTED_TOKENS:
            raise ValueError(f"Unsupported token type: {token_type}")

    def _validate_non_negative_amount(self, amount: float):
        if not self._is_non_negative_amount(amount):
            raise ValueError("Amount must be finite and non-negative")

    def _is_positive_amount(self, amount: float) -> bool:
        return isinstance(amount, (int, float)) and math.isfinite(amount) and amount > 0

    def _is_non_negative_amount(self, amount: float) -> bool:
        return isinstance(amount, (int, float)) and math.isfinite(amount) and amount >= 0
