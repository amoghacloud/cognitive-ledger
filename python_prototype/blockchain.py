import time
import json
import copy
import math
from typing import Callable, List, Dict, Any, Optional
from pydantic import BaseModel, Field

from config import ChainConfig, SUPPORTED_TOKENS, SUPPORTED_TX_TYPES
from crypto_utils import hash_data, verify_signature
from neural_vm import NeuralModel, calculate_gas
from tokenomics import TokenomicsManager

class Transaction(BaseModel):
    sender: str
    receiver: str
    amount: float
    token_type: str = "FLOP"  # "FLOP", "DATA", "ATTN"
    tx_type: str = "TRANSFER"  # "TRANSFER", "REGISTER_MODEL", "INFER_CONTRACT", "STAKE_DATA"
    model_id: Optional[str] = None
    model_dna: Optional[str] = None
    input_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    output_dim: Optional[int] = None
    seed: Optional[int] = None
    dataset_id: Optional[str] = None
    inputs: Optional[List[float]] = None
    gas_limit: Optional[int] = None
    max_fee: Optional[float] = None
    attention_bid: Optional[float] = 0.0
    signature: Optional[str] = None
    nonce: int

    def to_signable_str(self) -> str:
        # Create a deterministic dictionary representation, excluding the signature
        if hasattr(self, "model_dump"):
            d = self.model_dump(exclude={"signature"})
        else:
            d = self.dict(exclude={"signature"})
        return json.dumps(d, sort_keys=True)

class Block(BaseModel):
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    validator: str
    proof_of_inference: Dict[str, Any] = Field(default_factory=dict)
    state_root: str
    nonce: int
    hash: str = ""

    def calculate_hash(self) -> str:
        if hasattr(self, "model_dump"):
            d = self.model_dump(exclude={"hash"})
        else:
            d = self.dict(exclude={"hash"})
        return hash_data(json.dumps(d, sort_keys=True))

class Blockchain:
    def __init__(
        self,
        config: Optional[ChainConfig] = None,
        genesis_balances: Optional[Dict[str, Dict[str, float]]] = None,
        on_commit: Optional[Callable[["Blockchain"], None]] = None,
    ):
        self.config = config or ChainConfig()
        if genesis_balances is None and self.config.founder_address:
            genesis_balances = {
                self.config.founder_address: self.config.founder_initial_balances
            }
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.tokenomics = TokenomicsManager(self.config, genesis_balances)
        if self.config.founder_address:
            self.tokenomics.validators[self.config.founder_address] = self.config.min_validator_stake
        self.registered_models: Dict[str, Dict[str, Any]] = {}
        self.account_nonces: Dict[str, int] = {}
        self.difficulty = self.config.difficulty
        self.on_commit = on_commit
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_tx = Transaction(
            sender="GENESIS",
            receiver="GENESIS",
            amount=0.0,
            token_type="FLOP",
            tx_type="TRANSFER",
            nonce=0,
            signature="GENESIS_SIG"
        )
        state_root = self.get_state_hash()
        genesis_block = Block(
            index=0,
            timestamp=0.0,
            transactions=[genesis_tx],
            previous_hash="0",
            validator="GENESIS",
            proof_of_inference={},
            state_root=state_root,
            nonce=0
        )
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)

    def select_leader(self, block_index: int, prev_hash: str, state_tokenomics: TokenomicsManager) -> str:
        """Selects a validator weighted by their staked FLOP tokens."""
        validators = state_tokenomics.validators
        if not validators:
            return self.config.founder_address or "GENESIS"
        
        sorted_validators = sorted(validators.keys())
        total_stake = sum(validators[v] for v in sorted_validators)
        if total_stake <= 0.0:
            return self.config.founder_address or "GENESIS"
            
        seed_hash = hash_data(f"{prev_hash}-{block_index}")
        seed_int = int(seed_hash[:16], 16)
        
        target = (seed_int % int(total_stake * 100)) / 100.0
        current = 0.0
        for val in sorted_validators:
            current += validators[val]
            if current >= target:
                return val
        return sorted_validators[-1]

    def set_commit_callback(self, on_commit: Optional[Callable[["Blockchain"], None]]):
        self.on_commit = on_commit

    def rebuild_from_chain(self, chain: List[Block]):
        if not chain:
            raise ValueError("Cannot rebuild from empty chain")

        genesis = Blockchain(config=self.config)
        expected_genesis = genesis.chain[0]
        if chain[0].hash != expected_genesis.hash:
            raise ValueError("Genesis block does not match this chain configuration")

        candidate = Blockchain(config=self.config)
        for block in chain[1:]:
            if not candidate.validate_block(block, persist=False):
                raise ValueError(f"Invalid block at index {block.index}")

        self.chain = candidate.chain
        self.tokenomics = candidate.tokenomics
        self.registered_models = candidate.registered_models
        self.account_nonces = candidate.account_nonces

    def get_state_hash(self) -> str:
        """Computes a cryptographic fingerprint of the ledger state."""
        state_repr = {
            "balances": self.tokenomics.balances,
            "data_equity": self.tokenomics.data_equity,
            "model_datasets": self.tokenomics.model_datasets,
            "registered_models": self.registered_models,
            "account_nonces": self.account_nonces
        }
        return hash_data(json.dumps(state_repr, sort_keys=True))

    def add_transaction(self, tx: Transaction) -> bool:
        """Validates and adds a transaction to the memory pool."""
        if len(self.pending_transactions) >= self.config.max_mempool_size:
            return False
        if not self.validate_transaction_shape(tx):
            return False

        # 1. Signature Check (except for Genesis)
        if tx.sender != "GENESIS":
            if not tx.signature:
                return False
            if not verify_signature(tx.sender, tx.signature, tx.to_signable_str()):
                return False

        # 2. Nonce Check
        current_nonce = self.account_nonces.get(tx.sender, 0)
        if tx.nonce <= current_nonce:
            return False

        # 3. Balance Check
        self.tokenomics.ensure_balance_record(tx.sender)
        
        if tx.tx_type == "TRANSFER":
            if self.tokenomics.get_balance(tx.sender, tx.token_type) < tx.amount:
                return False
        elif tx.tx_type == "REGISTER_MODEL":
            # Registration fee of 10 $FLOP
            if self.tokenomics.get_balance(tx.sender, "FLOP") < 10.0:
                return False
        elif tx.tx_type == "INFER_CONTRACT":
            # Must cover max gas cost and attention bid
            required_flop = (tx.gas_limit or 100) * (tx.max_fee or 1.0)
            required_attn = tx.attention_bid or 0.0
            if self.tokenomics.get_balance(tx.sender, "FLOP") < required_flop:
                return False
            if self.tokenomics.get_balance(tx.sender, "ATTN") < required_attn:
                return False
        elif tx.tx_type == "STAKE_DATA":
            if self.tokenomics.get_balance(tx.sender, "DATA") < tx.amount:
                return False
        elif tx.tx_type == "STAKE_VALIDATOR":
            if self.tokenomics.get_balance(tx.sender, "FLOP") < tx.amount:
                return False
        elif tx.tx_type == "UNSTAKE_VALIDATOR":
            staked = self.tokenomics.validators.get(tx.sender, 0.0)
            if staked < tx.amount:
                return False

        # 4. Check for duplicate pending txs
        for p_tx in self.pending_transactions:
            if p_tx.signature == tx.signature:
                return False
            if p_tx.sender == tx.sender and p_tx.nonce == tx.nonce:
                return False

        self.pending_transactions.append(tx)
        return True

    def validate_transaction_shape(self, tx: Transaction) -> bool:
        if tx.tx_type not in SUPPORTED_TX_TYPES or tx.token_type not in SUPPORTED_TOKENS:
            return False
        if not isinstance(tx.nonce, int) or tx.nonce <= 0:
            return False
        if not self._is_non_negative_number(tx.amount):
            return False
        if tx.attention_bid is not None and not self._is_non_negative_number(tx.attention_bid):
            return False

        if tx.tx_type == "TRANSFER":
            return bool(tx.receiver) and self._is_positive_number(tx.amount)

        if tx.tx_type == "REGISTER_MODEL":
            dims = [tx.input_dim, tx.hidden_dim, tx.output_dim]
            if not tx.model_id or not tx.model_dna or tx.seed is None:
                return False
            if any(not isinstance(dim, int) or dim <= 0 or dim > self.config.max_model_dimension for dim in dims):
                return False
            return True

        if tx.tx_type == "INFER_CONTRACT":
            if not tx.model_id or not tx.inputs:
                return False
            if len(tx.inputs) > self.config.max_input_length:
                return False
            if tx.gas_limit is None or not isinstance(tx.gas_limit, int) or tx.gas_limit <= 0:
                return False
            if tx.max_fee is None or not self._is_positive_number(tx.max_fee):
                return False
            return all(self._is_finite_number(value) for value in tx.inputs)

        if tx.tx_type == "STAKE_DATA":
            return bool(tx.dataset_id) and self._is_positive_number(tx.amount)
        if tx.tx_type == "STAKE_VALIDATOR":
            return tx.token_type == "FLOP" and tx.amount >= self.config.min_validator_stake
        if tx.tx_type == "UNSTAKE_VALIDATOR":
            return tx.token_type == "FLOP" and self._is_positive_number(tx.amount)
        return False

    def execute_transaction_on_state(
        self, 
        tx: Transaction, 
        state_tokenomics: TokenomicsManager, 
        state_models: Dict[str, Any],
        state_nonces: Dict[str, int],
        validator_address: str
    ) -> Dict[str, Any]:
        """
        Executes a transaction against local state variables, updating them in place.
        Returns execution results (used for proof_of_inference).
        """
        # Ensure sender/receiver records exist
        state_tokenomics.ensure_balance_record(tx.sender)
        if tx.receiver:
            state_tokenomics.ensure_balance_record(tx.receiver)
        
        result = {}

        if tx.tx_type == "TRANSFER":
            success = state_tokenomics.transfer(tx.sender, tx.receiver, tx.amount, tx.token_type)
            if not success:
                raise ValueError("Insufficient balance for transfer")

        elif tx.tx_type == "REGISTER_MODEL":
            if tx.model_id in state_models:
                raise ValueError("Model already registered")
            # Deduct 10 $FLOP fee for registration, route to validator
            success = state_tokenomics.transfer(
                tx.sender, validator_address, self.config.model_registration_fee, "FLOP"
            )
            if not success:
                raise ValueError("Insufficient balance for model registration fee")
            
            # Register details
            state_models[tx.model_id] = {
                "model_id": tx.model_id,
                "model_dna": tx.model_dna,
                "input_dim": tx.input_dim,
                "hidden_dim": tx.hidden_dim,
                "output_dim": tx.output_dim,
                "seed": tx.seed,
                "dataset_id": tx.dataset_id
            }
            if tx.dataset_id:
                state_tokenomics.bind_model_to_dataset(tx.model_id, tx.dataset_id)

        elif tx.tx_type == "INFER_CONTRACT":
            model_info = state_models.get(tx.model_id)
            if not model_info:
                raise ValueError(f"Model {tx.model_id} not registered")

            # Initialize NeuralVM with deterministic parameters
            model = NeuralModel(
                model_id=model_info["model_id"],
                seed=model_info["seed"],
                input_dim=model_info["input_dim"],
                hidden_dim=model_info["hidden_dim"],
                output_dim=model_info["output_dim"]
            )
            
            # Execute quantized inference
            import numpy as np
            inputs_array = np.array(tx.inputs, dtype=np.float32)
            logits, confidence, bounds = model.forward(inputs_array, quantize=True)
            
            # Calculate gas
            gas_used = calculate_gas(len(tx.inputs), model, context_len=1, vram_time=0.1)
            if tx.gas_limit and gas_used > tx.gas_limit:
                raise ValueError("Gas limit exceeded")

            # Calculate total fee
            total_fee = gas_used * (tx.max_fee or 1.0)
            
            # Deduct gas fee from sender
            if state_tokenomics.get_balance(tx.sender, "FLOP") < total_fee:
                raise ValueError("Insufficient FLOP balance for gas")
            state_tokenomics.balances[tx.sender]["FLOP"] -= total_fee
            
            # Route royalties to DET holders
            validator_share = state_tokenomics.route_execution_royalties(tx.model_id, total_fee, validator_address)
            # Pay validator
            state_tokenomics.balances[validator_address]["FLOP"] += validator_share

            # Handle attention priority bid if non-zero
            if tx.attention_bid and tx.attention_bid > 0.0:
                if state_tokenomics.get_balance(tx.sender, "ATTN") < tx.attention_bid:
                    raise ValueError("Insufficient ATTN balance for bid")
                state_tokenomics.balances[tx.sender]["ATTN"] -= tx.attention_bid
                state_tokenomics.balances[validator_address]["ATTN"] += tx.attention_bid

            result = {
                "logits": logits,
                "confidence": confidence,
                "uncertainty_bounds": bounds,
                "gas_used": gas_used,
                "success": True
            }

        elif tx.tx_type == "STAKE_DATA":
            success = state_tokenomics.stake_data_equity(tx.sender, tx.dataset_id, tx.amount)
            if not success:
                raise ValueError("Insufficient DATA balance for staking")
        elif tx.tx_type == "STAKE_VALIDATOR":
            success = state_tokenomics.stake_validator(tx.sender, tx.amount)
            if not success:
                raise ValueError("Insufficient FLOP balance for validator staking")
        elif tx.tx_type == "UNSTAKE_VALIDATOR":
            success = state_tokenomics.unstake_validator(tx.sender, tx.amount)
            if not success:
                raise ValueError("Insufficient staked balance for validator unstaking")

        # Update nonce
        state_nonces[tx.sender] = tx.nonce
        return result

    def mine_block(self, validator_address: str) -> Optional[Block]:
        """
        Assembles pending transactions, runs VCG auction priority sorting,
        executes the neural VMs, updates local state ledger, and proposes a new PoS block.
        """
        if not self.pending_transactions:
            return None

        # 1. Run Attention Priority Auction
        bids = []
        for tx in self.pending_transactions:
            bids.append({
                "address": tx.sender,
                "bid": tx.attention_bid or 0.0,
                "tx_id": tx.signature
            })
        
        sorted_bids = self.tokenomics.run_vcg_attention_auction(bids)
        priority_sigs = [b["tx_id"] for b in sorted_bids]
        
        # Sort pending txs: priority ones first, then remaining
        priority_txs = []
        other_txs = []
        for tx in self.pending_transactions:
            if tx.signature in priority_sigs:
                priority_txs.append(tx)
            else:
                other_txs.append(tx)
        
        # Sort priority transactions in the order determined by auction
        priority_txs.sort(key=lambda x: priority_sigs.index(x.signature))
        candidate_txs = (priority_txs + other_txs)[:self.config.max_transactions_per_block]

        # Clone current state to execute transactions atomically
        temp_tokenomics = copy.deepcopy(self.tokenomics)
        temp_models = copy.deepcopy(self.registered_models)
        temp_nonces = copy.deepcopy(self.account_nonces)

        executed_txs = []
        proof_of_inference = {}

        for tx in candidate_txs:
            try:
                inf_result = self.execute_transaction_on_state(
                    tx, temp_tokenomics, temp_models, temp_nonces, validator_address
                )
                executed_txs.append(tx)
                if tx.tx_type == "INFER_CONTRACT":
                    proof_of_inference[tx.signature] = inf_result
            except Exception as e:
                # Discard invalid transactions during execution
                print(f"[Block Miner] Tx {tx.signature[:8]} execution failed: {e}")
                continue

        if not executed_txs:
            # Clear invalid pending txs that failed execution
            for tx in candidate_txs:
                if tx in self.pending_transactions:
                    self.pending_transactions.remove(tx)
            return None

        # Create Block structure
        prev_block = self.chain[-1]
        
        # Verify slot leader
        expected_leader = self.select_leader(prev_block.index + 1, prev_block.hash, self.tokenomics)
        if validator_address != expected_leader:
            raise ValueError(f"Not slot leader. Leader: {expected_leader[:16]}...")
        
        # Update self stats to compute temporary state root
        self.tokenomics = temp_tokenomics
        self.registered_models = temp_models
        self.account_nonces = temp_nonces
        state_root = self.get_state_hash()

        new_block = Block(
            index=prev_block.index + 1,
            timestamp=time.time(),
            transactions=executed_txs,
            previous_hash=prev_block.hash,
            validator=validator_address,
            proof_of_inference=proof_of_inference,
            state_root=state_root,
            nonce=0
        )

        new_block.hash = new_block.calculate_hash()

        # Remove executed transactions from pending memory pool
        for tx in executed_txs:
            if tx in self.pending_transactions:
                self.pending_transactions.remove(tx)

        # Append to blockchain
        self.chain.append(new_block)
        self._commit()
        return new_block

    def validate_block(self, block: Block, persist: bool = True) -> bool:
        """
        Validates block fields, signatures, slot leader checks,
        Proof-of-Inference neural logic execution, and triggers slashing if invalid.
        """
        # 1. Structure Check
        if not self.chain:
            return False
        prev_block = self.chain[-1]
        
        if block.index != prev_block.index + 1:
            return False
        if block.previous_hash != prev_block.hash:
            return False
        if block.hash != block.calculate_hash():
            return False
            
        # Verify slot leader
        expected_leader = self.select_leader(block.index, prev_block.hash, self.tokenomics)
        if block.validator != expected_leader:
            return False

        # 2. Validate transactions and recreate state updates
        temp_tokenomics = copy.deepcopy(self.tokenomics)
        temp_models = copy.deepcopy(self.registered_models)
        temp_nonces = copy.deepcopy(self.account_nonces)

        for tx in block.transactions:
            if not self.validate_transaction_shape(tx):
                # Slash validator for proposing invalid transaction
                self.tokenomics.slash_validator(block.validator, self.config.validator_slashing_rate)
                self._commit()
                return False
            # Verify signature
            if tx.sender != "GENESIS":
                if not tx.signature or not verify_signature(tx.sender, tx.signature, tx.to_signable_str()):
                    self.tokenomics.slash_validator(block.validator, self.config.validator_slashing_rate)
                    self._commit()
                    return False

            # Verify nonce
            curr_nonce = temp_nonces.get(tx.sender, 0)
            if tx.nonce <= curr_nonce:
                self.tokenomics.slash_validator(block.validator, self.config.validator_slashing_rate)
                self._commit()
                return False

            # Execute transaction on temp state
            try:
                res = self.execute_transaction_on_state(
                    tx, temp_tokenomics, temp_models, temp_nonces, block.validator
                )
                # Verify Proof-of-Inference outputs match EXACTLY (quantized determinism)
                if tx.tx_type == "INFER_CONTRACT":
                    recorded_res = block.proof_of_inference.get(tx.signature)
                    if not recorded_res:
                        self.tokenomics.slash_validator(block.validator, self.config.validator_slashing_rate)
                        self._commit()
                        return False
                    # Compare logits, confidence, and bounds (within tolerances or exactly)
                    if recorded_res.get("logits") != res["logits"]:
                        self.tokenomics.slash_validator(block.validator, self.config.validator_slashing_rate)
                        self._commit()
                        return False
                    if recorded_res.get("confidence") != res["confidence"]:
                        self.tokenomics.slash_validator(block.validator, self.config.validator_slashing_rate)
                        self._commit()
                        return False
                    if recorded_res.get("uncertainty_bounds") != res["uncertainty_bounds"]:
                        self.tokenomics.slash_validator(block.validator, self.config.validator_slashing_rate)
                        self._commit()
                        return False
            except Exception as e:
                print(f"[Block Validator] Transaction validation failure: {e}")
                self.tokenomics.slash_validator(block.validator, self.config.validator_slashing_rate)
                self._commit()
                return False

        # 3. Check State Root Hash
        state_repr = {
            "balances": temp_tokenomics.balances,
            "data_equity": temp_tokenomics.data_equity,
            "model_datasets": temp_tokenomics.model_datasets,
            "registered_models": temp_models,
            "account_nonces": temp_nonces
        }
        computed_state_root = hash_data(json.dumps(state_repr, sort_keys=True))
        if computed_state_root != block.state_root:
            self.tokenomics.slash_validator(block.validator, self.config.validator_slashing_rate)
            self._commit()
            return False

        # Apply state changes permanently
        self.tokenomics = temp_tokenomics
        self.registered_models = temp_models
        self.account_nonces = temp_nonces
        self.chain.append(block)
        if persist:
            self.pending_transactions = [
                tx for tx in self.pending_transactions
                if tx.signature not in {btx.signature for btx in block.transactions}
            ]
            self._commit()
        return True

    def _commit(self):
        if self.on_commit:
            self.on_commit(self)

    def _is_finite_number(self, value: float) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value)

    def _is_non_negative_number(self, value: float) -> bool:
        return self._is_finite_number(value) and value >= 0

    def _is_positive_number(self, value: float) -> bool:
        return self._is_finite_number(value) and value > 0
