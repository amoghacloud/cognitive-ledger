import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from blockchain import Blockchain, Block, Transaction


class ChainStorage:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self, blockchain: Blockchain) -> bool:
        if not self.path.exists():
            return False

        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("chain_id") != blockchain.config.chain_id:
            raise ValueError(
                f"Chain id mismatch: store={data.get('chain_id')} node={blockchain.config.chain_id}"
            )

        chain = []
        for block_data in data.get("chain", []):
            txs = [Transaction(**tx_data) for tx_data in block_data.get("transactions", [])]
            block = Block(
                index=block_data["index"],
                timestamp=block_data["timestamp"],
                transactions=txs,
                previous_hash=block_data["previous_hash"],
                validator=block_data["validator"],
                proof_of_inference=block_data.get("proof_of_inference", {}),
                state_root=block_data["state_root"],
                nonce=block_data["nonce"],
                hash=block_data["hash"],
            )
            chain.append(block)

        blockchain.rebuild_from_chain(chain)
        blockchain.pending_transactions = [
            Transaction(**tx_data) for tx_data in data.get("pending_transactions", [])
        ]
        return True

    def save(self, blockchain: Blockchain):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chain_id": blockchain.config.chain_id,
            "state_hash": blockchain.get_state_hash(),
            "chain": [
                block.model_dump() if hasattr(block, "model_dump") else block.dict()
                for block in blockchain.chain
            ],
            "pending_transactions": [
                tx.model_dump() if hasattr(tx, "model_dump") else tx.dict()
                for tx in blockchain.pending_transactions
            ],
        }

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
