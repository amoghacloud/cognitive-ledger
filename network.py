import threading
import json
import requests
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from config import ChainConfig
from blockchain import Blockchain, Transaction, Block
from storage import ChainStorage

class P2PNode:
    def __init__(
        self,
        port: int,
        peers: list,
        validator_address: str,
        config: ChainConfig = None,
        storage_path: str = None,
    ):
        self.port = port
        self.peers = set()
        self.validator_address = validator_address
        self.config = config or ChainConfig()
        self.storage = ChainStorage(storage_path) if storage_path else None
        self.blockchain = Blockchain(config=self.config, on_commit=self.persist)
        if self.storage:
            self.storage.load(self.blockchain)
            self.blockchain.set_commit_callback(self.persist)
        
        # Keep track of recently gossiped items to avoid loops
        self.seen_transactions = set()
        self.seen_blocks = set()

        for peer in peers:
            self.register_peer(peer)
            
        self.blockchain.tokenomics.ensure_balance_record(validator_address)
        self.persist()

    def persist(self, blockchain: Blockchain = None):
        if self.storage:
            self.storage.save(blockchain or self.blockchain)

    def register_peer(self, peer: str):
        if not peer:
            return
        if not peer.startswith("http"):
            peer = f"http://{peer}"
        # Parse peer url to check validity and clean trailing slashes
        parsed = urlparse(peer)
        clean_peer = f"{parsed.scheme}://{parsed.netloc}"
        if clean_peer != f"http://localhost:{self.port}" and clean_peer != f"http://127.0.0.1:{self.port}":
            self.peers.add(clean_peer)

    def gossip_transaction(self, tx: Transaction):
        if tx.signature in self.seen_transactions:
            return
        self.seen_transactions.add(tx.signature)
        
        data = tx.model_dump() if hasattr(tx, "model_dump") else tx.dict()
        for peer in list(self.peers):
            try:
                requests.post(f"{peer}/transactions/new", json=data, timeout=2)
            except Exception:
                pass

    def gossip_block(self, block: Block):
        if block.hash in self.seen_blocks:
            return
        self.seen_blocks.add(block.hash)
        
        data = block.model_dump() if hasattr(block, "model_dump") else block.dict()
        for peer in list(self.peers):
            try:
                requests.post(f"{peer}/blocks/new", json=data, timeout=2)
            except Exception:
                pass

    def sync_chain(self) -> bool:
        """Fetches chains from peers and adopts the longest valid chain."""
        longest_chain = None
        current_len = len(self.blockchain.chain)
        
        for peer in list(self.peers):
            try:
                response = requests.get(f"{peer}/chain", timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    chain_data = data.get("chain", [])
                    if len(chain_data) > current_len:
                        # Validate the incoming chain from scratch
                        candidate = Blockchain()
                        candidate.chain = []  # clear genesis to populate
                        valid = True
                        
                        for b_dict in chain_data:
                            txs = []
                            for tx_dict in b_dict.get("transactions", []):
                                txs.append(Transaction(**tx_dict))
                            
                            b = Block(
                                index=b_dict["index"],
                                timestamp=b_dict["timestamp"],
                                transactions=txs,
                                previous_hash=b_dict["previous_hash"],
                                validator=b_dict["validator"],
                                proof_of_inference=b_dict.get("proof_of_inference", {}),
                                state_root=b_dict["state_root"],
                                nonce=b_dict["nonce"],
                                hash=b_dict["hash"]
                            )
                            
                            if b.index == 0:
                                candidate.chain.append(b)
                            else:
                                if not candidate.validate_block(b, persist=False):
                                    valid = False
                                    break
                                    
                        if valid and len(candidate.chain) > current_len:
                            longest_chain = candidate.chain
                            current_len = len(candidate.chain)
                            # Copy state over
                            self.blockchain.chain = candidate.chain
                            self.blockchain.tokenomics = candidate.tokenomics
                            self.blockchain.registered_models = candidate.registered_models
                            self.blockchain.account_nonces = candidate.account_nonces
                            self.persist()
            except Exception as e:
                print(f"[P2P Sync] Sync failure with peer {peer}: {e}")
                
        if longest_chain:
            print(f"[P2P Sync] Synced longest chain. Blocks: {len(self.blockchain.chain)}")
            return True
        return False

    def connect_to_network(self):
        """Registers self with existing network peers and pulls historical chain."""
        my_address = f"http://localhost:{self.port}"
        for peer in list(self.peers):
            try:
                # Register self on peer
                requests.post(f"{peer}/peers/register", json={"peer": my_address}, timeout=2)
                # Pull other peers from peer
                resp = requests.get(f"{peer}/peers", timeout=2)
                if resp.status_code == 200:
                    network_peers = resp.json().get("peers", [])
                    for np in network_peers:
                        self.register_peer(np)
            except Exception:
                pass
        self.sync_chain()

def make_handler(node: P2PNode):
    class P2PRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress internal server outputs to keep visual CLI clean

        def send_json(self, status: int, data: dict):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        def do_GET(self):
            parsed_url = urlparse(self.path)
            path = parsed_url.path

            if path == "/chain":
                chain_serialized = []
                for block in node.blockchain.chain:
                    chain_serialized.append(block.model_dump() if hasattr(block, "model_dump") else block.dict())
                self.send_json(200, {"chain": chain_serialized, "length": len(chain_serialized)})

            elif path == "/peers":
                self.send_json(200, {"peers": list(node.peers)})

            elif path == "/state":
                self.send_json(200, {
                    "state_hash": node.blockchain.get_state_hash(),
                    "registered_models": node.blockchain.registered_models,
                    "account_nonces": node.blockchain.account_nonces,
                    "balances": node.blockchain.tokenomics.balances
                })

            elif path == "/balances":
                self.send_json(200, {
                    "balances": node.blockchain.tokenomics.balances
                })
            else:
                self.send_json(404, {"error": "Not Found"})

        def do_POST(self):
            parsed_url = urlparse(self.path)
            path = parsed_url.path

            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            try:
                data = json.loads(post_data) if post_data else {}
            except Exception:
                self.send_json(400, {"error": "Invalid JSON format"})
                return

            if path == "/transactions/new":
                try:
                    tx = Transaction(**data)
                    if node.blockchain.add_transaction(tx):
                        # Add signature to seen so we don't loop gossip back
                        node.seen_transactions.add(tx.signature)
                        threading.Thread(target=node.gossip_transaction, args=(tx,)).start()
                        self.send_json(201, {"message": "Transaction verified and added", "signature": tx.signature})
                    else:
                        self.send_json(400, {"error": "Invalid transaction signature or duplicate"})
                except Exception as e:
                    self.send_json(400, {"error": f"Failed parsing transaction: {e}"})

            elif path == "/blocks/new":
                try:
                    txs = [Transaction(**tx_dict) for tx_dict in data.get("transactions", [])]
                    block = Block(
                        index=data["index"],
                        timestamp=data["timestamp"],
                        transactions=txs,
                        previous_hash=data["previous_hash"],
                        validator=data["validator"],
                        proof_of_inference=data.get("proof_of_inference", {}),
                        state_root=data["state_root"],
                        nonce=data["nonce"],
                        hash=data["hash"]
                    )
                    
                    if node.blockchain.validate_block(block):
                        node.seen_blocks.add(block.hash)
                        threading.Thread(target=node.gossip_block, args=(block,)).start()
                        self.send_json(201, {"message": "Block validated and added"})
                    else:
                        self.send_json(400, {"error": "Block validation rejected"})
                except Exception as e:
                    self.send_json(400, {"error": f"Failed parsing block: {e}"})

            elif path == "/peers/register":
                peer = data.get("peer")
                if peer:
                    node.register_peer(peer)
                    self.send_json(200, {"message": "Peer registered", "peers": list(node.peers)})
                else:
                    self.send_json(400, {"error": "Missing peer address"})
            
            elif path == "/mine":
                mined_block = node.blockchain.mine_block(node.validator_address)
                if mined_block:
                    node.seen_blocks.add(mined_block.hash)
                    threading.Thread(target=node.gossip_block, args=(mined_block,)).start()
                    self.send_json(200, {
                        "message": "Block mined and broadcast", 
                        "block": mined_block.model_dump() if hasattr(mined_block, "model_dump") else mined_block.dict()
                    })
                else:
                    self.send_json(400, {"error": "No pending transactions to mine"})
            else:
                self.send_json(404, {"error": "Not Found"})

    return P2PRequestHandler

def start_p2p_server(node: P2PNode):
    handler = make_handler(node)
    server = ThreadingHTTPServer(("0.0.0.0", node.port), handler)
    server.serve_forever()
