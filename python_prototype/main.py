import os
import sys
import time
import json
import argparse
import threading
import requests
import numpy as np

from cryptography.hazmat.primitives.serialization import load_pem_private_key, Encoding, PublicFormat
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align

from config import ChainConfig, DEFAULT_FOUNDER_ADDRESS
from crypto_utils import generate_keypair, hash_data, merkle_root
from blockchain import Transaction, Block
from network import P2PNode, start_p2p_server
from agent import AIAgent

def load_or_create_key(key_path: str = "wallet_private.pem") -> tuple:
    """Loads a PEM private/public keypair, or creates a new one if missing."""
    if os.path.exists(key_path):
        try:
            with open(key_path, "r") as f:
                priv_key_str = f.read()
            k = load_pem_private_key(priv_key_str.encode("utf-8"), password=None)
            pub_key_str = k.public_key().public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            ).decode("utf-8")
            return priv_key_str, pub_key_str
        except Exception:
            pass

    # Fallback/Generate new
    priv, pub = generate_keypair()
    with open(key_path, "w") as f:
        f.write(priv)
    pub_path = key_path.replace("private", "public")
    with open(pub_path, "w") as f:
        f.write(pub)
    return priv, pub

def auto_miner_loop(node: P2PNode):
    """Periodically checks the mempool and mines blocks if transactions exist."""
    while True:
        time.sleep(5)
        if node.blockchain.pending_transactions:
            mined = node.blockchain.mine_block(node.validator_address)
            if mined:
                print(f"\n[AutoMiner] Mined Block #{mined.index} (Hash: {mined.hash[:12]}...).")
                node.gossip_block(mined)

def print_help(console: Console):
    table = Table(title="Available CLI Commands")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_row("status", "Show current block height, mempool size, peer count, model registry count")
    table.add_row("balances", "Display current token balances ($FLOP, $DATA, $ATTN) for all active accounts")
    table.add_row("mine", "Manually trigger mining of pending transactions in the pool")
    table.add_row("peers", "List all registered peer addresses")
    table.add_row("chain", "Display a detailed summary of all blocks in the blockchain")
    table.add_row("models", "List registered neural network models, architectures, and datasets")
    table.add_row("exit", "Terminate the validator node process")
    console.print(table)

def print_status(console: Console, node: P2PNode):
    table = Table(title=f"Node {node.port} Status Dashboard")
    table.add_column("Metric / Property", style="cyan")
    table.add_column("Current State", style="green")
    table.add_row("Block Height", str(len(node.blockchain.chain)))
    table.add_row("Mempool Size", str(len(node.blockchain.pending_transactions)))
    table.add_row("Peer Count", str(len(node.peers)))
    table.add_row("Registered Models", str(len(node.blockchain.registered_models)))
    table.add_row("Chain ID", node.blockchain.config.chain_id)
    table.add_row("Difficulty", str(node.blockchain.difficulty))
    table.add_row("Validator Node Address", node.validator_address[:32] + "...")
    table.add_row("State Ledger Hash", node.blockchain.get_state_hash()[:16] + "...")
    console.print(table)

def print_balances(console: Console, node: P2PNode):
    table = Table(title="Ledger Token Balances")
    table.add_column("Address / Public Key", style="cyan", no_wrap=True)
    table.add_column("FLOP (Gas)", style="green")
    table.add_column("DATA (Quality)", style="magenta")
    table.add_column("ATTN (Priority)", style="yellow")
    
    balances = node.blockchain.tokenomics.balances
    for addr, bal in balances.items():
        disp_addr = addr[:24] + "..." if len(addr) > 24 else addr
        table.add_row(
            disp_addr,
            f"{bal.get('FLOP', 0.0):.2f}",
            f"{bal.get('DATA', 0.0):.2f}",
            f"{bal.get('ATTN', 0.0):.2f}"
        )
    console.print(table)

def print_peers(console: Console, node: P2PNode):
    if not node.peers:
        console.print("[yellow]No active peers. Node is running in isolation mode.[/yellow]")
        return
    table = Table(title="P2P Peer Nodes")
    table.add_column("Peer Server Address", style="cyan")
    for peer in node.peers:
        table.add_row(peer)
    console.print(table)

def print_chain(console: Console, node: P2PNode):
    table = Table(title="Blockchain Ledger History")
    table.add_column("Idx", style="cyan")
    table.add_column("Block Hash", style="green")
    table.add_column("Prev Hash", style="white")
    table.add_column("Txs", style="magenta")
    table.add_column("Validator Address", style="yellow")
    table.add_column("State Hash", style="blue")
    
    for block in node.blockchain.chain:
        table.add_row(
            str(block.index),
            block.hash[:12] + "...",
            block.previous_hash[:12] + "...",
            str(len(block.transactions)),
            block.validator[:12] + "...",
            block.state_root[:12] + "..."
        )
    console.print(table)

def print_models(console: Console, node: P2PNode):
    if not node.blockchain.registered_models:
        console.print("[yellow]No neural models registered on this ledger yet.[/yellow]")
        return
    table = Table(title="Cognitive Model Registry")
    table.add_column("Model ID (Pubkey)", style="cyan", no_wrap=True)
    table.add_column("Model DNA (Merkle Root)", style="green")
    table.add_column("Arch (In/Hidden/Out)", style="magenta")
    table.add_column("Consensus Seed", style="yellow")
    table.add_column("Bound Dataset ($DET)", style="blue")
    
    for m_id, info in node.blockchain.registered_models.items():
        disp_id = m_id[:20] + "..." if len(m_id) > 20 else m_id
        arch_str = f"{info['input_dim']} -> {info['hidden_dim']} -> {info['output_dim']}"
        dataset = info.get("dataset_id") or "None (No Royalties)"
        table.add_row(
            disp_id,
            info["model_dna"][:12] + "...",
            arch_str,
            str(info["seed"]),
            dataset
        )
    console.print(table)

def run_node(args):
    console = Console()
    priv_file = f"validator_private_{args.port}.pem"
    
    # 1. Load or Generate keys for this node
    console.print("[cyan]Initializing node keys...[/cyan]")
    priv_key, pub_key = load_or_create_key(priv_file)
    
    # 2. Instantiate local P2P Node
    peers_list = [p.strip() for p in args.peers.split(",") if p.strip()] if args.peers else []
    founder_address = None if args.no_founder_allocation else (args.founder_address or DEFAULT_FOUNDER_ADDRESS)
    config = ChainConfig(
        chain_id=args.chain_id,
        difficulty=args.difficulty,
        founder_address=founder_address,
        allow_dev_faucet=args.dev_faucet,
    )
    storage_path = os.path.join(args.data_dir, f"node_{args.port}.json") if args.data_dir else None
    node = P2PNode(
        port=args.port,
        peers=peers_list,
        validator_address=pub_key,
        config=config,
        storage_path=storage_path,
    )
    
    # 3. Start P2P HTTP Server on a daemon background thread
    server_thread = threading.Thread(target=start_p2p_server, args=(node,), daemon=True)
    server_thread.start()
    
    # 4. Connect and sync chain with peers
    console.print("[cyan]Connecting and synchronizing chain with peer network...[/cyan]")
    node.connect_to_network()
    
    # 5. Start AutoMiner if enabled
    if args.auto_mine:
        miner_thread = threading.Thread(target=auto_miner_loop, args=(node,), daemon=True)
        miner_thread.start()
        console.add_alternative_screen = False
        console.print("[green]AutoMiner active (mines pending mempool every 5 seconds).[/green]")
    
    # Welcome banner
    console.print(Panel(
        Align.center(f"[bold yellow]COGNITIVE LEDGER NODE ONLINE[/bold yellow]\n\n"
                     f"[cyan]Port:[/cyan] {args.port} | [cyan]Validator Address:[/cyan] {pub_key[:16]}...\n"
                     f"[cyan]Block Height:[/cyan] {len(node.blockchain.chain)} | [cyan]Mempool Limit:[/cyan] {node.blockchain.config.max_transactions_per_block} Txs/Block\n"
                     f"[cyan]Chain ID:[/cyan] {node.blockchain.config.chain_id} | [cyan]Store:[/cyan] {storage_path or 'memory'}\n"
                     f"Type [green]help[/green] to list console commands."),
        title="[bold green]Genesis Setup Complete[/bold green]"
    ))

    # Command loop
    while True:
        try:
            cmd = input(f"node:{args.port}> ").strip()
            if not cmd:
                continue
            
            parts = cmd.split()
            base = parts[0].lower()
            
            if base == "exit":
                console.print("[red]Shutting down node...[/red]")
                os._exit(0)
            elif base == "help":
                print_help(console)
            elif base == "status":
                print_status(console, node)
            elif base == "balances":
                print_balances(console, node)
            elif base == "mine":
                mined = node.blockchain.mine_block(node.validator_address)
                if mined:
                    console.print(f"[green]Successfully mined block #{mined.index}! Hash: {mined.hash[:16]}...[/green]")
                    node.gossip_block(mined)
                else:
                    console.print("[yellow]No valid transactions in the mempool to mine.[/yellow]")
            elif base == "peers":
                print_peers(console, node)
            elif base == "chain":
                print_chain(console, node)
            elif base == "models":
                print_models(console, node)
            else:
                console.print(f"[red]Unknown command '{base}'. Type 'help' for options.[/red]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]Exiting...[/red]")
            os._exit(0)

def handle_wallet(args):
    console = Console()
    key_file = args.key_file or "wallet_private.pem"
    
    if args.create:
        if os.path.exists(key_file):
            console.print(f"[yellow]Wallet private key already exists at {key_file}. Loading it.[/yellow]")
        priv, pub = load_or_create_key(key_file)
        
        console.print(Panel(
            f"[cyan]Private Key File:[/cyan] {key_file}\n"
            f"[cyan]Public Key File:[/cyan] {key_file.replace('private', 'public')}\n\n"
            f"[cyan]Address / Public Key (PEM):[/cyan]\n[green]{pub}[/green]",
            title="[bold green]Cognitive Wallet Details[/bold green]"
        ))
    elif args.check:
        if not os.path.exists(key_file):
            console.print(f"[red]Private key file not found at {key_file}. Create one using --create[/red]")
            return
        priv, pub = load_or_create_key(key_file)
        
        node_url = f"http://localhost:{args.port}"
        try:
            resp = requests.get(f"{node_url}/state", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                balances = data.get("balances", {})
                my_bal = balances.get(pub, {"FLOP": 0.0, "DATA": 0.0, "ATTN": 0.0})
                
                table = Table(title="Wallet Account Balances")
                table.add_column("Asset", style="cyan")
                table.add_column("Balance Value", style="green")
                table.add_row("FLOP (Gas Compute)", f"{my_bal.get('FLOP', 0.0):.2f}")
                table.add_row("DATA (Dataset Equity)", f"{my_bal.get('DATA', 0.0):.2f}")
                table.add_row("ATTN (Auction Priority)", f"{my_bal.get('ATTN', 0.0):.2f}")
                console.print(table)
            else:
                console.print(f"[red]Error fetching balances from node: {resp.status_code}[/red]")
        except Exception as e:
            console.print(f"[red]Failed connecting to node at {node_url}: {e}[/red]")
    else:
        console.print("[yellow]Specify --create to generate keys, or --check to verify balances.[/yellow]")

def handle_tx(args):
    console = Console()
    key_file = args.key_file or "wallet_private.pem"
    
    if not os.path.exists(key_file):
        console.print(f"[red]No private key file found at {key_file}. Run 'python main.py wallet --create' first.[/red]")
        return
        
    priv, pub = load_or_create_key(key_file)
    node_url = f"http://localhost:{args.port}"
    
    # 1. Fetch current nonce for account from node
    nonce = 0
    try:
        resp = requests.get(f"{node_url}/state", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            nonces = data.get("account_nonces", {})
            nonce = nonces.get(pub, 0)
    except Exception:
        console.print(f"[yellow]Unable to pull transaction count from node. Defaulting nonce to 0.[/yellow]")
        
    next_nonce = nonce + 1
    tx_type = args.type.upper()
    
    # 2. Build Transaction payload structure
    tx_args = {
        "sender": pub,
        "receiver": args.receiver or "",
        "amount": args.amount or 0.0,
        "token_type": args.token or "FLOP",
        "tx_type": tx_type,
        "nonce": next_nonce
    }
    
    if tx_type == "REGISTER_MODEL":
        if not args.model_dna:
            # Generate deterministic model weights using seed and compute DNA Merkle Root
            seed = args.seed or 42
            input_dim = args.input_dim or 128
            hidden_dim = args.hidden_dim or 64
            output_dim = args.output_dim or 10
            
            rng = np.random.default_rng(seed)
            w1 = rng.normal(0, 0.1, (input_dim, hidden_dim))
            w2 = rng.normal(0, 0.1, (hidden_dim, output_dim))
            w1_hash = hash_data(w1.tobytes().hex())
            w2_hash = hash_data(w2.tobytes().hex())
            args.model_dna = merkle_root([
                w1_hash, w2_hash, str(input_dim), str(hidden_dim), str(output_dim), str(seed)
            ])
            
        tx_args.update({
            "model_id": args.model_id or pub,
            "model_dna": args.model_dna,
            "input_dim": args.input_dim or 128,
            "hidden_dim": args.hidden_dim or 64,
            "output_dim": args.output_dim or 10,
            "seed": args.seed or 42,
            "dataset_id": args.dataset_id
        })
        
    elif tx_type == "INFER_CONTRACT":
        if not args.model_id:
            console.print("[red]--model-id is required for inference contract execution.[/red]")
            return
        if not args.inputs:
            console.print("[red]--inputs (comma separated floats) required for inference.[/red]")
            return
        try:
            inputs_list = [float(x) for x in args.inputs.split(",")]
        except Exception:
            console.print("[red]Inputs list must be comma-separated numeric float coordinates.[/red]")
            return
            
        tx_args.update({
            "model_id": args.model_id,
            "inputs": inputs_list,
            "gas_limit": args.gas_limit or 1000000,
            "max_fee": args.max_fee or 1.0,
            "attention_bid": args.attn_bid or 0.0
        })
        
    elif tx_type == "STAKE_DATA":
        if not args.dataset_id:
            console.print("[red]--dataset-id must be specified to stake DATA equity.[/red]")
            return
        if not args.amount or args.amount <= 0:
            console.print("[red]--amount to stake must be greater than 0.[/red]")
            return
        tx_args.update({
            "dataset_id": args.dataset_id,
            "amount": args.amount
        })
        
    elif tx_type == "TRANSFER":
        if not args.receiver:
            console.print("[red]--receiver public key required for TRANSFER tx.[/red]")
            return
        if not args.amount or args.amount <= 0:
            console.print("[red]--amount must be greater than 0.[/red]")
            return

    elif tx_type == "STAKE_VALIDATOR":
        if not args.amount or args.amount <= 0:
            console.print("[red]--amount must be specified and greater than 0 for STAKE_VALIDATOR[/red]")
            return
        tx_args.update({
            "amount": args.amount,
            "token_type": "FLOP"
        })

    elif tx_type == "UNSTAKE_VALIDATOR":
        if not args.amount or args.amount <= 0:
            console.print("[red]--amount must be specified and greater than 0 for UNSTAKE_VALIDATOR[/red]")
            return
        tx_args.update({
            "amount": args.amount,
            "token_type": "FLOP"
        })

    tx = Transaction(**tx_args)
    
    # 3. Cryptographically Sign transaction
    from crypto_utils import sign_data
    tx.signature = sign_data(priv, tx.to_signable_str())
    
    # 4. Post transaction to P2P Node endpoint
    try:
        data = tx.model_dump() if hasattr(tx, "model_dump") else tx.dict()
        resp = requests.post(f"{node_url}/transactions/new", json=data, timeout=5)
        if resp.status_code in (200, 201):
            console.print(f"[green]Transaction broadcasted to network successfully! Signature: {tx.signature[:16]}...[/green]")
            console.print(resp.json())
        else:
            console.print(f"[red]Node rejected transaction: {resp.status_code} - {resp.json().get('error', resp.text)}[/red]")
    except Exception as e:
        console.print(f"[red]Failed sending transaction request to local node: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description="Cognitive Ledger CLI Control Center")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Node runner
    node_parser = subparsers.add_parser("node", help="Starts a running validator peer node")
    node_parser.add_argument("--port", type=int, default=5000, help="Local port to bind node")
    node_parser.add_argument("--peers", type=str, default="", help="Comma separated list of peer ports/hosts (e.g. localhost:5001)")
    node_parser.add_argument("--auto-mine", action="store_true", help="Enables automatic background block mining")
    node_parser.add_argument("--data-dir", type=str, default="data", help="Directory for persistent node chain state")
    node_parser.add_argument("--chain-id", type=str, default="cognitive-ledger-localnet-v1", help="Expected chain/network identifier")
    node_parser.add_argument("--difficulty", type=int, default=2, help="Proof-of-work leading-zero difficulty")
    node_parser.add_argument("--founder-address", type=str, default=None, help="Override founder public key receiving genesis allocation")
    node_parser.add_argument("--no-founder-allocation", action="store_true", help="Disable default founder genesis allocation for private experiments")
    node_parser.add_argument("--dev-faucet", action="store_true", help="Auto-fund first-seen accounts for local demos only")

    # Wallet creator/querier
    wallet_parser = subparsers.add_parser("wallet", help="Handles key generation and balances check")
    wallet_parser.add_argument("--create", action="store_true", help="Generates a new SECP256R1 wallet keypair")
    wallet_parser.add_argument("--check", action="store_true", help="Check address balance on a live node")
    wallet_parser.add_argument("--port", type=int, default=5000, help="Local node port to query (default 5000)")
    wallet_parser.add_argument("--key-file", type=str, default="wallet_private.pem", help="Custom path for the key file")

    # Transaction sender
    tx_parser = subparsers.add_parser("tx", help="Builds, signs, and submits transactions to the ledger network")
    tx_parser.add_argument("--port", type=int, default=5000, help="Port of node to submit transaction")
    tx_parser.add_argument("--key-file", type=str, default="wallet_private.pem", help="Path to sender's private key file")
    tx_parser.add_argument("--type", type=str, required=True, choices=["transfer", "register_model", "infer_contract", "stake_data", "stake_validator", "unstake_validator"], help="Type of transaction")
    tx_parser.add_argument("--receiver", type=str, help="Recipient address for TRANSFER transactions")
    tx_parser.add_argument("--amount", type=float, help="Value/Amount for TRANSFER or STAKE_DATA transactions")
    tx_parser.add_argument("--token", type=str, default="FLOP", choices=["FLOP", "DATA", "ATTN"], help="Token asset type")
    
    # Model registration options
    tx_parser.add_argument("--model-id", type=str, help="Unique identifier/Public key of registered neural model")
    tx_parser.add_argument("--model-dna", type=str, help="Hereditary Merkle root hash of model architecture")
    tx_parser.add_argument("--input-dim", type=int, default=128, help="Inference model input vector dimension")
    tx_parser.add_argument("--hidden-dim", type=int, default=64, help="Inference model hidden layer dimension")
    tx_parser.add_argument("--output-dim", type=int, default=10, help="Inference model output prediction classes")
    tx_parser.add_argument("--seed", type=int, default=42, help="Consensus execution weights random seed")
    tx_parser.add_argument("--dataset-id", type=str, help="ID of training dataset to bind model for staker royalties")

    # Neural inference options
    tx_parser.add_argument("--inputs", type=str, help="Comma separated inputs floats list (e.g. 0.1,0.2,-0.5)")
    tx_parser.add_argument("--gas-limit", type=int, default=1000000, help="Maximum gas allowed for neural forward pass")
    tx_parser.add_argument("--max-fee", type=float, default=1.0, help="Maximum price per gas (FLOP) to pay")
    tx_parser.add_argument("--attn-bid", type=float, default=0.0, help="Vickrey priority auction attention token bid ($ATTN)")

    args = parser.parse_args()

    if args.command == "node":
        run_node(args)
    elif args.command == "wallet":
        handle_wallet(args)
    elif args.command == "tx":
        handle_tx(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
