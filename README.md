# Cognitive Ledger: P2P AI-Native Blockchain Node (Rust)

The **Cognitive Ledger** is a peer-to-peer (P2P) blockchain specifically engineered for autonomous machine intelligence. Smart contracts are replaced by **Neural Contracts**—executable, deterministic neural network inferences evaluated directly in a modular **Neural Virtual Machine (NVM)**.

This repository provides a complete, runnable blockchain node implemented in **Rust**, supporting secure transaction signing (ECDSA SECP256R1), peer-to-peer gossip networking, longest-chain consensus synchronization, deterministic integer-quantized model execution, persistent storage, and a tri-asset token economy ($FLOP, $DATA, $ATTN) alongside Vickrey-like attention auctions.

---

## Architectural Mapping (Rust implementation)

* **Cryptography (`src/crypto.rs`):** SECP256R1 keypair generation, signature verification, SHA-256, and Merkle tree roots.
* **Neural VM (`src/neural_vm.rs`):** Quantitative matrix execution (INT8), Shannon-entropy based uncertainty outputs, and Standard AI Compute Units (SACU) gas calculations.
* **Tokenomics Manager (`src/tokenomics.rs`):** Balance bookkeeping, dataset staking pools, validator staking registers, automated slashing, and Vickrey-like priority auction sorting.
* **Blockchain Core (`src/blockchain.rs`):** Transaction validation, state mapping, weighted validator selection (Proof-of-Stake), and automated 50% proposer slashing on invalid proposals.
* **Persistent Storage (`src/storage.rs`):** Atomic JSON database serializations.
* **P2P Gossip Network (`src/network.rs`):** Async HTTP routes via Axum and gossip client via Reqwest.
* **CLI Dashboard (`src/main.rs`):** Interactive console for wallet queries, transaction signing, and running validator nodes.

---

## Installation & Setup

1. **Ensure Rust is installed:**
   If not, install it using:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/amoghacloud/cognitive-ledger.git
   cd cognitive-ledger
   ```

3. **Verify compilation and run tests:**
   ```bash
   cargo test
   ```

4. **Build the release binary:**
   ```bash
   cargo build --release
   ```

---

## Localnet Sandbox Tutorial

Follow these steps to run a local multi-port network, register a model, stake dataset equity, and submit an inference transaction.

### Step 1: Create Wallets
```bash
# Generate founder keys
cargo run -- wallet --generate
```
This prints your SECP256R1 private key and public key PEM strings. Save these keys!

### Step 2: Start Node A (Port 5000)
Run the first node with auto-mining (`--mine`) enabled. This node will act as the slot leader using the default founder address:
```bash
cargo run -- node --port 5000 --mine --db data/chain_5000.json
```

### Step 3: Start Node B (Port 5001)
Open a new terminal, and launch a second node connected to the first one:
```bash
cargo run -- node --port 5001 --peers http://127.0.0.1:5000 --db data/chain_5001.json
```

### Step 4: Register a Model (Neural Contract)
From a third terminal, register a model of size `4` inputs, `4` hidden nodes, and `2` outputs:
```bash
cargo run -- tx --type REGISTER_MODEL \
  --sender-key wallet_private.pem \
  --model <MODEL_ID_HEX_OR_PEM> \
  --dna <MODEL_DNA_HASH> \
  --dims 4,4,2 \
  --seed 42 \
  --dataset dataset_mnist \
  --node-url http://127.0.0.1:5000
```

### Step 5: Stake DATA Tokens
Stake `50.0 DATA` tokens to register validator A as a dataset equity stakeholder:
```bash
cargo run -- tx --type STAKE_DATA \
  --sender-key wallet_private.pem \
  --dataset dataset_mnist \
  --amount 50.0 \
  --node-url http://127.0.0.1:5000
```

### Step 6: Submit a Quantized Inference Request
Submit a mock tensor coordinate array:
```bash
cargo run -- tx --type INFER_CONTRACT \
  --sender-key wallet_private.pem \
  --model <MODEL_ID_HEX_OR_PEM> \
  --inputs 0.1,0.5,-0.2,0.8 \
  --gas-limit 10000 \
  --max-fee 1.0 \
  --attn-bid 1.0 \
  --node-url http://127.0.0.1:5000
```

### Step 7: Check Balances & Slashing
You can query any wallet balance using:
```bash
cargo run -- wallet --balance <ADDRESS_PEM> --db data/chain_5000.json
```
The stakers of `dataset_mnist` automatically receive **10%** of the gas fee as royalty yield, while the block proposer receives **90%**.
If a validator attempts to propose a block containing invalid state roots or corrupted VM executions, they will be auto-slashed by **50%** of their staked balance.
