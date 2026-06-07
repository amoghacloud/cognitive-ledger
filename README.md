# Cognitive Ledger: P2P AI-Native Blockchain Node

The **Cognitive Ledger** is a peer-to-peer (P2P) blockchain specifically engineered for autonomous machine intelligence. Smart contracts are replaced by **Neural Contracts**—executable, deterministic neural network inferences evaluated directly in a modular **Neural Virtual Machine (NVM)**.

This repository provides a complete, runnable blockchain node implemented in Python, supporting secure transaction signing (ECDSA), peer-to-peer gossip networking, longest-chain consensus synchronization, deterministic integer-quantized model execution (Deterministic Inference Mode), persistent storage, and a tri-asset token economy ($FLOP, $DATA, $ATTN) alongside Vickrey attention priority routing.

---

## Conceptual Architecture Map

```
        ┌────────────────────────────────────────────────────────┐
        │                 AGENT / APPLICATION LAYER              │
        │      (AIAgent with Public/Private Keys & Model DNA)     │
        └───────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │                    P2P NETWORK LAYER                   │
        │       (Asynchronous HTTP Gossip & Chain Sync)          │
        └───────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │                    BLOCKCHAIN LEDGER                   │
        │   (Mempool, VCG Attention Auctions, Proof of Work)     │
        └───────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │                  NEURAL VIRTUAL MACHINE                │
        │    (INT8 Quantized Matrix Math, Entropy Uncertainty)   │
        └───────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │                    TOKENOMICS MANAGER                  │
        │    (Tri-Asset Ledgers & Data Equity $DET Royalties)    │
        └────────────────────────────────────────────────────────┘
```

---

## Architectural Specification (How It Works)

The implementation maps directly to the layers defined in the **Cognitive Ledger Master Plan**:

### 1. Layer 1: Identity & Attestation (`agent.py`)
AI Agents are cryptographic citizens. Each `AIAgent` holds a SECP256R1 keypair and inherits a **Model DNA**—a content-addressed Merkle Root computed by hashing the model's structural configuration, hyper-parameters, and initialization weights. This links the agent's identity directly to its cognitive state.

### 2. Layer 2: Consensus & Proof-of-Inference (`blockchain.py`)
To prevent block validators from acting lazily or recording fabricated model outputs, the ledger enforces **Proof of Inference (PoI)**. When a new block containing inference contracts is propagated:
1. Every validating node loads the target model from its registry.
2. The validator re-runs the quantized forward pass on the inputs.
3. The validator compares its locally computed outputs (`logits`, `confidence`, and `uncertainty_bounds`) against the block's `proof_of_inference` header.
4. If there is a single bit of discrepancy, the block is rejected.

### 3. Layer 3: Execution Runtime (NVM) (`neural_vm.py`)
Floating-point calculations (`float32`) vary slightly across different processor architectures (GPUs/CPUs) due to IEEE-754 optimization flags. The Neural VM solves this by running a **Deterministic Inference Mode (DIM)**:
* Inputs and weights are quantized into fixed-point `int8`/`int32` spaces.
* Matrix dot products are computed using pure integer algebra.
* Outputs are rescaled back to float spaces, ensuring bit-for-bit identical outputs globally.
* Metes gas in **Standardized AI Compute Units (SACUs)**:
  $$\text{Gas} = (\text{Tokens}_{\text{input}} \times \text{FLOPs}) + (\text{Tokens}_{\text{context}} \times \text{VRAM}_{\text{time}})$$
* Provides **Uncertainty-Aware Execution (UAE)** by computing the Shannon Entropy of output probability distributions to bound predictions.

### 4. Layer 6: Tri-Asset & Data Equity Economy (`tokenomics.py`)
* **$FLOP:** The utility token burned to pay for NVM forward-pass computational gas.
* **$DATA:** The dataset staking token used to participate in data validation pools.
* **$ATTN:** Bids placed by transactions to gain execution priority in the block queue.
* **$DET (Data Equity Tokens):** Stakers who lock up `$DATA` in a registered dataset pool earn continuous yield. Whenever a neural contract bound to that dataset executes, **10% of the gas fee ($FLOP)** is automatically streamed to stakers relative to their pool share. The executing validator receives the remaining 90%.

---

## Installation & Setup

1. **Clone & Open Project Directory:**
   ```bash
   git clone https://github.com/amoghacloud/cognitive-ledger.git
   cd cognitive-ledger
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Clean Compilation & Tests:**
   ```bash
   python3 test_node.py
   python3 -m py_compile *.py
   ```

---

## Step-by-Step Interactive Tutorial (Localnet Sandbox)

Follow these steps to run a local multi-port network, register a neural network, lock dataset equity, execute inference, and observe fee distribution.

### Step 1: Create Wallets
We generate default key files to sign our transactions.

```bash
# Generate Founder/Validator A key (default: wallet_private.pem)
python main.py wallet --create

# Generate Validator B key
python main.py wallet --create --key-file validator_private_5001.pem
```

### Step 2: Start Node A (Port 5000)
Boot Node A with auto-mining and the developer faucet enabled to fund accounts with starting tokens. Specify a persistent storage directory (`data`).

```bash
python main.py node --port 5000 --auto-mine --dev-faucet --data-dir data
```
*Note: This generates `data/node_5000.json` containing the genesis state.*

### Step 3: Start Node B (Port 5001) in a New Terminal
Open a new terminal shell, activate the virtual environment, and boot Node B. Connect it to Node A to kick off peer discovery and state synchronization.

```bash
python main.py node --port 5001 --peers localhost:5000 --auto-mine --dev-faucet --data-dir data
```

*Inside Node B's CLI prompt, type `peers` to verify connection to Node A. Type `chain` to confirm block height matches.*

---

### Step 4: Register a Neural Model (Neural Contract)
From a third terminal (acting as the client), register a neural model. We define its layer layout (128 inputs -> 64 hidden nodes -> 10 output classes), initialize weights using seed 42, and bind it to dataset `dataset_mnist`.

```bash
python main.py tx --port 5000 --type register_model --seed 42 --input-dim 128 --hidden-dim 64 --output-dim 10 --dataset-id dataset_mnist
```

*Verify the model is visible on all nodes by typing `models` in Node B's terminal.*

### Step 5: Stake DATA Tokens into dataset_mnist
Stake `50.0 DATA` tokens to the dataset. This registers Validator A as an equity stakeholder entitled to royalties.

```bash
python main.py tx --port 5000 --type stake_data --dataset-id dataset_mnist --amount 50.0
```

### Step 6: Submit a Quantized Inference Request
Construct a 128-dimensional mock tensor coordinate (e.g. 128 values of `0.5` separated by commas) and submit it for execution. 

First, get the model ID (which is the Founder public key printed inside `public_key.pem` or visible by running `models` on the node).

Generate 128 inputs:
```python
python -c "print(','.join(['0.5']*128))"
```

Submit the inference request:
```bash
python main.py tx --port 5000 --type infer_contract --model-id <MODEL_ID_HEX> --inputs <128_COMMA_SEPARATED_FLOATS> --gas-limit 100000 --max-fee 1.0 --attn-bid 1.0
```

### Step 7: Inspect Proof-of-Inference and State Roots
In Node A or Node B's terminal, view the block history:

```
node:5000> chain
```

You will see the mined blocks. The block containing the inference transaction will include a `proof_of_inference` block mapping containing:
* `logits`: The exact deterministic outputs.
* `confidence`: The softmax confidence.
* `uncertainty_bounds`: The Shannon-entropy bounds.
* `gas_used`: The gas consumed by the execution.

Also, type `balances` to see that:
* The model registrar's `$FLOP` gas was deducted.
* **10%** of the gas fee was paid to the staker of `dataset_mnist` as royalty yield.
* **90%** of the gas fee was paid to the block validator.

---

## CLI Command References

### Global Subcommands
* `python main.py node [args]`: Starts a validator node server.
* `python main.py wallet [args]`: Generates keys and displays account balances.
* `python main.py tx [args]`: Constructs and broadcasts signed transactions.

### Running Node Shell Commands
* `status`: Prints active metrics, heights, and state root.
* `balances`: Lists the balance ledger of all addresses.
* `mine`: Manually forces a block mining cycle.
* `peers`: Lists registered P2P network peers.
* `chain`: Renders full blockchain block records.
* `models`: Renders the registered model directory.
* `exit`: Cleanly closes node server.
