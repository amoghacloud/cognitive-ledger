import numpy as np
import time
import tempfile

from config import ChainConfig
from crypto_utils import generate_keypair, hash_data, merkle_root, verify_signature
from neural_vm import NeuralModel, calculate_gas
from tokenomics import TokenomicsManager
from blockchain import Blockchain, Transaction, Block
from storage import ChainStorage

def test_crypto():
    print("Testing Cryptography...")
    priv, pub = generate_keypair()
    assert priv is not None
    assert pub is not None
    
    data = "hello world"
    from crypto_utils import sign_data
    sig = sign_data(priv, data)
    assert verify_signature(pub, sig, data)
    assert not verify_signature(pub, sig, "modified data")
    
    h1 = hash_data("a")
    h2 = hash_data("b")
    root = merkle_root(["a", "b"])
    assert root == hash_data(h1 + h2)
    print("Cryptography OK!")

def test_neural_vm():
    print("Testing NeuralVM...")
    model = NeuralModel(model_id="test_model", seed=42, input_dim=4, hidden_dim=4, output_dim=2)
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    
    # Check floating point forward pass
    logits_fp, conf_fp, bounds_fp = model.forward(x, quantize=False)
    assert len(logits_fp) == 2
    assert 0.0 <= conf_fp <= 1.0
    
    # Check quantized forward pass (reproducible)
    logits_q, conf_q, bounds_q = model.forward(x, quantize=True)
    assert len(logits_q) == 2
    assert 0.0 <= conf_q <= 1.0
    
    # Re-instantiate with same seed, check output matches quantized exactly
    model2 = NeuralModel(model_id="test_model", seed=42, input_dim=4, hidden_dim=4, output_dim=2)
    logits_q2, conf_q2, bounds_q2 = model2.forward(x, quantize=True)
    assert logits_q == logits_q2
    assert conf_q == conf_q2
    assert bounds_q == bounds_q2
    
    # Verify gas calculation
    gas = calculate_gas(4, model, context_len=1, vram_time=0.1)
    assert gas >= 10
    print("NeuralVM OK!")

def test_blockchain_state():
    print("Testing Blockchain State...")
    config = ChainConfig(allow_dev_faucet=True)
    blockchain = Blockchain(config=config)
    
    # Create keypairs
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    
    # Ensure balance records
    blockchain.tokenomics.ensure_balance_record(pub_a)
    blockchain.tokenomics.ensure_balance_record(pub_b)
    
    # 1. Test TRANSFER
    tx_transfer = Transaction(
        sender=pub_a,
        receiver=pub_b,
        amount=100.0,
        token_type="FLOP",
        tx_type="TRANSFER",
        nonce=1
    )
    tx_transfer.signature = tx_transfer.signature or "MOCK_SIG" # will sign properly
    from crypto_utils import sign_data
    tx_transfer.signature = sign_data(priv_a, tx_transfer.to_signable_str())
    
    success = blockchain.add_transaction(tx_transfer)
    assert success, "Transfer tx addition failed"
    
    # Mine block
    block1 = blockchain.mine_block(validator_address=pub_b)
    assert block1 is not None
    assert len(block1.transactions) == 1
    
    # Verify balance
    bal_a = blockchain.tokenomics.get_balance(pub_a, "FLOP")
    bal_b = blockchain.tokenomics.get_balance(pub_b, "FLOP")
    assert bal_a == 9900.0, f"Expected 9900.0, got {bal_a}"
    assert bal_b == 10100.0, f"Expected 10100.0, got {bal_b}"
    
    # 2. Test MODEL REGISTRATION
    tx_register = Transaction(
        sender=pub_a,
        receiver="",
        amount=0.0,
        tx_type="REGISTER_MODEL",
        model_id=pub_a,
        model_dna=merkle_root(["model", pub_a, "10"]),
        input_dim=4,
        hidden_dim=4,
        output_dim=2,
        seed=10,
        dataset_id="mnist_data",
        nonce=2
    )
    tx_register.signature = sign_data(priv_a, tx_register.to_signable_str())
    success = blockchain.add_transaction(tx_register)
    assert success, "Model registration tx addition failed"
    
    # Mine block 2
    block2 = blockchain.mine_block(validator_address=pub_b)
    assert block2 is not None
    assert pub_a in blockchain.registered_models
    
    # 3. Test DATA STAKING
    tx_stake = Transaction(
        sender=pub_b,
        receiver="",
        amount=20.0,
        tx_type="STAKE_DATA",
        dataset_id="mnist_data",
        nonce=1
    )
    tx_stake.signature = sign_data(priv_b, tx_stake.to_signable_str())
    success = blockchain.add_transaction(tx_stake)
    assert success, "Data staking tx addition failed"
    
    block3 = blockchain.mine_block(validator_address=pub_a)
    assert block3 is not None
    assert blockchain.tokenomics.data_equity["mnist_data"]["total_staked"] == 20.0
    
    # 4. Test INFERENCE CONTRACT
    inputs_list = [0.1, 0.5, -0.2, 0.8]
    tx_infer = Transaction(
        sender=pub_b,
        receiver="",
        amount=0.0,
        tx_type="INFER_CONTRACT",
        model_id=pub_a,
        inputs=inputs_list,
        gas_limit=10000,
        max_fee=1.0,
        attention_bid=5.0,
        nonce=2
    )
    tx_infer.signature = sign_data(priv_b, tx_infer.to_signable_str())
    success = blockchain.add_transaction(tx_infer)
    assert success, "Inference tx addition failed"
    
    # Prior to mining, query balances of staker pub_b (for mnist_data) and validator pub_a
    pre_bal_staker = blockchain.tokenomics.get_balance(pub_b, "FLOP")
    pre_bal_val = blockchain.tokenomics.get_balance(pub_a, "FLOP")
    
    # Mine block 4
    block4 = blockchain.mine_block(validator_address=pub_a)
    assert block4 is not None
    assert tx_infer.signature in block4.proof_of_inference
    
    # Verify outputs in proof of inference
    proof = block4.proof_of_inference[tx_infer.signature]
    assert "logits" in proof
    assert len(proof["logits"]) == 2
    assert "confidence" in proof
    
    # Royalty check: Since model was bound to dataset mnist_data and pub_b staked 100% of it,
    # the 10% royalty should go to pub_b (the staker), and 90% to pub_a (the validator).
    # Also, attention bid of 5.0 ATTN goes to validator pub_a.
    gas_used = proof["gas_used"]
    total_fee = gas_used * 1.0
    royalty = total_fee * 0.1
    validator_cut = total_fee - royalty
    
    post_bal_staker = blockchain.tokenomics.get_balance(pub_b, "FLOP")
    post_bal_val = blockchain.tokenomics.get_balance(pub_a, "FLOP")
    
    assert post_bal_staker == pre_bal_staker - total_fee + royalty, f"Royalty mismatch: {post_bal_staker} vs {pre_bal_staker - total_fee + royalty}"
    assert post_bal_val == pre_bal_val + validator_cut, f"Validator cut mismatch: {post_bal_val} vs {pre_bal_val + validator_cut}"
    # Note: validator pub_a also gets 5.0 ATTN bid from pub_b
    assert blockchain.tokenomics.get_balance(pub_a, "ATTN") == 15.0
    assert blockchain.tokenomics.get_balance(pub_b, "ATTN") == 5.0
    
    # 5. Test VALIDATION of block chain reconstruction
    validator_chain = Blockchain(config=config)
    validator_chain.chain = [] # clear genesis
    
    for b in blockchain.chain:
        if b.index == 0:
            validator_chain.chain.append(b)
        else:
            valid = validator_chain.validate_block(b)
            assert valid, f"Block index {b.index} failed validation"
            
    print("Blockchain State & Validation OK!")

def test_hardening_guards():
    print("Testing Hardening Guards...")
    config = ChainConfig(allow_dev_faucet=True)
    blockchain = Blockchain(config=config)
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    from crypto_utils import sign_data

    bad_transfer = Transaction(
        sender=pub_a,
        receiver=pub_b,
        amount=-1.0,
        tx_type="TRANSFER",
        nonce=1,
    )
    bad_transfer.signature = sign_data(priv_a, bad_transfer.to_signable_str())
    assert not blockchain.add_transaction(bad_transfer), "Negative transfer should be rejected"

    tx1 = Transaction(
        sender=pub_a,
        receiver=pub_b,
        amount=1.0,
        tx_type="TRANSFER",
        nonce=1,
    )
    tx1.signature = sign_data(priv_a, tx1.to_signable_str())
    assert blockchain.add_transaction(tx1)

    tx2 = Transaction(
        sender=pub_a,
        receiver=pub_b,
        amount=2.0,
        tx_type="TRANSFER",
        nonce=1,
    )
    tx2.signature = sign_data(priv_a, tx2.to_signable_str())
    assert not blockchain.add_transaction(tx2), "Duplicate pending nonce should be rejected"
    print("Hardening Guards OK!")

def test_founder_allocation():
    print("Testing Founder Allocation...")
    _, founder_pub = generate_keypair()
    config = ChainConfig(founder_address=founder_pub)
    blockchain = Blockchain(config=config)
    assert blockchain.tokenomics.get_balance(founder_pub, "FLOP") == 15000000.0
    assert blockchain.tokenomics.get_balance(founder_pub, "DATA") == 1500000.0
    assert blockchain.tokenomics.get_balance(founder_pub, "ATTN") == 1500000.0
    print("Founder Allocation OK!")

def test_storage_round_trip():
    print("Testing Persistent Storage...")
    config = ChainConfig(allow_dev_faucet=True)
    blockchain = Blockchain(config=config)
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    from crypto_utils import sign_data

    tx = Transaction(
        sender=pub_a,
        receiver=pub_b,
        amount=25.0,
        tx_type="TRANSFER",
        nonce=1,
    )
    tx.signature = sign_data(priv_a, tx.to_signable_str())
    assert blockchain.add_transaction(tx)
    mined = blockchain.mine_block(pub_b)
    assert mined is not None

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = ChainStorage(f"{tmp_dir}/chain.json")
        storage.save(blockchain)
        restored = Blockchain(config=config)
        assert storage.load(restored)
        assert len(restored.chain) == len(blockchain.chain)
        assert restored.get_state_hash() == blockchain.get_state_hash()
        assert restored.tokenomics.get_balance(pub_b, "FLOP") == blockchain.tokenomics.get_balance(pub_b, "FLOP")
    print("Persistent Storage OK!")

if __name__ == "__main__":
    test_crypto()
    test_neural_vm()
    test_blockchain_state()
    test_hardening_guards()
    test_founder_allocation()
    test_storage_round_trip()
    print("All checks completed successfully!")
