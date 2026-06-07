import numpy as np
import time
import tempfile
import copy

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
    
    # Boost pub_a and pub_b as validators in PoS
    blockchain.tokenomics.validators[pub_a] = 1000.0
    blockchain.tokenomics.validators[pub_b] = 1000.0
    
    # 1. Test TRANSFER
    tx_transfer = Transaction(
        sender=pub_a,
        receiver=pub_b,
        amount=100.0,
        token_type="FLOP",
        tx_type="TRANSFER",
        nonce=1
    )
    from crypto_utils import sign_data
    tx_transfer.signature = sign_data(priv_a, tx_transfer.to_signable_str())
    
    success = blockchain.add_transaction(tx_transfer)
    assert success, "Transfer tx addition failed"
    
    # Mine block using the expected slot leader
    leader1 = blockchain.select_leader(1, blockchain.chain[-1].hash, blockchain.tokenomics)
    block1 = blockchain.mine_block(validator_address=leader1)
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
    
    # Mine block 2 using the expected slot leader
    leader2 = blockchain.select_leader(2, blockchain.chain[-1].hash, blockchain.tokenomics)
    block2 = blockchain.mine_block(validator_address=leader2)
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
    
    # Mine block 3 using the expected slot leader
    leader3 = blockchain.select_leader(3, blockchain.chain[-1].hash, blockchain.tokenomics)
    block3 = blockchain.mine_block(validator_address=leader3)
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
    
    # Prior to mining, query balances of staker pub_b (for mnist_data) and validator/miner
    pre_bal_staker = blockchain.tokenomics.get_balance(pub_b, "FLOP")
    
    # Determine the validator address for the next block
    leader4 = blockchain.select_leader(4, blockchain.chain[-1].hash, blockchain.tokenomics)
    pre_bal_val = blockchain.tokenomics.get_balance(leader4, "FLOP")
    
    # Mine block 4
    block4 = blockchain.mine_block(validator_address=leader4)
    assert block4 is not None
    assert tx_infer.signature in block4.proof_of_inference
    
    # Verify outputs in proof of inference
    proof = block4.proof_of_inference[tx_infer.signature]
    assert "logits" in proof
    assert len(proof["logits"]) == 2
    assert "confidence" in proof
    
    gas_used = proof["gas_used"]
    total_fee = gas_used * 1.0
    royalty = total_fee * 0.1
    validator_cut = total_fee - royalty
    
    post_bal_staker = blockchain.tokenomics.get_balance(pub_b, "FLOP")
    post_bal_val = blockchain.tokenomics.get_balance(leader4, "FLOP")
    
    # Check staker balance (pub_b pays the gas fee and also receives the 100% of staker royalty)
    if leader4 == pub_b:
        expected_staker_bal = pre_bal_staker
        expected_val_bal = pre_bal_val
    else:
        expected_staker_bal = pre_bal_staker - total_fee + royalty
        expected_val_bal = pre_bal_val + validator_cut

    assert post_bal_staker == expected_staker_bal, f"Royalty mismatch: {post_bal_staker} vs {expected_staker_bal}"
    # Check validator balance
    assert post_bal_val == expected_val_bal, f"Validator cut mismatch: {post_bal_val} vs {expected_val_bal}"
    
    # 5. Test VALIDATION of block chain reconstruction
    validator_chain = Blockchain(config=config)
    validator_chain.chain = [] # clear genesis
    
    for b in blockchain.chain:
        if b.index == 0:
            validator_chain.chain.append(b)
        else:
            # Reconstruct validators on validator chain so validation passes
            validator_chain.tokenomics.validators[pub_a] = 1000.0
            validator_chain.tokenomics.validators[pub_b] = 1000.0
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
    config = ChainConfig(allow_dev_faucet=True, min_validator_stake=1000.0)
    blockchain = Blockchain(config=config)
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    from crypto_utils import sign_data

    # Register pub_a as validator via transaction
    tx_stake = Transaction(
        sender=pub_a,
        receiver="",
        amount=1000.0,
        tx_type="STAKE_VALIDATOR",
        token_type="FLOP",
        nonce=1
    )
    tx_stake.signature = sign_data(priv_a, tx_stake.to_signable_str())
    assert blockchain.add_transaction(tx_stake)

    # Mine block 1 (Founder is slot leader by default since no validators are staked yet)
    founder = config.founder_address
    block1 = blockchain.mine_block(founder)
    assert block1 is not None

    # Now pub_a is registered. Propose block 2 using pub_a
    tx = Transaction(
        sender=pub_a,
        receiver=pub_b,
        amount=25.0,
        tx_type="TRANSFER",
        nonce=2,
    )
    tx.signature = sign_data(priv_a, tx.to_signable_str())
    assert blockchain.add_transaction(tx)
    
    leader = blockchain.select_leader(2, block1.hash, blockchain.tokenomics)
    mined = blockchain.mine_block(leader)
    assert mined is not None

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = ChainStorage(f"{tmp_dir}/chain.json")
        storage.save(blockchain)
        restored = Blockchain(config=config)
        
        assert storage.load(restored)
        assert len(restored.chain) == len(blockchain.chain)
        assert restored.get_state_hash() == blockchain.get_state_hash()
    print("Persistent Storage OK!")

def test_pos_consensus_and_slashing():
    print("Testing PoS Consensus & Auto-Slashing...")
    config = ChainConfig(allow_dev_faucet=True, min_validator_stake=1000.0)
    blockchain = Blockchain(config=config)
    
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    from crypto_utils import sign_data

    blockchain.tokenomics.ensure_balance_record(pub_a)
    blockchain.tokenomics.ensure_balance_record(pub_b)

    # 1. Stake pub_a as validator via transaction
    tx_stake = Transaction(
        sender=pub_a,
        receiver="",
        amount=2000.0,
        tx_type="STAKE_VALIDATOR",
        token_type="FLOP",
        nonce=1
    )
    tx_stake.signature = sign_data(priv_a, tx_stake.to_signable_str())
    assert blockchain.add_transaction(tx_stake)

    # Mine block 1 (Founder is slot leader by default since no validators are staked yet)
    founder = config.founder_address
    block1 = blockchain.mine_block(founder)
    assert block1 is not None
    assert blockchain.tokenomics.validators[pub_a] == 2000.0

    # 2. Block 2 proposer check
    # Now pub_a is the only active validator staked.
    expected_leader = blockchain.select_leader(2, block1.hash, blockchain.tokenomics)
    assert expected_leader == pub_a

    # Add transaction first so block mining executes
    tx_dummy = Transaction(
        sender=pub_a,
        receiver=pub_b,
        amount=1.0,
        tx_type="TRANSFER",
        nonce=2
    )
    tx_dummy.signature = sign_data(priv_a, tx_dummy.to_signable_str())
    assert blockchain.add_transaction(tx_dummy)

    # Trying to propose block 2 as pub_b should raise ValueError (Not slot leader)
    try:
        blockchain.mine_block(validator_address=pub_b)
        assert False, "Block proposing from non-leader should have failed"
    except ValueError as e:
        assert "Not slot leader" in str(e)

    # Proposing block 2 as pub_a succeeds
    block2 = blockchain.mine_block(pub_a)
    assert block2 is not None

    # 3. Test Slashing on validate_block
    # We construct a forged block where pub_a (the leader) signed an invalid transaction, 
    # or the state_root is corrupted.
    bad_block = copy.deepcopy(block2)
    bad_block.index = 3
    bad_block.previous_hash = block2.hash
    bad_block.state_root = "CORRUPTED_ROOT"
    bad_block.hash = bad_block.calculate_hash()

    # validate_block on bad_block should return False AND slash pub_a by 50%
    pre_slash_stake = blockchain.tokenomics.validators[pub_a]
    assert pre_slash_stake == 2000.0

    valid = blockchain.validate_block(bad_block, persist=False)
    assert not valid, "Corrupted block should be rejected by validation"

    post_slash_stake = blockchain.tokenomics.validators.get(pub_a, 0.0)
    assert post_slash_stake == 1000.0, f"Validator stake was not slashed. Got {post_slash_stake}"
    print("PoS Consensus & Auto-Slashing OK!")

if __name__ == "__main__":
    test_crypto()
    test_neural_vm()
    test_blockchain_state()
    test_hardening_guards()
    test_founder_allocation()
    test_storage_round_trip()
    test_pos_consensus_and_slashing()
    print("All checks completed successfully!")
