use cognitive_ledger::config::ChainConfig;
use cognitive_ledger::crypto;
use cognitive_ledger::neural_vm::{NeuralModel, calculate_gas};
use cognitive_ledger::blockchain::{Blockchain, Transaction};
use cognitive_ledger::storage::ChainStorage;

#[test]
fn test_crypto() {
    println!("Testing Cryptography...");
    let (priv_key, pub_key) = crypto::generate_keypair().unwrap();
    assert!(!priv_key.is_empty());
    assert!(!pub_key.is_empty());

    let data = "hello world";
    let sig = crypto::sign_data(&priv_key, data).unwrap();
    assert!(crypto::verify_signature(&pub_key, &sig, data));
    assert!(!crypto::verify_signature(&pub_key, &sig, "modified data"));

    let h1 = crypto::hash_data("a");
    let h2 = crypto::hash_data("b");
    let root = crypto::merkle_root(&["a".to_string(), "b".to_string()]);
    assert_eq!(root, crypto::hash_data(&format!("{}{}", h1, h2)));
    println!("Cryptography OK!");
}

#[test]
fn test_neural_vm() {
    println!("Testing NeuralVM...");
    let model = NeuralModel::new("test_model".to_string(), 42, 4, 4, 2);
    let x = vec![0.1, 0.2, 0.3, 0.4];

    // Floating-point forward pass
    let (logits_fp, conf_fp, _bounds_fp) = model.forward(&x, false).unwrap();
    assert_eq!(logits_fp.len(), 2);
    assert!((0.0..=1.0).contains(&conf_fp));

    // Quantized forward pass (should be reproducible)
    let (logits_q, conf_q, bounds_q) = model.forward(&x, true).unwrap();
    assert_eq!(logits_q.len(), 2);
    assert!((0.0..=1.0).contains(&conf_q));

    // Re-instantiate with same seed, check output matches quantized exactly
    let model2 = NeuralModel::new("test_model".to_string(), 42, 4, 4, 2);
    let (logits_q2, conf_q2, bounds_q2) = model2.forward(&x, true).unwrap();
    assert_eq!(logits_q, logits_q2);
    assert_eq!(conf_q, conf_q2);
    assert_eq!(bounds_q, bounds_q2);

    // Verify gas calculation
    let gas = calculate_gas(4, &model, 1, 0.1);
    assert!(gas >= 10);
    println!("NeuralVM OK!");
}

#[test]
fn test_blockchain_state() {
    println!("Testing Blockchain State...");
    let mut config = ChainConfig::default();
    config.allow_dev_faucet = true;
    let mut blockchain = Blockchain::new(config, None);

    let (priv_a, pub_a) = crypto::generate_keypair().unwrap();
    let (priv_b, pub_b) = crypto::generate_keypair().unwrap();

    blockchain.tokenomics.ensure_balance_record(&pub_a);
    blockchain.tokenomics.ensure_balance_record(&pub_b);

    // Boost validator stakes
    blockchain.tokenomics.validators.insert(pub_a.clone(), 1000.0);
    blockchain.tokenomics.validators.insert(pub_b.clone(), 1000.0);

    // 1. Test TRANSFER
    let mut tx_transfer = Transaction {
        sender: pub_a.clone(),
        receiver: pub_b.clone(),
        amount: 100.0,
        token_type: "FLOP".to_string(),
        tx_type: "TRANSFER".to_string(),
        model_id: None,
        model_dna: None,
        input_dim: None,
        hidden_dim: None,
        output_dim: None,
        seed: None,
        dataset_id: None,
        inputs: None,
        gas_limit: None,
        max_fee: None,
        attention_bid: Some(0.0),
        signature: None,
        nonce: 1,
    };
    tx_transfer.signature = Some(crypto::sign_data(&priv_a, &tx_transfer.to_signable_str()).unwrap());
    
    let success = blockchain.add_transaction(tx_transfer);
    assert!(success, "Transfer tx addition failed");

    let leader1 = blockchain.select_leader(1, &blockchain.chain.last().unwrap().hash, &blockchain.tokenomics.validators);
    let block1 = blockchain.mine_block(&leader1).unwrap();
    assert_eq!(block1.transactions.len(), 1);

    let bal_a = blockchain.tokenomics.get_balance(&pub_a, "FLOP");
    let bal_b = blockchain.tokenomics.get_balance(&pub_b, "FLOP");
    assert_eq!(bal_a, 9900.0);
    assert_eq!(bal_b, 10100.0);

    // 2. Test MODEL REGISTRATION
    let mut tx_register = Transaction {
        sender: pub_a.clone(),
        receiver: "".to_string(),
        amount: 0.0,
        token_type: "FLOP".to_string(),
        tx_type: "REGISTER_MODEL".to_string(),
        model_id: Some(pub_a.clone()),
        model_dna: Some(crypto::merkle_root(&["model".to_string(), pub_a.clone(), "10".to_string()])),
        input_dim: Some(4),
        hidden_dim: Some(4),
        output_dim: Some(2),
        seed: Some(10),
        dataset_id: Some("mnist_data".to_string()),
        inputs: None,
        gas_limit: None,
        max_fee: None,
        attention_bid: Some(0.0),
        signature: None,
        nonce: 2,
    };
    tx_register.signature = Some(crypto::sign_data(&priv_a, &tx_register.to_signable_str()).unwrap());
    assert!(blockchain.add_transaction(tx_register), "Model registration tx failed");

    let leader2 = blockchain.select_leader(2, &blockchain.chain.last().unwrap().hash, &blockchain.tokenomics.validators);
    let block2 = blockchain.mine_block(&leader2).unwrap();
    assert_eq!(block2.transactions.len(), 1);
    assert!(blockchain.registered_models.contains_key(&pub_a));

    // 3. Test DATA STAKING
    let mut tx_stake = Transaction {
        sender: pub_b.clone(),
        receiver: "".to_string(),
        amount: 20.0,
        token_type: "DATA".to_string(),
        tx_type: "STAKE_DATA".to_string(),
        model_id: None,
        model_dna: None,
        input_dim: None,
        hidden_dim: None,
        output_dim: None,
        seed: None,
        dataset_id: Some("mnist_data".to_string()),
        inputs: None,
        gas_limit: None,
        max_fee: None,
        attention_bid: Some(0.0),
        signature: None,
        nonce: 1,
    };
    tx_stake.signature = Some(crypto::sign_data(&priv_b, &tx_stake.to_signable_str()).unwrap());
    assert!(blockchain.add_transaction(tx_stake), "Data staking tx failed");

    let leader3 = blockchain.select_leader(3, &blockchain.chain.last().unwrap().hash, &blockchain.tokenomics.validators);
    let block3 = blockchain.mine_block(&leader3).unwrap();
    assert_eq!(block3.transactions.len(), 1);
    assert_eq!(blockchain.tokenomics.data_equity.get("mnist_data").unwrap().total_staked, 20.0);

    // 4. Test INFER CONTRACT
    let mut tx_infer = Transaction {
        sender: pub_b.clone(),
        receiver: "".to_string(),
        amount: 0.0,
        token_type: "FLOP".to_string(),
        tx_type: "INFER_CONTRACT".to_string(),
        model_id: Some(pub_a.clone()),
        model_dna: None,
        input_dim: None,
        hidden_dim: None,
        output_dim: None,
        seed: None,
        dataset_id: None,
        inputs: Some(vec![0.1, 0.5, -0.2, 0.8]),
        gas_limit: Some(10000),
        max_fee: Some(1.0),
        attention_bid: Some(5.0),
        signature: None,
        nonce: 2,
    };
    tx_infer.signature = Some(crypto::sign_data(&priv_b, &tx_infer.to_signable_str()).unwrap());
    assert!(blockchain.add_transaction(tx_infer.clone()), "Inference tx failed");

    let pre_bal_staker = blockchain.tokenomics.get_balance(&pub_b, "FLOP");
    let leader4 = blockchain.select_leader(4, &blockchain.chain.last().unwrap().hash, &blockchain.tokenomics.validators);
    let pre_bal_val = blockchain.tokenomics.get_balance(&leader4, "FLOP");

    let block4 = blockchain.mine_block(&leader4).unwrap();
    let sig_key = tx_infer.signature.clone().unwrap();
    assert!(block4.proof_of_inference.contains_key(&sig_key));

    let proof = &block4.proof_of_inference[&sig_key];
    assert!(proof.get("logits").is_some());
    assert_eq!(proof["logits"].as_array().unwrap().len(), 2);

    let gas_used = proof["gas_used"].as_u64().unwrap() as f64;
    let total_fee = gas_used * 1.0;
    let royalty = total_fee * 0.1;
    let validator_cut = total_fee - royalty;

    let post_bal_staker = blockchain.tokenomics.get_balance(&pub_b, "FLOP");
    let post_bal_val = blockchain.tokenomics.get_balance(&leader4, "FLOP");

    let (expected_staker_bal, expected_val_bal) = if leader4 == pub_b {
        (pre_bal_staker, pre_bal_val)
    } else {
        (pre_bal_staker - total_fee + royalty, pre_bal_val + validator_cut)
    };

    assert_eq!(post_bal_staker, expected_staker_bal);
    assert_eq!(post_bal_val, expected_val_bal);

    // 5. Test VALIDATION of block chain reconstruction
    let mut validator_chain = Blockchain::new(blockchain.config.clone(), None);
    validator_chain.chain = Vec::new(); // clear genesis

    for b in &blockchain.chain {
        if b.index == 0 {
            validator_chain.chain.push(b.clone());
        } else {
            validator_chain.tokenomics.validators.insert(pub_a.clone(), 1000.0);
            validator_chain.tokenomics.validators.insert(pub_b.clone(), 1000.0);
            let valid = validator_chain.validate_block(b, true);
            assert!(valid, "Block index {} failed validation", b.index);
        }
    }
    println!("Blockchain State & Validation OK!");
}

#[test]
fn test_hardening_guards() {
    println!("Testing Hardening Guards...");
    let mut config = ChainConfig::default();
    config.allow_dev_faucet = true;
    let mut blockchain = Blockchain::new(config, None);

    let (priv_a, pub_a) = crypto::generate_keypair().unwrap();
    let (_priv_b, pub_b) = crypto::generate_keypair().unwrap();

    let mut bad_transfer = Transaction {
        sender: pub_a.clone(),
        receiver: pub_b.clone(),
        amount: -1.0,
        token_type: "FLOP".to_string(),
        tx_type: "TRANSFER".to_string(),
        model_id: None,
        model_dna: None,
        input_dim: None,
        hidden_dim: None,
        output_dim: None,
        seed: None,
        dataset_id: None,
        inputs: None,
        gas_limit: None,
        max_fee: None,
        attention_bid: Some(0.0),
        signature: None,
        nonce: 1,
    };
    bad_transfer.signature = Some(crypto::sign_data(&priv_a, &bad_transfer.to_signable_str()).unwrap());
    assert!(!blockchain.add_transaction(bad_transfer), "Negative transfer should be rejected");

    let mut tx1 = Transaction {
        sender: pub_a.clone(),
        receiver: pub_b.clone(),
        amount: 1.0,
        token_type: "FLOP".to_string(),
        tx_type: "TRANSFER".to_string(),
        model_id: None,
        model_dna: None,
        input_dim: None,
        hidden_dim: None,
        output_dim: None,
        seed: None,
        dataset_id: None,
        inputs: None,
        gas_limit: None,
        max_fee: None,
        attention_bid: Some(0.0),
        signature: None,
        nonce: 1,
    };
    tx1.signature = Some(crypto::sign_data(&priv_a, &tx1.to_signable_str()).unwrap());
    assert!(blockchain.add_transaction(tx1));

    let mut tx2 = Transaction {
        sender: pub_a.clone(),
        receiver: pub_b.clone(),
        amount: 2.0,
        token_type: "FLOP".to_string(),
        tx_type: "TRANSFER".to_string(),
        model_id: None,
        model_dna: None,
        input_dim: None,
        hidden_dim: None,
        output_dim: None,
        seed: None,
        dataset_id: None,
        inputs: None,
        gas_limit: None,
        max_fee: None,
        attention_bid: Some(0.0),
        signature: None,
        nonce: 1,
    };
    tx2.signature = Some(crypto::sign_data(&priv_a, &tx2.to_signable_str()).unwrap());
    assert!(!blockchain.add_transaction(tx2), "Duplicate pending nonce should be rejected");
    println!("Hardening Guards OK!");
}

#[test]
fn test_founder_allocation() {
    println!("Testing Founder Allocation...");
    let (_, founder_pub) = crypto::generate_keypair().unwrap();
    let mut config = ChainConfig::default();
    config.founder_address = Some(founder_pub.clone());
    let mut blockchain = Blockchain::new(config, None);
    assert_eq!(blockchain.tokenomics.get_balance(&founder_pub, "FLOP"), 15000000.0);
    assert_eq!(blockchain.tokenomics.get_balance(&founder_pub, "DATA"), 1500000.0);
    assert_eq!(blockchain.tokenomics.get_balance(&founder_pub, "ATTN"), 1500000.0);
    println!("Founder Allocation OK!");
}

#[test]
fn test_storage_round_trip() {
    println!("Testing Persistent Storage...");
    let mut config = ChainConfig::default();
    config.allow_dev_faucet = true;
    config.min_validator_stake = 1000.0;
    let mut blockchain = Blockchain::new(config.clone(), None);

    let (priv_a, pub_a) = crypto::generate_keypair().unwrap();
    let (_, pub_b) = crypto::generate_keypair().unwrap();

    let mut tx_stake = Transaction {
        sender: pub_a.clone(),
        receiver: "".to_string(),
        amount: 1000.0,
        token_type: "FLOP".to_string(),
        tx_type: "STAKE_VALIDATOR".to_string(),
        model_id: None,
        model_dna: None,
        input_dim: None,
        hidden_dim: None,
        output_dim: None,
        seed: None,
        dataset_id: None,
        inputs: None,
        gas_limit: None,
        max_fee: None,
        attention_bid: Some(0.0),
        signature: None,
        nonce: 1,
    };
    tx_stake.signature = Some(crypto::sign_data(&priv_a, &tx_stake.to_signable_str()).unwrap());
    assert!(blockchain.add_transaction(tx_stake));

    let founder = config.founder_address.clone().unwrap();
    let block1 = blockchain.mine_block(&founder).unwrap();

    let mut tx = Transaction {
        sender: pub_a.clone(),
        receiver: pub_b.clone(),
        amount: 25.0,
        token_type: "FLOP".to_string(),
        tx_type: "TRANSFER".to_string(),
        model_id: None,
        model_dna: None,
        input_dim: None,
        hidden_dim: None,
        output_dim: None,
        seed: None,
        dataset_id: None,
        inputs: None,
        gas_limit: None,
        max_fee: None,
        attention_bid: Some(0.0),
        signature: None,
        nonce: 2,
    };
    tx.signature = Some(crypto::sign_data(&priv_a, &tx.to_signable_str()).unwrap());
    assert!(blockchain.add_transaction(tx));

    let leader = blockchain.select_leader(2, &block1.hash, &blockchain.tokenomics.validators);
    let mined = blockchain.mine_block(&leader);
    assert!(mined.is_some());

    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("chain.json").to_str().unwrap().to_string();

    let storage = ChainStorage::new(db_path);
    storage.save(&blockchain).unwrap();

    let mut restored = Blockchain::new(config, None);
    assert!(storage.load(&mut restored));
    assert_eq!(restored.chain.len(), blockchain.chain.len());
    assert_eq!(restored.get_state_hash(), blockchain.get_state_hash());
    println!("Persistent Storage OK!");
}

#[test]
fn test_pos_consensus_and_slashing() {
    println!("Testing PoS Consensus & Auto-Slashing...");
    let mut config = ChainConfig::default();
    config.allow_dev_faucet = true;
    config.min_validator_stake = 1000.0;
    let mut blockchain = Blockchain::new(config.clone(), None);

    let (priv_a, pub_a) = crypto::generate_keypair().unwrap();
    let (_, pub_b) = crypto::generate_keypair().unwrap();

    blockchain.tokenomics.ensure_balance_record(&pub_a);
    blockchain.tokenomics.ensure_balance_record(&pub_b);

    let mut tx_stake = Transaction {
        sender: pub_a.clone(),
        receiver: "".to_string(),
        amount: 2000.0,
        token_type: "FLOP".to_string(),
        tx_type: "STAKE_VALIDATOR".to_string(),
        model_id: None,
        model_dna: None,
        input_dim: None,
        hidden_dim: None,
        output_dim: None,
        seed: None,
        dataset_id: None,
        inputs: None,
        gas_limit: None,
        max_fee: None,
        attention_bid: Some(0.0),
        signature: None,
        nonce: 1,
    };
    tx_stake.signature = Some(crypto::sign_data(&priv_a, &tx_stake.to_signable_str()).unwrap());
    assert!(blockchain.add_transaction(tx_stake));

    let founder = config.founder_address.clone().unwrap();
    let block1 = blockchain.mine_block(&founder).unwrap();
    assert_eq!(blockchain.tokenomics.validators[&pub_a], 2000.0);

    let expected_leader = blockchain.select_leader(2, &block1.hash, &blockchain.tokenomics.validators);
    assert_eq!(expected_leader, pub_a);

    let mut tx_dummy = Transaction {
        sender: pub_a.clone(),
        receiver: pub_b.clone(),
        amount: 1.0,
        token_type: "FLOP".to_string(),
        tx_type: "TRANSFER".to_string(),
        model_id: None,
        model_dna: None,
        input_dim: None,
        hidden_dim: None,
        output_dim: None,
        seed: None,
        dataset_id: None,
        inputs: None,
        gas_limit: None,
        max_fee: None,
        attention_bid: Some(0.0),
        signature: None,
        nonce: 2,
    };
    tx_dummy.signature = Some(crypto::sign_data(&priv_a, &tx_dummy.to_signable_str()).unwrap());
    assert!(blockchain.add_transaction(tx_dummy));

    // Proposing block 2 as pub_b should panic / be rejected (we check slot leader in mine_block)
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut bc_temp = Blockchain::new(blockchain.config.clone(), None);
        bc_temp.chain = blockchain.chain.clone();
        bc_temp.tokenomics = blockchain.tokenomics.clone();
        bc_temp.registered_models = blockchain.registered_models.clone();
        bc_temp.account_nonces = blockchain.account_nonces.clone();
        bc_temp.pending_transactions = blockchain.pending_transactions.clone();
        bc_temp.mine_block(&pub_b);
    }));
    assert!(res.is_err(), "Block proposing from non-leader should have failed/panicked");

    let block2 = blockchain.mine_block(&pub_a).unwrap();
    assert_eq!(block2.validator, pub_a);

    // Test Slashing on validate_block
    let mut bad_block = block2.clone();
    bad_block.index = 3;
    bad_block.previous_hash = block2.hash.clone();
    bad_block.state_root = "CORRUPTED_ROOT".to_string();
    bad_block.hash = bad_block.calculate_hash();

    let pre_slash_stake = blockchain.tokenomics.validators[&pub_a];
    assert_eq!(pre_slash_stake, 2000.0);

    let valid = blockchain.validate_block(&bad_block, false);
    assert!(!valid, "Corrupted block should be rejected by validation");

    let post_slash_stake = blockchain.tokenomics.validators.get(&pub_a).cloned().unwrap_or(0.0);
    assert_eq!(post_slash_stake, 1000.0);
    println!("PoS Consensus & Auto-Slashing OK!");
}
