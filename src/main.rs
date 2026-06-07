use cognitive_ledger::config::ChainConfig;
use cognitive_ledger::blockchain::{Blockchain, Transaction, Block};
use cognitive_ledger::storage::ChainStorage;
use cognitive_ledger::network::{GossipNodeState, start_node_server, sync_with_peers};
use cognitive_ledger::crypto;

use p256::pkcs8::{DecodePrivateKey, EncodePublicKey};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_help();
        return Ok(());
    }

    let command = args[1].as_str();
    match command {
        "wallet" => {
            handle_wallet(&args).await?;
        }
        "node" => {
            handle_node(&args).await?;
        }
        "tx" => {
            handle_tx(&args).await?;
        }
        _ => {
            print_help();
        }
    }

    Ok(())
}

fn print_help() {
    println!("=== Cognitive Ledger CLI (Rust) ===");
    println!("Usage:");
    println!("  cognitive_ledger wallet --generate");
    println!("  cognitive_ledger wallet --balance <address> [--db <db_path>]");
    println!("  cognitive_ledger node --port <port> [--peers <p1,p2>] [--db <db_path>] [--mine]");
    println!("  cognitive_ledger tx --type <type> --sender-key <path_or_pem> --receiver <addr> --amount <val> --node-url <url> ...");
    println!("\nTransaction options:");
    println!("  --type: TRANSFER, REGISTER_MODEL, INFER_CONTRACT, STAKE_DATA, STAKE_VALIDATOR, UNSTAKE_VALIDATOR");
    println!("  --token: FLOP, DATA, ATTN");
    println!("  --model: <model_id>");
    println!("  --dna: <model_dna>");
    println!("  --dims: <input,hidden,output>");
    println!("  --seed: <seed>");
    println!("  --dataset: <dataset_id>");
    println!("  --inputs: <comma_separated_floats>");
    println!("  --gas-limit: <limit>");
    println!("  --max-fee: <fee>");
    println!("  --attn-bid: <bid>");
}

async fn handle_wallet(args: &[String]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if get_arg(args, "--generate").is_some() {
        let (priv_key, pub_key) = crypto::generate_keypair()?;
        println!("=== SECP256R1 ECDSA KEYPAIR GENERATED ===");
        println!("--- PRIVATE KEY PEM ---");
        println!("{}", priv_key);
        println!("--- PUBLIC KEY PEM (ADDRESS) ---");
        println!("{}", pub_key);
        return Ok(());
    }

    if let Some(addr) = get_arg(args, "--balance") {
        let db_path = get_arg(args, "--db").unwrap_or_else(|| "data/chain_5000.json".to_string());
        let config = ChainConfig::default();
        let mut bc = Blockchain::new(config, None);
        let storage = ChainStorage::new(db_path);
        
        if storage.load(&mut bc) {
            bc.tokenomics.ensure_balance_record(&addr);
            println!("Balances for address:");
            println!("  FLOP: {}", bc.tokenomics.get_balance(&addr, "FLOP"));
            println!("  DATA: {}", bc.tokenomics.get_balance(&addr, "DATA"));
            println!("  ATTN: {}", bc.tokenomics.get_balance(&addr, "ATTN"));
            if let Some(stake) = bc.tokenomics.validators.get(&addr) {
                println!("  Staked Validator FLOP: {}", stake);
            }
        } else {
            println!("Failed to load database from: {}", storage.file_path);
        }
        return Ok(());
    }

    print_help();
    Ok(())
}

async fn handle_node(args: &[String]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let port_str = get_arg(args, "--port").unwrap_or_else(|| "5000".to_string());
    let port: u16 = port_str.parse().unwrap_or(5000);

    let db_path = get_arg(args, "--db").unwrap_or_else(|| format!("data/chain_{}.json", port));
    
    // Ensure parent directory exists for DB
    if let Some(parent) = std::path::Path::new(&db_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let peers_str = get_arg(args, "--peers").unwrap_or_default();
    let mut peers_list = Vec::new();
    for p in peers_str.split(',') {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            peers_list.push(trimmed.to_string());
        }
    }

    let should_mine = get_arg(args, "--mine").is_some();

    let config = ChainConfig::default();
    let mut bc = Blockchain::new(config, None);
    let storage = ChainStorage::new(db_path.clone());
    
    if storage.load(&mut bc) {
        println!("[Node Init] Successfully loaded chain from database (height: {}).", bc.chain.len());
    } else {
        println!("[Node Init] No existing chain database found. Initialized genesis block.");
        storage.save(&bc)?;
    }

    let bc_shared = Arc::new(RwLock::new(bc));
    let peers_shared = Arc::new(RwLock::new(peers_list));
    
    let state = Arc::new(GossipNodeState {
        blockchain: bc_shared.clone(),
        peers: peers_shared.clone(),
        db_path: db_path.clone(),
    });

    // Run synchronization task
    let sync_state = state.clone();
    tokio::spawn(async move {
        println!("[P2P Sync] Starting peer synchronization...");
        sync_with_peers(sync_state).await;
    });

    // Mining Loop (if --mine is specified)
    if should_mine {
        let mine_state = state.clone();
        tokio::spawn(async move {
            println!("[Node Miner] Proposer thread active.");
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                
                let mut bc = mine_state.blockchain.write().await;
                if bc.pending_transactions.is_empty() {
                    continue;
                }

                let prev_block = bc.chain.last().unwrap().clone();
                let leader = bc.select_leader(prev_block.index + 1, &prev_block.hash, &bc.tokenomics.validators);
                
                let miner_address = bc.config.founder_address.clone().unwrap_or_default();
                if leader == miner_address {
                    println!("[Node Miner] We are slot leader for block index {}. Proposing block...", prev_block.index + 1);
                    if let Some(block) = bc.mine_block(&miner_address) {
                        println!("[Node Miner] Block {} proposed successfully. Hash: {}", block.index, block.hash);
                        let storage = ChainStorage::new(mine_state.db_path.clone());
                        let _ = storage.save(&bc);
                        
                        // Gossip block to peers
                        let peers = mine_state.peers.read().await.clone();
                        let client = reqwest::Client::new();
                        for peer in peers {
                            let url = format!("{}/blocks", peer);
                            let _ = client.post(&url).json(&block).send().await;
                        }
                    }
                }
            }
        });
    }

    start_node_server(state, port).await?;
    Ok(())
}

async fn handle_tx(args: &[String]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tx_type = get_arg(args, "--type").ok_or("Missing --type")?;
    let node_url = get_arg(args, "--node-url").unwrap_or_else(|| "http://127.0.0.1:5000".to_string());
    
    // Fetch private key
    let priv_key_raw = get_arg(args, "--sender-key").ok_or("Missing --sender-key")?;
    let priv_key = if std::path::Path::new(&priv_key_raw).exists() {
        std::fs::read_to_string(priv_key_raw)?
    } else {
        priv_key_raw.replace("\\n", "\n")
    };

    // Extract public key to identify sender address
    let signing_key = p256::ecdsa::SigningKey::from_pkcs8_pem(&priv_key)?;
    let verifying_key = p256::ecdsa::VerifyingKey::from(&signing_key);
    let sender_pub = verifying_key.to_public_key_pem(p256::pkcs8::LineEnding::LF)?;

    // Fetch chain from node to determine account state and nonce
    println!("[CLI Tx] Fetching blockchain state from node to determine nonce...");
    let client = reqwest::Client::new();
    let sync_url = format!("{}/sync", node_url);
    
    let res = client.get(&sync_url).send().await?;
    if !res.status().is_success() {
        return Err(format!("Node returned error status: {}", res.status()).into());
    }

    let chain: Vec<Block> = res.json().await?;
    let config = ChainConfig::default();
    let mut bc = Blockchain::new(config, None);
    bc.rebuild_from_chain(&chain)?;

    let current_nonce = bc.account_nonces.get(&sender_pub).cloned().unwrap_or(0);
    let nonce = current_nonce + 1;

    let receiver = get_arg(args, "--receiver").unwrap_or_default();
    let amount: f64 = get_arg(args, "--amount").unwrap_or_default().parse().unwrap_or(0.0);
    let token_type = get_arg(args, "--token").unwrap_or_else(|| "FLOP".to_string());
    
    let model_id = get_arg(args, "--model");
    let model_dna = get_arg(args, "--dna");
    
    let dims_str = get_arg(args, "--dims").unwrap_or_default();
    let mut input_dim = None;
    let mut hidden_dim = None;
    let mut output_dim = None;
    if !dims_str.is_empty() {
        let parts: Vec<&str> = dims_str.split(',').collect();
        if parts.len() == 3 {
            input_dim = Some(parts[0].parse()?);
            hidden_dim = Some(parts[1].parse()?);
            output_dim = Some(parts[2].parse()?);
        }
    }

    let seed = get_arg(args, "--seed").map(|s| s.parse().unwrap_or(42));
    let dataset_id = get_arg(args, "--dataset");

    let inputs_str = get_arg(args, "--inputs").unwrap_or_default();
    let mut inputs = None;
    if !inputs_str.is_empty() {
        let mut vec = Vec::new();
        for val in inputs_str.split(',') {
            vec.push(val.trim().parse()?);
        }
        inputs = Some(vec);
    }

    let gas_limit = get_arg(args, "--gas-limit").map(|g| g.parse().unwrap_or(100));
    let max_fee = get_arg(args, "--max-fee").map(|f| f.parse().unwrap_or(1.0));
    let attention_bid = get_arg(args, "--attn-bid").map(|b| b.parse().unwrap_or(0.0));

    let mut tx = Transaction {
        sender: sender_pub.clone(),
        receiver,
        amount,
        token_type,
        tx_type,
        model_id,
        model_dna,
        input_dim,
        hidden_dim,
        output_dim,
        seed,
        dataset_id,
        inputs,
        gas_limit,
        max_fee,
        attention_bid,
        signature: None,
        nonce,
    };

    // Sign the transaction
    let signable = tx.to_signable_str();
    let signature = crypto::sign_data(&priv_key, &signable)?;
    tx.signature = Some(signature);

    // Send transaction to node
    println!("[CLI Tx] Sending signed transaction to node...");
    let tx_url = format!("{}/transactions", node_url);
    let post_res = client.post(&tx_url).json(&tx).send().await?;

    let res_text = post_res.text().await?;
    println!("Node Response:\n{}", res_text);

    Ok(())
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == flag {
            if i + 1 < args.len() {
                return Some(args[i + 1].clone());
            } else {
                return Some("".to_string());
            }
        }
    }
    None
}
