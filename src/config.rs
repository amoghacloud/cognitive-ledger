use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DEFAULT_FOUNDER_ADDRESS: &str = "-----BEGIN PUBLIC KEY-----\n\
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEYQlPKanrlObeW0lO0TPOENPMenj/\n\
RtKRycB+KHHuXF+CCmV7+31AshaslqeyC32PNY/TP2Wk+xBC07bruRYBDQ==\n\
-----END PUBLIC KEY-----\n";

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChainConfig {
    pub chain_id: String,
    pub difficulty: usize,
    pub max_transactions_per_block: usize,
    pub max_mempool_size: usize,
    pub max_model_dimension: usize,
    pub max_input_length: usize,
    pub model_registration_fee: f64,
    pub royalty_rate: f64,
    pub min_validator_stake: f64,
    pub validator_slashing_rate: f64,
    pub founder_address: Option<String>,
    pub founder_initial_balances: HashMap<String, f64>,
    pub allow_dev_faucet: bool,
    pub dev_initial_balances: HashMap<String, f64>,
}

impl Default for ChainConfig {
    fn default() -> Self {
        let mut founder_initial_balances = HashMap::new();
        founder_initial_balances.insert("FLOP".to_string(), 15000000.0);
        founder_initial_balances.insert("DATA".to_string(), 1500000.0);
        founder_initial_balances.insert("ATTN".to_string(), 1500000.0);

        let mut dev_initial_balances = HashMap::new();
        dev_initial_balances.insert("FLOP".to_string(), 10000.0);
        dev_initial_balances.insert("DATA".to_string(), 100.0);
        dev_initial_balances.insert("ATTN".to_string(), 10.0);

        Self {
            chain_id: "cognitive-ledger-localnet-v1".to_string(),
            difficulty: 2,
            max_transactions_per_block: 10,
            max_mempool_size: 1000,
            max_model_dimension: 4096,
            max_input_length: 4096,
            model_registration_fee: 10.0,
            royalty_rate: 0.10,
            min_validator_stake: 1000.0,
            validator_slashing_rate: 0.50,
            founder_address: Some(DEFAULT_FOUNDER_ADDRESS.to_string()),
            founder_initial_balances,
            allow_dev_faucet: false,
            dev_initial_balances,
        }
    }
}
