use crate::config::ChainConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DatasetStakingPool {
    pub stakers: HashMap<String, f64>,
    pub total_staked: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AttentionBid {
    pub address: String,
    pub bid: f64,
    pub tx_id: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TokenomicsManager {
    pub config: ChainConfig,
    pub balances: HashMap<String, HashMap<String, f64>>,
    pub data_equity: HashMap<String, DatasetStakingPool>,
    pub model_datasets: HashMap<String, String>,
    pub validators: HashMap<String, f64>,
}

impl TokenomicsManager {
    pub fn new(config: ChainConfig, genesis_balances: Option<HashMap<String, HashMap<String, f64>>>) -> Self {
        let mut manager = Self {
            config,
            balances: HashMap::new(),
            data_equity: HashMap::new(),
            model_datasets: HashMap::new(),
            validators: HashMap::new(),
        };

        if let Some(genesis) = genesis_balances {
            for (address, tokens) in genesis {
                manager.set_balance_record(&address, tokens);
            }
        }

        manager
    }

    pub fn set_balance_record(&mut self, address: &str, tokens: HashMap<String, f64>) {
        if address.is_empty() {
            return;
        }
        let mut map = HashMap::new();
        map.insert("FLOP".to_string(), 0.0);
        map.insert("DATA".to_string(), 0.0);
        map.insert("ATTN".to_string(), 0.0);

        for (token, amount) in tokens {
            if amount.is_finite() && amount >= 0.0 {
                map.insert(token, amount);
            }
        }
        self.balances.insert(address.to_string(), map);
    }

    pub fn ensure_balance_record(&mut self, address: &str) {
        if address.is_empty() {
            return;
        }
        if !self.balances.contains_key(address) {
            let mut map = HashMap::new();
            if self.config.allow_dev_faucet {
                map.insert("FLOP".to_string(), *self.config.dev_initial_balances.get("FLOP").unwrap_or(&0.0));
                map.insert("DATA".to_string(), *self.config.dev_initial_balances.get("DATA").unwrap_or(&0.0));
                map.insert("ATTN".to_string(), *self.config.dev_initial_balances.get("ATTN").unwrap_or(&0.0));
            } else {
                map.insert("FLOP".to_string(), 0.0);
                map.insert("DATA".to_string(), 0.0);
                map.insert("ATTN".to_string(), 0.0);
            }
            self.balances.insert(address.to_string(), map);
        }
    }

    pub fn get_balance(&mut self, address: &str, token_type: &str) -> f64 {
        self.ensure_balance_record(address);
        self.balances
            .get(address)
            .and_then(|m| m.get(token_type))
            .cloned()
            .unwrap_or(0.0)
    }

    pub fn transfer(&mut self, sender: &str, receiver: &str, amount: f64, token_type: &str) -> bool {
        if amount <= 0.0 || !amount.is_finite() {
            return false;
        }
        self.ensure_balance_record(sender);
        self.ensure_balance_record(receiver);

        let sender_bal = self.get_balance(sender, token_type);
        if sender_bal < amount {
            return false;
        }

        if let Some(s_map) = self.balances.get_mut(sender) {
            if let Some(val) = s_map.get_mut(token_type) {
                *val -= amount;
            }
        }

        if let Some(r_map) = self.balances.get_mut(receiver) {
            if let Some(val) = r_map.get_mut(token_type) {
                *val += amount;
            }
        }

        true
    }

    pub fn register_dataset_equity(&mut self, dataset_id: &str) {
        if dataset_id.is_empty() {
            return;
        }
        self.data_equity
            .entry(dataset_id.to_string())
            .or_insert(DatasetStakingPool {
                stakers: HashMap::new(),
                total_staked: 0.0,
            });
    }

    pub fn stake_data_equity(&mut self, address: &str, dataset_id: &str, amount: f64) -> bool {
        if amount <= 0.0 || !amount.is_finite() {
            return false;
        }
        self.ensure_balance_record(address);
        self.register_dataset_equity(dataset_id);

        let data_bal = self.get_balance(address, "DATA");
        if data_bal < amount {
            return false;
        }

        if let Some(s_map) = self.balances.get_mut(address) {
            if let Some(val) = s_map.get_mut("DATA") {
                *val -= amount;
            }
        }

        if let Some(pool) = self.data_equity.get_mut(dataset_id) {
            let entry = pool.stakers.entry(address.to_string()).or_insert(0.0);
            *entry += amount;
            pool.total_staked += amount;
        }

        true
    }

    pub fn bind_model_to_dataset(&mut self, model_id: &str, dataset_id: &str) {
        if model_id.is_empty() || dataset_id.is_empty() {
            return;
        }
        self.model_datasets.insert(model_id.to_string(), dataset_id.to_string());
    }

    pub fn route_execution_royalties(&mut self, model_id: &str, total_gas_paid: f64) -> f64 {
        if total_gas_paid <= 0.0 || !total_gas_paid.is_finite() {
            return 0.0;
        }

        let dataset_id = match self.model_datasets.get(model_id) {
            Some(id) => id.clone(),
            None => return total_gas_paid,
        };

        let pool = match self.data_equity.get(&dataset_id) {
            Some(p) => p,
            None => return total_gas_paid,
        };

        let total_staked = pool.total_staked;
        if total_staked <= 0.0 {
            return total_gas_paid;
        }

        let royalty_amount = total_gas_paid * self.config.royalty_rate;
        let validator_cut = total_gas_paid - royalty_amount;

        // Collect stakers info to avoid borrow conflicts
        let stakers: Vec<(String, f64)> = pool.stakers.iter().map(|(k, &v)| (k.clone(), v)).collect();

        for (staker, staked_val) in stakers {
            self.ensure_balance_record(&staker);
            let share_ratio = staked_val / total_staked;
            let staker_royalty = royalty_amount * share_ratio;
            if let Some(m) = self.balances.get_mut(&staker) {
                if let Some(val) = m.get_mut("FLOP") {
                    *val += staker_royalty;
                }
            }
        }

        validator_cut
    }

    pub fn run_vcg_attention_auction(&mut self, bids: &[AttentionBid]) -> Vec<AttentionBid> {
        let mut valid_bids = Vec::new();
        for bid in bids {
            self.ensure_balance_record(&bid.address);
            let bal = self.get_balance(&bid.address, "ATTN");
            if bid.bid >= 0.0 && bid.bid.is_finite() && bal >= bid.bid {
                valid_bids.push(bid.clone());
            }
        }

        valid_bids.sort_by(|a, b| {
            b.bid.partial_cmp(&a.bid).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.tx_id.cmp(&b.tx_id))
        });

        valid_bids
    }

    pub fn stake_validator(&mut self, address: &str, amount: f64) -> bool {
        if amount <= 0.0 || !amount.is_finite() {
            return false;
        }
        self.ensure_balance_record(address);
        let flop_bal = self.get_balance(address, "FLOP");
        if flop_bal < amount {
            return false;
        }

        if let Some(m) = self.balances.get_mut(address) {
            if let Some(val) = m.get_mut("FLOP") {
                *val -= amount;
            }
        }

        let entry = self.validators.entry(address.to_string()).or_insert(0.0);
        *entry += amount;
        true
    }

    pub fn unstake_validator(&mut self, address: &str, amount: f64) -> bool {
        if amount <= 0.0 || !amount.is_finite() {
            return false;
        }
        let staked = self.validators.get(address).cloned().unwrap_or(0.0);
        if staked < amount {
            return false;
        }

        if let Some(entry) = self.validators.get_mut(address) {
            *entry -= amount;
        }

        if let Some(&val) = self.validators.get(address) {
            if val <= 0.0 {
                self.validators.remove(address);
            }
        }

        self.ensure_balance_record(address);
        if let Some(m) = self.balances.get_mut(address) {
            if let Some(val) = m.get_mut("FLOP") {
                *val += amount;
            }
        }

        true
    }

    pub fn slash_validator(&mut self, address: &str, percentage: f64) -> f64 {
        let staked = self.validators.get(address).cloned().unwrap_or(0.0);
        if staked <= 0.0 {
            return 0.0;
        }

        let slashed_amount = staked * percentage;
        if let Some(entry) = self.validators.get_mut(address) {
            *entry -= slashed_amount;
        }

        if let Some(&val) = self.validators.get(address) {
            if val <= 0.0 {
                self.validators.remove(address);
            }
        }

        slashed_amount
    }
}
