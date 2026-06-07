use crate::config::ChainConfig;
use crate::tokenomics::{TokenomicsManager, AttentionBid};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Transaction {
    pub sender: String,
    pub receiver: String,
    pub amount: f64,
    #[serde(default = "default_token_type")]
    pub token_type: String, // "FLOP", "DATA", "ATTN"
    #[serde(default = "default_tx_type")]
    pub tx_type: String,    // "TRANSFER", "REGISTER_MODEL", "INFER_CONTRACT", "STAKE_DATA", "STAKE_VALIDATOR", "UNSTAKE_VALIDATOR"
    pub model_id: Option<String>,
    pub model_dna: Option<String>,
    pub input_dim: Option<usize>,
    pub hidden_dim: Option<usize>,
    pub output_dim: Option<usize>,
    pub seed: Option<u64>,
    pub dataset_id: Option<String>,
    pub inputs: Option<Vec<f64>>,
    pub gas_limit: Option<usize>,
    pub max_fee: Option<f64>,
    pub attention_bid: Option<f64>,
    pub signature: Option<String>,
    pub nonce: u64,
}

fn default_token_type() -> String {
    "FLOP".to_string()
}

fn default_tx_type() -> String {
    "TRANSFER".to_string()
}

impl Transaction {
    pub fn to_signable_str(&self) -> String {
        let mut val = serde_json::to_value(self).unwrap();
        if let Some(obj) = val.as_object_mut() {
            obj.remove("signature");
        }
        serde_json::to_string(&val).unwrap()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Block {
    pub index: usize,
    pub timestamp: f64,
    pub transactions: Vec<Transaction>,
    pub previous_hash: String,
    pub validator: String,
    pub proof_of_inference: HashMap<String, serde_json::Value>,
    pub state_root: String,
    pub nonce: u64,
    #[serde(default)]
    pub hash: String,
}

impl Block {
    pub fn calculate_hash(&self) -> String {
        let mut val = serde_json::to_value(self).unwrap();
        if let Some(obj) = val.as_object_mut() {
            obj.remove("hash");
        }
        let serialized = serde_json::to_string(&val).unwrap();
        crate::crypto::hash_data(&serialized)
    }
}

pub struct Blockchain {
    pub config: ChainConfig,
    pub chain: Vec<Block>,
    pub pending_transactions: Vec<Transaction>,
    pub tokenomics: TokenomicsManager,
    pub registered_models: HashMap<String, serde_json::Value>,
    pub account_nonces: HashMap<String, u64>,
    pub difficulty: usize,
    pub on_commit: Option<Box<dyn Fn(&Blockchain) + Send + Sync>>,
}

impl Blockchain {
    pub fn new(config: ChainConfig, genesis_balances: Option<HashMap<String, HashMap<String, f64>>>) -> Self {
        let mut genesis_bal = genesis_balances.clone();
        if genesis_bal.is_none() {
            if let Some(ref founder) = config.founder_address {
                let mut map = HashMap::new();
                map.insert(founder.clone(), config.founder_initial_balances.clone());
                genesis_bal = Some(map);
            }
        }

        let mut tokenomics = TokenomicsManager::new(config.clone(), genesis_bal);
        if let Some(ref founder) = config.founder_address {
            tokenomics.validators.insert(founder.clone(), config.min_validator_stake);
        }

        let mut blockchain = Self {
            config,
            chain: Vec::new(),
            pending_transactions: Vec::new(),
            tokenomics,
            registered_models: HashMap::new(),
            account_nonces: HashMap::new(),
            difficulty: 0,
            on_commit: None,
        };

        blockchain.difficulty = blockchain.config.difficulty;
        blockchain.create_genesis_block();
        blockchain
    }

    fn create_genesis_block(&mut self) {
        let genesis_tx = Transaction {
            sender: "GENESIS".to_string(),
            receiver: "GENESIS".to_string(),
            amount: 0.0,
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
            signature: Some("GENESIS_SIG".to_string()),
            nonce: 0,
        };

        let state_root = self.get_state_hash();
        let mut genesis_block = Block {
            index: 0,
            timestamp: 0.0,
            transactions: vec![genesis_tx],
            previous_hash: "0".to_string(),
            validator: "GENESIS".to_string(),
            proof_of_inference: HashMap::new(),
            state_root,
            nonce: 0,
            hash: "".to_string(),
        };

        genesis_block.hash = genesis_block.calculate_hash();
        self.chain.push(genesis_block);
    }

    pub fn select_leader(&self, block_index: usize, prev_hash: &str, state_validators: &HashMap<String, f64>) -> String {
        if state_validators.is_empty() {
            return self.config.founder_address.clone().unwrap_or_else(|| "GENESIS".to_string());
        }

        let mut sorted_validators: Vec<String> = state_validators.keys().cloned().collect();
        sorted_validators.sort();

        let total_stake: f64 = sorted_validators.iter()
            .map(|v| state_validators.get(v).cloned().unwrap_or(0.0))
            .sum();
        
        if total_stake <= 0.0 {
            return self.config.founder_address.clone().unwrap_or_else(|| "GENESIS".to_string());
        }

        let seed_hash = crate::crypto::hash_data(&format!("{}-{}", prev_hash, block_index));
        let seed_int = u64::from_str_radix(&seed_hash[..16], 16).unwrap_or(0);

        let target = (seed_int % (total_stake * 100.0) as u64) as f64 / 100.0;
        let mut current = 0.0;
        for val in sorted_validators.iter() {
            current += state_validators.get(val).cloned().unwrap_or(0.0);
            if current >= target {
                return val.clone();
            }
        }
        sorted_validators.last().unwrap().clone()
    }

    pub fn rebuild_from_chain(&mut self, chain: &[Block]) -> Result<(), String> {
        if chain.is_empty() {
            return Err("Cannot rebuild from empty chain".to_string());
        }

        let expected_genesis_hash = self.chain[0].hash.clone();
        if chain[0].hash != expected_genesis_hash {
            return Err("Genesis block does not match this chain configuration".to_string());
        }

        let mut candidate = Self::new(self.config.clone(), None);
        candidate.chain = vec![chain[0].clone()];

        for block in &chain[1..] {
            if !candidate.validate_block(block, false) {
                return Err(format!("Invalid block at index {}", block.index));
            }
        }

        self.chain = candidate.chain;
        self.tokenomics = candidate.tokenomics;
        self.registered_models = candidate.registered_models;
        self.account_nonces = candidate.account_nonces;
        Ok(())
    }

    pub fn get_state_hash(&self) -> String {
        let mut state_repr = serde_json::Map::new();
        state_repr.insert("balances".to_string(), serde_json::to_value(&self.tokenomics.balances).unwrap());
        state_repr.insert("data_equity".to_string(), serde_json::to_value(&self.tokenomics.data_equity).unwrap());
        state_repr.insert("model_datasets".to_string(), serde_json::to_value(&self.tokenomics.model_datasets).unwrap());
        state_repr.insert("registered_models".to_string(), serde_json::to_value(&self.registered_models).unwrap());
        state_repr.insert("account_nonces".to_string(), serde_json::to_value(&self.account_nonces).unwrap());

        let val = serde_json::Value::Object(state_repr);
        let serialized = serde_json::to_string(&val).unwrap();
        crate::crypto::hash_data(&serialized)
    }

    pub fn add_transaction(&mut self, tx: Transaction) -> bool {
        if self.pending_transactions.len() >= self.config.max_mempool_size {
            return false;
        }

        if !self.validate_transaction_shape(&tx) {
            return false;
        }

        if tx.sender != "GENESIS" {
            let sig = match &tx.signature {
                None => return false,
                Some(s) => s,
            };
            if !crate::crypto::verify_signature(&tx.sender, sig, &tx.to_signable_str()) {
                return false;
            }
        }

        let current_nonce = self.account_nonces.get(&tx.sender).cloned().unwrap_or(0);
        if tx.nonce <= current_nonce {
            return false;
        }

        self.tokenomics.ensure_balance_record(&tx.sender);

        if tx.tx_type == "TRANSFER" {
            if self.tokenomics.get_balance(&tx.sender, &tx.token_type) < tx.amount {
                return false;
            }
        } else if tx.tx_type == "REGISTER_MODEL" {
            if self.tokenomics.get_balance(&tx.sender, "FLOP") < self.config.model_registration_fee {
                return false;
            }
        } else if tx.tx_type == "INFER_CONTRACT" {
            let limit = tx.gas_limit.unwrap_or(100);
            let fee = tx.max_fee.unwrap_or(1.0);
            let bid = tx.attention_bid.unwrap_or(0.0);
            if self.tokenomics.get_balance(&tx.sender, "FLOP") < (limit as f64 * fee) {
                return false;
            }
            if self.tokenomics.get_balance(&tx.sender, "ATTN") < bid {
                return false;
            }
        } else if tx.tx_type == "STAKE_DATA" {
            if self.tokenomics.get_balance(&tx.sender, "DATA") < tx.amount {
                return false;
            }
        } else if tx.tx_type == "STAKE_VALIDATOR" {
            if self.tokenomics.get_balance(&tx.sender, "FLOP") < tx.amount {
                return false;
            }
        } else if tx.tx_type == "UNSTAKE_VALIDATOR" {
            let staked = self.tokenomics.validators.get(&tx.sender).cloned().unwrap_or(0.0);
            if staked < tx.amount {
                return false;
            }
        }

        // Check for duplicates
        for pending in &self.pending_transactions {
            if pending.signature == tx.signature {
                return false;
            }
            if pending.sender == tx.sender && pending.nonce == tx.nonce {
                return false;
            }
        }

        self.pending_transactions.push(tx);
        true
    }

    pub fn validate_transaction_shape(&self, tx: &Transaction) -> bool {
        let supported_tokens: HashSet<&str> = ["FLOP", "DATA", "ATTN"].iter().cloned().collect();
        let supported_tx_types: HashSet<&str> = [
            "TRANSFER",
            "REGISTER_MODEL",
            "INFER_CONTRACT",
            "STAKE_DATA",
            "STAKE_VALIDATOR",
            "UNSTAKE_VALIDATOR",
        ]
        .iter()
        .cloned()
        .collect();

        if !supported_tx_types.contains(tx.tx_type.as_str()) || !supported_tokens.contains(tx.token_type.as_str()) {
            return false;
        }

        if tx.nonce == 0 {
            return false;
        }
        if tx.amount < 0.0 || !tx.amount.is_finite() {
            return false;
        }
        if let Some(bid) = tx.attention_bid {
            if bid < 0.0 || !bid.is_finite() {
                return false;
            }
        }

        if tx.tx_type == "TRANSFER" {
            return !tx.receiver.is_empty() && tx.amount > 0.0;
        }

        if tx.tx_type == "REGISTER_MODEL" {
            let input = match tx.input_dim {
                Some(i) => i,
                None => return false,
            };
            let hidden = match tx.hidden_dim {
                Some(h) => h,
                None => return false,
            };
            let output = match tx.output_dim {
                Some(o) => o,
                None => return false,
            };

            if tx.model_id.is_none() || tx.model_dna.is_none() || tx.seed.is_none() {
                return false;
            }

            let limit = self.config.max_model_dimension;
            if input == 0 || input > limit || hidden == 0 || hidden > limit || output == 0 || output > limit {
                return false;
            }
            return true;
        }

        if tx.tx_type == "INFER_CONTRACT" {
            if tx.model_id.is_none() || tx.inputs.is_none() {
                return false;
            }
            let inputs = tx.inputs.as_ref().unwrap();
            if inputs.len() > self.config.max_input_length {
                return false;
            }
            if tx.gas_limit.is_none() || tx.gas_limit.unwrap() == 0 {
                return false;
            }
            if tx.max_fee.is_none() || tx.max_fee.unwrap() <= 0.0 || !tx.max_fee.unwrap().is_finite() {
                return false;
            }
            return inputs.iter().all(|&val| val.is_finite());
        }

        if tx.tx_type == "STAKE_DATA" {
            return tx.dataset_id.is_some() && tx.amount > 0.0;
        }

        if tx.tx_type == "STAKE_VALIDATOR" {
            return tx.token_type == "FLOP" && tx.amount >= self.config.min_validator_stake;
        }

        if tx.tx_type == "UNSTAKE_VALIDATOR" {
            return tx.token_type == "FLOP" && tx.amount > 0.0;
        }

        false
    }

    pub fn execute_transaction_on_state(
        &self,
        tx: &Transaction,
        state_tokenomics: &mut TokenomicsManager,
        state_models: &mut HashMap<String, serde_json::Value>,
        state_nonces: &mut HashMap<String, u64>,
        validator_address: &str,
    ) -> Result<serde_json::Value, String> {
        state_tokenomics.ensure_balance_record(&tx.sender);
        if !tx.receiver.is_empty() {
            state_tokenomics.ensure_balance_record(&tx.receiver);
        }

        let mut result = serde_json::json!({});

        if tx.tx_type == "TRANSFER" {
            let success = state_tokenomics.transfer(&tx.sender, &tx.receiver, tx.amount, &tx.token_type);
            if !success {
                return Err("Insufficient balance for transfer".to_string());
            }
        } else if tx.tx_type == "REGISTER_MODEL" {
            let model_id = tx.model_id.clone().ok_or("Missing model_id")?;
            if state_models.contains_key(&model_id) {
                return Err("Model already registered".to_string());
            }

            let success = state_tokenomics.transfer(
                &tx.sender,
                validator_address,
                self.config.model_registration_fee,
                "FLOP",
            );
            if !success {
                return Err("Insufficient balance for model registration fee".to_string());
            }

            let model_info = serde_json::json!({
                "model_id": model_id,
                "model_dna": tx.model_dna.clone().ok_or("Missing model_dna")?,
                "input_dim": tx.input_dim.ok_or("Missing input_dim")?,
                "hidden_dim": tx.hidden_dim.ok_or("Missing hidden_dim")?,
                "output_dim": tx.output_dim.ok_or("Missing output_dim")?,
                "seed": tx.seed.ok_or("Missing seed")?,
                "dataset_id": tx.dataset_id.clone()
            });

            state_models.insert(model_id.clone(), model_info);
            if let Some(ref ds_id) = tx.dataset_id {
                state_tokenomics.bind_model_to_dataset(&model_id, ds_id);
            }
        } else if tx.tx_type == "INFER_CONTRACT" {
            let model_id = tx.model_id.clone().ok_or("Missing model_id")?;
            let model_info = state_models.get(&model_id).ok_or_else(|| format!("Model {} not registered", model_id))?;

            let m_id = model_info["model_id"].as_str().ok_or("Invalid model_id")?.to_string();
            let seed = model_info["seed"].as_u64().ok_or("Invalid seed")?;
            let input_dim = model_info["input_dim"].as_u64().ok_or("Invalid input_dim")? as usize;
            let hidden_dim = model_info["hidden_dim"].as_u64().ok_or("Invalid hidden_dim")? as usize;
            let output_dim = model_info["output_dim"].as_u64().ok_or("Invalid output_dim")? as usize;

            let model = crate::neural_vm::NeuralModel::new(m_id, seed, input_dim, hidden_dim, output_dim);
            let inputs = tx.inputs.clone().ok_or("Missing inputs")?;
            let (logits, confidence, bounds) = model.forward(&inputs, true)?;

            let gas_used = crate::neural_vm::calculate_gas(inputs.len(), &model, 1, 0.1);
            if let Some(limit) = tx.gas_limit {
                if gas_used > limit {
                    return Err("Gas limit exceeded".to_string());
                }
            }

            let total_fee = gas_used as f64 * tx.max_fee.unwrap_or(1.0);
            if state_tokenomics.get_balance(&tx.sender, "FLOP") < total_fee {
                return Err("Insufficient FLOP balance for gas".to_string());
            }

            // Deduct fee
            if let Some(m) = state_tokenomics.balances.get_mut(&tx.sender) {
                if let Some(val) = m.get_mut("FLOP") {
                    *val -= total_fee;
                }
            }

            // Route royalties
            let validator_share = state_tokenomics.route_execution_royalties(&model_id, total_fee);
            
            // Pay validator
            state_tokenomics.ensure_balance_record(validator_address);
            if let Some(m) = state_tokenomics.balances.get_mut(validator_address) {
                if let Some(val) = m.get_mut("FLOP") {
                    *val += validator_share;
                }
            }

            // Pay attention priority bid
            if let Some(attn_bid) = tx.attention_bid {
                if attn_bid > 0.0 {
                    if state_tokenomics.get_balance(&tx.sender, "ATTN") < attn_bid {
                        return Err("Insufficient ATTN balance for bid".to_string());
                    }
                    if let Some(m) = state_tokenomics.balances.get_mut(&tx.sender) {
                        if let Some(val) = m.get_mut("ATTN") {
                            *val -= attn_bid;
                        }
                    }
                    state_tokenomics.ensure_balance_record(validator_address);
                    if let Some(m) = state_tokenomics.balances.get_mut(validator_address) {
                        if let Some(val) = m.get_mut("ATTN") {
                            *val += attn_bid;
                        }
                    }
                }
            }

            result = serde_json::json!({
                "logits": logits,
                "confidence": confidence,
                "uncertainty_bounds": [bounds.0, bounds.1],
                "gas_used": gas_used,
                "success": true
            });
        } else if tx.tx_type == "STAKE_DATA" {
            let dataset_id = tx.dataset_id.clone().ok_or("Missing dataset_id")?;
            let success = state_tokenomics.stake_data_equity(&tx.sender, &dataset_id, tx.amount);
            if !success {
                return Err("Insufficient DATA balance for staking".to_string());
            }
        } else if tx.tx_type == "STAKE_VALIDATOR" {
            let success = state_tokenomics.stake_validator(&tx.sender, tx.amount);
            if !success {
                return Err("Insufficient FLOP balance for validator staking".to_string());
            }
        } else if tx.tx_type == "UNSTAKE_VALIDATOR" {
            let success = state_tokenomics.unstake_validator(&tx.sender, tx.amount);
            if !success {
                return Err("Insufficient staked balance for validator unstaking".to_string());
            }
        }

        state_nonces.insert(tx.sender.clone(), tx.nonce);
        Ok(result)
    }

    pub fn mine_block(&mut self, validator_address: &str) -> Option<Block> {
        if self.pending_transactions.is_empty() {
            return None;
        }

        let mut bids = Vec::new();
        for tx in &self.pending_transactions {
            bids.push(AttentionBid {
                address: tx.sender.clone(),
                bid: tx.attention_bid.unwrap_or(0.0),
                tx_id: tx.signature.clone().unwrap_or_default(),
            });
        }

        let sorted_bids = self.tokenomics.run_vcg_attention_auction(&bids);
        let priority_sigs: HashSet<String> = sorted_bids.iter().map(|b| b.tx_id.clone()).collect();
        let priority_order: Vec<String> = sorted_bids.iter().map(|b| b.tx_id.clone()).collect();

        let mut priority_txs = Vec::new();
        let mut other_txs = Vec::new();
        for tx in &self.pending_transactions {
            if let Some(ref sig) = tx.signature {
                if priority_sigs.contains(sig) {
                    priority_txs.push(tx.clone());
                } else {
                    other_txs.push(tx.clone());
                }
            } else {
                other_txs.push(tx.clone());
            }
        }

        // Sort priority txs by VCG auction order
        priority_txs.sort_by_key(|tx| {
            let sig = tx.signature.clone().unwrap_or_default();
            priority_order.iter().position(|s| s == &sig).unwrap_or(usize::MAX)
        });

        let mut candidate_txs: Vec<Transaction> = priority_txs.into_iter().chain(other_txs.into_iter()).collect();
        candidate_txs.truncate(self.config.max_transactions_per_block);

        let mut temp_tokenomics = self.tokenomics.clone();
        let mut temp_models = self.registered_models.clone();
        let mut temp_nonces = self.account_nonces.clone();

        let mut executed_txs = Vec::new();
        let mut proof_of_inference = HashMap::new();

        for tx in candidate_txs {
            match self.execute_transaction_on_state(&tx, &mut temp_tokenomics, &mut temp_models, &mut temp_nonces, validator_address) {
                Ok(res) => {
                    executed_txs.push(tx.clone());
                    if tx.tx_type == "INFER_CONTRACT" {
                        if let Some(ref sig) = tx.signature {
                            proof_of_inference.insert(sig.clone(), res);
                        }
                    }
                }
                Err(e) => {
                    println!("[Block Miner] Tx execution failed: {}", e);
                }
            }
        }

        if executed_txs.is_empty() {
            // Remove the failed txs from mempool so they don't block
            self.pending_transactions.retain(|t| {
                !executed_txs.iter().any(|e_tx| e_tx.signature == t.signature)
            });
            return None;
        }

        let prev_block = self.chain.last().unwrap();
        
        // Slot leader check
        let expected_leader = self.select_leader(prev_block.index + 1, &prev_block.hash, &self.tokenomics.validators);
        if validator_address != expected_leader {
            panic!("Not slot leader. Leader: {}", expected_leader);
        }

        self.tokenomics = temp_tokenomics;
        self.registered_models = temp_models;
        self.account_nonces = temp_nonces;
        let state_root = self.get_state_hash();

        let mut new_block = Block {
            index: prev_block.index + 1,
            timestamp: chrono::Utc::now().timestamp() as f64,
            transactions: executed_txs.clone(),
            previous_hash: prev_block.hash.clone(),
            validator: validator_address.to_string(),
            proof_of_inference,
            state_root,
            nonce: 0,
            hash: "".to_string(),
        };

        new_block.hash = new_block.calculate_hash();

        let executed_sigs: HashSet<String> = executed_txs.iter().flat_map(|tx| tx.signature.clone()).collect();
        self.pending_transactions.retain(|tx| {
            tx.signature.as_ref().map(|sig| !executed_sigs.contains(sig)).unwrap_or(true)
        });

        self.chain.push(new_block.clone());
        self._commit();
        Some(new_block)
    }

    pub fn validate_block(&mut self, block: &Block, persist: bool) -> bool {
        if self.chain.is_empty() {
            return false;
        }
        let prev_block = self.chain.last().unwrap();

        if block.index != prev_block.index + 1 {
            return false;
        }
        if block.previous_hash != prev_block.hash {
            return false;
        }
        if block.hash != block.calculate_hash() {
            return false;
        }

        let expected_leader = self.select_leader(block.index, &prev_block.hash, &self.tokenomics.validators);
        if block.validator != expected_leader {
            return false;
        }

        let mut temp_tokenomics = self.tokenomics.clone();
        let mut temp_models = self.registered_models.clone();
        let mut temp_nonces = self.account_nonces.clone();

        for tx in &block.transactions {
            if !self.validate_transaction_shape(tx) {
                self.tokenomics.slash_validator(&block.validator, self.config.validator_slashing_rate);
                self._commit();
                return false;
            }

            if tx.sender != "GENESIS" {
                let sig = match &tx.signature {
                    None => {
                        self.tokenomics.slash_validator(&block.validator, self.config.validator_slashing_rate);
                        self._commit();
                        return false;
                    }
                    Some(s) => s,
                };
                if !crate::crypto::verify_signature(&tx.sender, sig, &tx.to_signable_str()) {
                    self.tokenomics.slash_validator(&block.validator, self.config.validator_slashing_rate);
                    self._commit();
                    return false;
                }
            }

            let curr_nonce = temp_nonces.get(&tx.sender).cloned().unwrap_or(0);
            if tx.nonce <= curr_nonce {
                self.tokenomics.slash_validator(&block.validator, self.config.validator_slashing_rate);
                self._commit();
                return false;
            }

            match self.execute_transaction_on_state(tx, &mut temp_tokenomics, &mut temp_models, &mut temp_nonces, &block.validator) {
                Err(_) => {
                    self.tokenomics.slash_validator(&block.validator, self.config.validator_slashing_rate);
                    self._commit();
                    return false;
                }
                Ok(res) => {
                    if tx.tx_type == "INFER_CONTRACT" {
                        let sig = tx.signature.clone().unwrap_or_default();
                        let recorded_res = match block.proof_of_inference.get(&sig) {
                            None => {
                                self.tokenomics.slash_validator(&block.validator, self.config.validator_slashing_rate);
                                self._commit();
                                return false;
                            }
                            Some(val) => val,
                        };

                        if recorded_res["logits"] != res["logits"]
                            || recorded_res["confidence"] != res["confidence"]
                            || recorded_res["uncertainty_bounds"] != res["uncertainty_bounds"]
                        {
                            self.tokenomics.slash_validator(&block.validator, self.config.validator_slashing_rate);
                            self._commit();
                            return false;
                        }
                    }
                }
            }
        }

        // Verify state root hash
        let mut state_repr = serde_json::Map::new();
        state_repr.insert("balances".to_string(), serde_json::to_value(&temp_tokenomics.balances).unwrap());
        state_repr.insert("data_equity".to_string(), serde_json::to_value(&temp_tokenomics.data_equity).unwrap());
        state_repr.insert("model_datasets".to_string(), serde_json::to_value(&temp_tokenomics.model_datasets).unwrap());
        state_repr.insert("registered_models".to_string(), serde_json::to_value(&temp_models).unwrap());
        state_repr.insert("account_nonces".to_string(), serde_json::to_value(&temp_nonces).unwrap());

        let val = serde_json::Value::Object(state_repr);
        let computed_state_root = crate::crypto::hash_data(&serde_json::to_string(&val).unwrap());

        if computed_state_root != block.state_root {
            self.tokenomics.slash_validator(&block.validator, self.config.validator_slashing_rate);
            self._commit();
            return false;
        }

        self.tokenomics = temp_tokenomics;
        self.registered_models = temp_models;
        self.account_nonces = temp_nonces;
        self.chain.push(block.clone());

        if persist {
            let block_tx_sigs: HashSet<String> = block.transactions.iter()
                .flat_map(|tx| tx.signature.clone())
                .collect();
            self.pending_transactions.retain(|tx| {
                tx.signature.as_ref().map(|sig| !block_tx_sigs.contains(sig)).unwrap_or(true)
            });
            self._commit();
        }

        true
    }

    fn _commit(&self) {
        if let Some(ref cb) = self.on_commit {
            cb(self);
        }
    }
}
