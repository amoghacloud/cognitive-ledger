use crate::blockchain::{Blockchain, Block};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct ChainStorage {
    pub file_path: String,
}

impl ChainStorage {
    pub fn new(file_path: String) -> Self {
        Self { file_path }
    }

    pub fn save(&self, blockchain: &Blockchain) -> Result<(), String> {
        let serialized = serde_json::to_string_pretty(&blockchain.chain)
            .map_err(|e| e.to_string())?;

        let temp_path = format!("{}.tmp", self.file_path);
        {
            let mut file = File::create(&temp_path).map_err(|e| e.to_string())?;
            file.write_all(serialized.as_bytes()).map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
        }
        
        std::fs::rename(temp_path, &self.file_path).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load(&self, blockchain: &mut Blockchain) -> bool {
        let path = Path::new(&self.file_path);
        if !path.exists() {
            return false;
        }

        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };

        let mut content = String::new();
        if file.read_to_string(&mut content).is_err() {
            return false;
        }

        let chain: Vec<Block> = match serde_json::from_str(&content) {
            Ok(c) => c,
            Err(_) => return false,
        };

        blockchain.rebuild_from_chain(&chain).is_ok()
    }
}
