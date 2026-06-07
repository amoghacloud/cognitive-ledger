use p256::ecdsa::signature::{Signer, Verifier};
use p256::pkcs8::{EncodePrivateKey, EncodePublicKey, DecodePrivateKey, DecodePublicKey};
use sha2::{Digest, Sha256};

pub fn generate_keypair() -> Result<(String, String), String> {
    let signing_key = p256::ecdsa::SigningKey::random(&mut rand::thread_rng());
    let private_pem = signing_key
        .to_pkcs8_pem(p256::pkcs8::LineEnding::LF)
        .map_err(|e| e.to_string())?
        .to_string();
    
    let verifying_key = p256::ecdsa::VerifyingKey::from(&signing_key);
    let public_pem = verifying_key
        .to_public_key_pem(p256::pkcs8::LineEnding::LF)
        .map_err(|e| e.to_string())?;

    Ok((private_pem, public_pem))
}

pub fn sign_data(private_key_pem_str: &str, data_str: &str) -> Result<String, String> {
    let signing_key = p256::ecdsa::SigningKey::from_pkcs8_pem(private_key_pem_str)
        .map_err(|e| e.to_string())?;
    let signature: p256::ecdsa::Signature = signing_key.sign(data_str.as_bytes());
    Ok(hex::encode(signature.to_bytes()))
}

pub fn verify_signature(public_key_pem_str: &str, signature_hex: &str, data_str: &str) -> bool {
    let verifying_key = match p256::ecdsa::VerifyingKey::from_public_key_pem(public_key_pem_str) {
        Ok(k) => k,
        Err(_) => return false,
    };
    let sig_bytes = match hex::decode(signature_hex) {
        Ok(b) => b,
        Err(_) => return false,
    };
    let signature = match p256::ecdsa::Signature::from_slice(&sig_bytes) {
        Ok(s) => s,
        Err(_) => return false,
    };
    verifying_key.verify(data_str.as_bytes(), &signature).is_ok()
}

pub fn hash_data(data_str: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data_str.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn merkle_root(data_list: &[String]) -> String {
    if data_list.is_empty() {
        return hash_data("");
    }

    let mut current_level: Vec<String> = data_list.iter().map(|item| hash_data(item)).collect();

    while current_level.len() > 1 {
        let mut next_level = Vec::new();
        for i in (0..current_level.len()).step_by(2) {
            if i + 1 < current_level.len() {
                let combined = format!("{}{}", current_level[i], current_level[i + 1]);
                next_level.push(hash_data(&combined));
            } else {
                // Odd element: duplicate and hash
                let combined = format!("{}{}", current_level[i], current_level[i]);
                next_level.push(hash_data(&combined));
            }
        }
        current_level = next_level;
    }

    current_level[0].clone()
}
