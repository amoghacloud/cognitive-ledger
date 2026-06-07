use std::sync::Arc;
use tokio::sync::RwLock;
use crate::blockchain::{Blockchain, Block, Transaction};
use axum::{
    routing::get,
    Router, Json,
    extract::State,
    response::IntoResponse,
    http::StatusCode,
};
use serde_json::json;

pub struct GossipNodeState {
    pub blockchain: Arc<RwLock<Blockchain>>,
    pub peers: Arc<RwLock<Vec<String>>>,
    pub db_path: String,
}

pub async fn start_node_server(
    state: Arc<GossipNodeState>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = Router::new()
        .route("/blocks", get(handle_get_blocks).post(handle_post_block))
        .route("/transactions", get(handle_get_transactions).post(handle_post_transaction))
        .route("/sync", get(handle_get_sync))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    println!("[P2P Server] Starting node API on http://{}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

async fn handle_get_blocks(State(state): State<Arc<GossipNodeState>>) -> impl IntoResponse {
    let bc = state.blockchain.read().await;
    Json(bc.chain.clone())
}

async fn handle_get_transactions(State(state): State<Arc<GossipNodeState>>) -> impl IntoResponse {
    let bc = state.blockchain.read().await;
    Json(bc.pending_transactions.clone())
}

async fn handle_get_sync(State(state): State<Arc<GossipNodeState>>) -> impl IntoResponse {
    let bc = state.blockchain.read().await;
    Json(bc.chain.clone())
}

async fn handle_post_transaction(
    State(state): State<Arc<GossipNodeState>>,
    Json(tx): Json<Transaction>,
) -> impl IntoResponse {
    let mut bc = state.blockchain.write().await;
    let signature = tx.signature.clone().unwrap_or_default();
    if bc.add_transaction(tx.clone()) {
        drop(bc); // release lock before network IO
        let peers = state.peers.read().await.clone();
        tokio::spawn(async move {
            gossip_transaction(peers, tx).await;
        });
        (StatusCode::OK, Json(json!({"success": true, "tx_id": signature})))
    } else {
        (StatusCode::BAD_REQUEST, Json(json!({"success": false, "error": "Invalid transaction or duplicate"})))
    }
}

async fn handle_post_block(
    State(state): State<Arc<GossipNodeState>>,
    Json(block): Json<Block>,
) -> impl IntoResponse {
    let mut bc = state.blockchain.write().await;
    if bc.chain.iter().any(|b| b.hash == block.hash) {
        return (StatusCode::OK, Json(json!({"success": true, "message": "Block already exists"})));
    }

    if bc.validate_block(&block, true) {
        let storage = crate::storage::ChainStorage::new(state.db_path.clone());
        let _ = storage.save(&bc);
        drop(bc); // release lock before network IO

        let peers = state.peers.read().await.clone();
        tokio::spawn(async move {
            gossip_block(peers, block).await;
        });
        (StatusCode::OK, Json(json!({"success": true})))
    } else {
        (StatusCode::BAD_REQUEST, Json(json!({"success": false, "error": "Invalid block or validation failed"})))
    }
}

async fn gossip_transaction(peers: Vec<String>, tx: Transaction) {
    let client = reqwest::Client::new();
    for peer in peers {
        let url = format!("{}/transactions", peer);
        let _ = client.post(&url).json(&tx).send().await;
    }
}

async fn gossip_block(peers: Vec<String>, block: Block) {
    let client = reqwest::Client::new();
    for peer in peers {
        let url = format!("{}/blocks", peer);
        let _ = client.post(&url).json(&block).send().await;
    }
}

pub async fn sync_with_peers(state: Arc<GossipNodeState>) {
    let peers = state.peers.read().await.clone();
    let client = reqwest::Client::new();
    let mut best_chain: Option<Vec<Block>> = None;
    let mut max_len = {
        let bc = state.blockchain.read().await;
        bc.chain.len()
    };

    for peer in peers {
        let url = format!("{}/sync", peer);
        if let Ok(res) = client.get(&url).send().await {
            if let Ok(chain) = res.json::<Vec<Block>>().await {
                if chain.len() > max_len {
                    max_len = chain.len();
                    best_chain = Some(chain);
                }
            }
        }
    }

    if let Some(chain) = best_chain {
        let mut bc = state.blockchain.write().await;
        if bc.rebuild_from_chain(&chain).is_ok() {
            println!("[P2P Sync] Synced longer chain of length {}", chain.len());
            let storage = crate::storage::ChainStorage::new(state.db_path.clone());
            let _ = storage.save(&bc);
        } else {
            println!("[P2P Sync] Failed to validate longer chain from peer.");
        }
    }
}
