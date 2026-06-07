import json
from crypto_utils import generate_keypair, hash_data, merkle_root
from neural_vm import NeuralModel

class AIAgent:
    def __init__(self, name: str, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 10, seed: int = 42):
        """
        Represents an autonomous AI agent with a cryptographic wallet, 
        hereditary Model DNA, and deterministic execution weights.
        """
        self.name = name
        self.private_key, self.public_key = generate_keypair()
        self.seed = seed
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Instantiate deterministic neural weights representing agent cognition
        self.model = NeuralModel(
            model_id=self.public_key,
            seed=seed,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Compute Model DNA as content-addressed Merkle Root
        w1_hash = hash_data(self.model.w1.tobytes().hex())
        w2_hash = hash_data(self.model.w2.tobytes().hex())
        self.model_dna = merkle_root([
            w1_hash,
            w2_hash,
            str(self.input_dim),
            str(self.hidden_dim),
            str(self.output_dim),
            str(seed)
        ])

    def to_attestation_certificate(self, enclave_measurement: str) -> dict:
        """
        Returns a simulated Model Attestation Certificate signed inside TEE enclave.
        """
        return {
            "model_id": self.public_key,
            "model_dna": self.model_dna,
            "enclave_measurement": enclave_measurement,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim
        }
