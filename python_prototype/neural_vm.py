import numpy as np

class NeuralModel:
    def __init__(self, model_id: str, seed: int = 42, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 10):
        """
        Initializes weights deterministically using the seed to ensure
        reproducibility across all nodes on the network (Deterministic Inference Mode).
        """
        self.model_id = model_id
        self.seed = seed
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Deterministic random generator
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.1, (input_dim, hidden_dim))
        self.b1 = rng.normal(0, 0.1, (hidden_dim,))
        self.w2 = rng.normal(0, 0.1, (hidden_dim, output_dim))
        self.b2 = rng.normal(0, 0.1, (output_dim,))

    def forward(self, x: np.ndarray, quantize: bool = False) -> tuple:
        """
        Runs the neural forward pass.
        If quantize is True, runs with strict fixed-point INT8 quantization to avoid FP32 variations.
        
        Returns: 
            logits (list of floats)
            confidence (float, max softmax probability)
            uncertainty_bounds (tuple of float, representing the confidence bounds based on entropy)
        """
        # Ensure input dimensions match model
        if len(x) != self.input_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.input_dim}, got {len(x)}")

        if quantize:
            # Scale float input to INT8 range [-128, 127]
            scale = 127.0
            
            x_int = np.clip(np.round(x * scale), -128, 127).astype(np.int8)
            w1_int = np.clip(np.round(self.w1 * scale), -128, 127).astype(np.int8)
            b1_int = np.clip(np.round(self.b1 * (scale * scale)), -2147483648, 2147483647).astype(np.int32)
            
            # Layer 1 linear projection
            h_int = np.dot(x_int.astype(np.int32), w1_int.astype(np.int32)) + b1_int
            # ReLU activation
            h_relu_int = np.maximum(h_int, 0)
            
            # Rescale before Layer 2 to prevent integer overflow
            h_scaled = h_relu_int / (scale * scale)
            h_scaled_int = np.clip(np.round(h_scaled * scale), -128, 127).astype(np.int8)
            
            w2_int = np.clip(np.round(self.w2 * scale), -128, 127).astype(np.int8)
            b2_int = np.clip(np.round(self.b2 * (scale * scale)), -2147483648, 2147483647).astype(np.int32)
            
            # Layer 2 linear projection
            logits_int = np.dot(h_scaled_int.astype(np.int32), w2_int.astype(np.int32)) + b2_int
            logits = logits_int / (scale * scale)
        else:
            h = np.dot(x, self.w1) + self.b1
            h_relu = np.maximum(h, 0)
            logits = np.dot(h_relu, self.w2) + self.b2
            
        # Calculate Softmax probability distribution
        exp_logits = np.exp(logits - np.max(logits))  # numerically stable softmax
        probs = exp_logits / np.sum(exp_logits)
        
        confidence = float(np.max(probs))
        
        # Uncertainty Estimation using Shannon Entropy of output probability distribution
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        low_bound = max(0.0, confidence - entropy * 0.1)
        high_bound = min(1.0, confidence + entropy * 0.1)
        
        return logits.tolist(), confidence, (low_bound, high_bound)

def calculate_gas(input_len: int, model: NeuralModel, context_len: int, vram_time: float) -> int:
    """
    Computes gas fee in Standard Compute Units (SCUs) using the formula:
    Gas = (Tokens_input * FLOPs) + (Tokens_context * VRAM_time)
    """
    # Approximate floating point operations per token forward pass
    layer1_flops = 2 * model.input_dim * model.hidden_dim + model.hidden_dim
    layer2_flops = 2 * model.hidden_dim * model.output_dim + model.output_dim
    total_flops = layer1_flops + layer2_flops
    
    compute_gas = input_len * total_flops
    memory_gas = int(context_len * vram_time * 10)  # scale cache holding duration
    
    # Return standard compute units (minimum of 10 SCUs to prevent zero-cost operations)
    return max(10, compute_gas + memory_gas)
