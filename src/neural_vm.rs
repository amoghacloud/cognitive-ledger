struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let mut rng = SimpleRng { state: seed };
        for _ in 0..10 {
            rng.next_u32();
        }
        rng
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 32) as u32
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / (u32::MAX as f64 + 1.0)
    }

    fn next_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + z0 * std_dev
    }
}

pub struct NeuralModel {
    pub model_id: String,
    pub seed: u64,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub w1: Vec<f64>, // flat row-major: input_dim * hidden_dim
    pub b1: Vec<f64>, // hidden_dim
    pub w2: Vec<f64>, // flat row-major: hidden_dim * output_dim
    pub b2: Vec<f64>, // output_dim
}

impl NeuralModel {
    pub fn new(model_id: String, seed: u64, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = SimpleRng::new(seed);
        
        let mut w1 = Vec::with_capacity(input_dim * hidden_dim);
        for _ in 0..(input_dim * hidden_dim) {
            w1.push(rng.next_normal(0.0, 0.1));
        }

        let mut b1 = Vec::with_capacity(hidden_dim);
        for _ in 0..hidden_dim {
            b1.push(rng.next_normal(0.0, 0.1));
        }

        let mut w2 = Vec::with_capacity(hidden_dim * output_dim);
        for _ in 0..(hidden_dim * output_dim) {
            w2.push(rng.next_normal(0.0, 0.1));
        }

        let mut b2 = Vec::with_capacity(output_dim);
        for _ in 0..output_dim {
            b2.push(rng.next_normal(0.0, 0.1));
        }

        Self {
            model_id,
            seed,
            input_dim,
            hidden_dim,
            output_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }

    pub fn forward(&self, x: &[f64], quantize: bool) -> Result<(Vec<f64>, f64, (f64, f64)), String> {
        if x.len() != self.input_dim {
            return Err(format!("Input dimension mismatch. Expected {}, got {}", self.input_dim, x.len()));
        }

        let logits: Vec<f64> = if quantize {
            let scale = 127.0;
            
            // Quantize inputs, weights, biases
            let x_int: Vec<i8> = x.iter()
                .map(|&v| (v * scale).round().clamp(-128.0, 127.0) as i8)
                .collect();
                
            let w1_int: Vec<i8> = self.w1.iter()
                .map(|&v| (v * scale).round().clamp(-128.0, 127.0) as i8)
                .collect();
                
            let b1_int: Vec<i32> = self.b1.iter()
                .map(|&v| (v * scale * scale).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
                .collect();

            // Layer 1
            let mut h_int = Vec::with_capacity(self.hidden_dim);
            for j in 0..self.hidden_dim {
                let mut sum = b1_int[j];
                for i in 0..self.input_dim {
                    sum += (x_int[i] as i32) * (w1_int[i * self.hidden_dim + j] as i32);
                }
                h_int.push(sum);
            }

            // ReLU
            let h_relu_int: Vec<i32> = h_int.into_iter().map(|val| val.max(0)).collect();

            // Rescale and re-quantize to INT8 for Layer 2
            let h_scaled_int: Vec<i8> = h_relu_int.iter()
                .map(|&val| {
                    let h_scaled = (val as f64) / (scale * scale);
                    (h_scaled * scale).round().clamp(-128.0, 127.0) as i8
                })
                .collect();

            let w2_int: Vec<i8> = self.w2.iter()
                .map(|&v| (v * scale).round().clamp(-128.0, 127.0) as i8)
                .collect();

            let b2_int: Vec<i32> = self.b2.iter()
                .map(|&v| (v * scale * scale).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
                .collect();

            // Layer 2
            let mut logits_int = Vec::with_capacity(self.output_dim);
            for k in 0..self.output_dim {
                let mut sum = b2_int[k];
                for j in 0..self.hidden_dim {
                    sum += (h_scaled_int[j] as i32) * (w2_int[j * self.output_dim + k] as i32);
                }
                logits_int.push(sum);
            }

            logits_int.iter().map(|&val| (val as f64) / (scale * scale)).collect()
        } else {
            // Float32 Layer 1
            let mut h = Vec::with_capacity(self.hidden_dim);
            for j in 0..self.hidden_dim {
                let mut sum = self.b1[j];
                for i in 0..self.input_dim {
                    sum += x[i] * self.w1[i * self.hidden_dim + j];
                }
                h.push(sum);
            }

            // ReLU
            let h_relu: Vec<f64> = h.into_iter().map(|val| val.max(0.0)).collect();

            // Layer 2
            let mut logits = Vec::with_capacity(self.output_dim);
            for k in 0..self.output_dim {
                let mut sum = self.b2[k];
                for j in 0..self.hidden_dim {
                    sum += h_relu[j] * self.w2[j * self.output_dim + k];
                }
                logits.push(sum);
            }
            logits
        };

        // Softmax
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();
        let probs: Vec<f64> = exp_logits.iter().map(|&e| e / sum_exp).collect();

        let confidence = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Shannon Entropy uncertainty
        let mut entropy = 0.0;
        for &p in &probs {
            entropy -= p * (p + 1e-9).ln();
        }

        let low_bound = (confidence - entropy * 0.1).max(0.0);
        let high_bound = (confidence + entropy * 0.1).min(1.0);

        Ok((logits, confidence, (low_bound, high_bound)))
    }
}

pub fn calculate_gas(input_len: usize, model: &NeuralModel, context_len: usize, vram_time: f64) -> usize {
    let layer1_flops = 2 * model.input_dim * model.hidden_dim + model.hidden_dim;
    let layer2_flops = 2 * model.hidden_dim * model.output_dim + model.output_dim;
    let total_flops = layer1_flops + layer2_flops;

    let compute_gas = input_len * total_flops;
    let memory_gas = (context_len as f64 * vram_time * 10.0) as usize;

    (compute_gas + memory_gas).max(10)
}
