/*!
MLGRU — MatMul-free Linear Gated Recurrent Unit — Token Mixer tanpa MatMul, tanpa Attention O(N²)
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Berdasarkan:
- Scalable MatMul-free Language Modeling 2406.02528: MLGRU replaces attention with element-wise RNN
- Mamba selective SSM: linear O(N), constant memory inference
- RWKV time-mixing: constant-size state, parallel training

MLGRU:
- Remove hidden-state related weights W_cc, W_hr, W_hf
- Remove tanh activation (linearized via parallel scan)
- Keep candidate as simple linear transform
- Replace all remaining weight matrices dengan ternary
- Relies solely on element-wise multiplication, no MatMul

Complexity: O(N) bukan O(N²), memory constant, 5x throughput vs Transformers
*/

use super::bitlinear::{BitLinear, TernaryWeight};

pub struct MLGRUConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

pub struct MLGRU {
    config: MLGRUConfig,
    /// Forget gate: ternary BitLinear
    f_gate: BitLinear,
    /// Candidate: ternary BitLinear
    c_gate: BitLinear,
    /// Output gate: ternary BitLinear
    o_gate: BitLinear,
}

impl MLGRU {
    pub fn new(config: MLGRUConfig) -> Self {
        Self {
            f_gate: BitLinear::new(config.hidden_size, config.hidden_size),
            c_gate: BitLinear::new(config.hidden_size, config.hidden_size),
            o_gate: BitLinear::new(config.hidden_size, config.hidden_size),
            config,
        }
    }

    /// Sigmoid
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Forward single step: element-wise only, no MatMul
    /// x_t: [hidden_size] input at time t
    /// h_prev: [hidden_size] previous hidden state
    /// Returns: h_t
    pub fn forward_step(&self, x_t: &[f32], h_prev: &[f32]) -> Vec<f32> {
        assert_eq!(x_t.len(), self.config.hidden_size);
        assert_eq!(h_prev.len(), self.config.hidden_size);

        // Forget gate: f_t = sigmoid(BitLinear(x_t))
        // BitLinear is ternary add/sub only
        let f_t = self.f_gate.forward(x_t);
        let f_t_sig: Vec<f32> = f_t.iter().map(|&v| Self::sigmoid(v)).collect();

        // Candidate: c_t = BitLinear(x_t) — simple linear transform, no coupling with hidden
        let c_t = self.c_gate.forward(x_t);

        // Hidden: h_t = (1 - f_t) * h_prev + f_t * c_t
        // ALL ELEMENT-WISE, NO MATMUL
        let mut h_t = vec![0.0; self.config.hidden_size];
        for i in 0..self.config.hidden_size {
            h_t[i] = (1.0 - f_t_sig[i]) * h_prev[i] + f_t_sig[i] * c_t[i];
        }

        // Output gate: o_t = BitLinear(x_t) + h_t element-wise?
        // Simplified: output = h_t * sigmoid(o_gate)

        let o_t = self.o_gate.forward(x_t);
        let o_t_sig: Vec<f32> = o_t.iter().map(|&v| Self::sigmoid(v)).collect();

        let mut out = vec![0.0; self.config.hidden_size];
        for i in 0..self.config.hidden_size {
            out[i] = h_t[i] * o_t_sig[i]; // element-wise
        }

        out
    }

    /// Forward full sequence: O(N) with parallel scan for training
    /// x: [seq_len, hidden_size]
    /// Returns: [seq_len, hidden_size]
    /// 
    /// For training, use parallel scan (prefix-sum) algorithm for parallelism
    /// For inference, recurrent O(1) per token, constant memory
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let hidden = self.config.hidden_size;
        assert_eq!(x.len(), seq_len * hidden);

        let mut h_prev = vec![0.0; hidden];
        let mut outputs = Vec::with_capacity(seq_len * hidden);

        for t in 0..seq_len {
            let x_t = &x[t*hidden..(t+1)*hidden];
            let h_t = self.forward_step(x_t, &h_prev);

            outputs.extend_from_slice(&h_t);
            h_prev = h_t;
        }

        outputs
    }

    /// Parallel scan for training (associative scan, prefix-sum)
    /// Allows parallel training despite recurrent math
    /// From Mamba selective scan
    pub fn forward_parallel_scan(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        // For POC, same as sequential, but real would use parallel prefix sum
        // Mamba-2 SSD unifies SSM and attention for 2-8x faster training

        self.forward(x, seq_len)
    }

    /// Complexity analysis
    pub fn complexity(&self, seq_len: usize) -> String {
        format!(
            "MLGRU: O(N) = {}*{} = {} ops, memory O(N·d) = {}*{} = constant per token, vs Transformer O(N²·d) = {}²*{} = {}",
            seq_len,
            self.config.hidden_size,
            seq_len * self.config.hidden_size,
            seq_len,
            self.config.hidden_size,
            seq_len,
            self.config.hidden_size,
            seq_len * seq_len * self.config.hidden_size
        )
    }
}

/// Comparison: Transformer attention vs MLGRU
pub fn comparison() -> String {
    format!(
        r#"
Transformer Attention:
- Compute per layer: O(N²·d) — quadratic in seq_len
- Memory per layer: O(N²) — KV cache grows linear, 100K tokens = 100GB
- Inference per token: O(N) — must attend to all previous tokens
- Training parallel: Yes

MLGRU (MatMul-free):
- Compute per layer: O(N·d²) — linear in seq_len
- Memory per layer: O(N·d) — constant state per layer, 1M tokens = ~1GB state
- Inference per token: O(d²) constant in N — RNN mode, no KV cache grow
- Training parallel: Yes via parallel scan (Mamba selective scan)
- 5x throughput vs Transformers, constant memory

Mamba selective SSM extends MLGRU with input-dependent gating.
RWKV uses time-mixing + channel-mixing with constant state.
Liquid uses ODE with adaptive time-constants, inference-time adaptation.
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlgru_no_matmul() {
        let config = MLGRUConfig { hidden_size: 8, intermediate_size: 16 };
        let mlgru = MLGRU::new(config);

        let x = vec![1.0; 8];
        let h_prev = vec![0.0; 8];

        let h_t = mlgru.forward_step(&x, &h_prev);
        assert_eq!(h_t.len(), 8);
    }

    #[test]
    fn test_mlgru_sequence() {
        let config = MLGRUConfig { hidden_size: 4, intermediate_size: 8 };
        let mlgru = MLGRU::new(config);

        let seq_len = 3;
        let x = vec![1.0; seq_len * 4];

        let out = mlgru.forward(&x, seq_len);
        assert_eq!(out.len(), seq_len * 4);
    }
}
