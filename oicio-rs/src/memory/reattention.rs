/*!
ReAttention — Finite Scope Infinite Context — Rust CPU-Only
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

3 syarat infinite context:
1. PE not OOD
2. Stable entropy
3. Effective awareness

Core: position-agnostic top-k BEFORE position-aware attention
*/

pub struct ReAttentionConfig {
    pub global_tokens: usize,
    pub local_tokens: usize,
    pub select_span: usize,
    pub top_k_prime: usize,
}

pub struct ReAttention {
    config: ReAttentionConfig,
    max_scope: usize,
}

impl ReAttention {
    pub fn new(config: ReAttentionConfig) -> Self {
        let max_scope = config.global_tokens + config.local_tokens + config.top_k_prime * config.select_span;
        println!("[ReAttention] Max scope: {} (global {} + local {} + {}*{})", max_scope, config.global_tokens, config.local_tokens, config.top_k_prime, config.select_span);
        Self { config, max_scope }
    }

    /// Split cache into global, middle, local
    pub fn split_cache<'a>(&self, kv_cache: &'a [f32], dim: usize) -> (&'a [f32], &'a [f32], &'a [f32]) {
        let seq_len = kv_cache.len() / dim;
        if seq_len <= self.config.global_tokens + self.config.local_tokens {
            return (&kv_cache[..0], kv_cache, &kv_cache[..0]);
        }

        let global_end = self.config.global_tokens * dim;
        let local_start = (seq_len - self.config.local_tokens) * dim;

        let global = &kv_cache[..global_end];
        let middle = &kv_cache[global_end..local_start];
        let local = &kv_cache[local_start..];

        (global, middle, local)
    }

    /// Position-agnostic selection: q_t * K_middle^T without RoPE
    pub fn position_agnostic_selection(&self, query: &[f32], middle_k: &[f32], dim: usize) -> Vec<usize> {
        let middle_len = middle_k.len() / dim;
        let mut scores = vec![0.0; middle_len];

        for i in 0..middle_len {
            let mut dot = 0.0;
            for d in 0..dim {
                dot += query[d] * middle_k[i*dim + d];
            }
            scores[i] = dot;
        }

        // Top-k' spans
        let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i,s)| (i,*s)).collect();
        indexed.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

        let mut selected = std::collections::HashSet::new();
        for (idx, _) in indexed.iter().take(self.config.top_k_prime*2) {
            let start = idx.saturating_sub(self.config.select_span/2);
            let end = (start + self.config.select_span).min(middle_len);
            for j in start..end {
                selected.insert(j);
                if selected.len() >= self.config.top_k_prime * self.config.select_span {
                    break;
                }
            }
            if selected.len() >= self.config.top_k_prime * self.config.select_span {
                break;
            }
        }

        let mut selected_vec: Vec<usize> = selected.into_iter().collect();
        selected_vec.sort();
        selected_vec.truncate(self.config.top_k_prime * self.config.select_span);
        selected_vec
    }

    pub fn max_scope(&self) -> usize {
        self.max_scope
    }
}
