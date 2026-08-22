/*!
EM-LLM Surprise-based Event Segmentation — Rust CPU-Only
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Berdasarkan ICLR 2025 EM-LLM: Bayesian surprise + graph refinement
*/

pub struct Event {
    pub start: usize,
    pub end: usize,
    pub representative_tokens: Vec<Vec<f32>>, // topk per event
}

pub struct SurpriseSegmenter {
    gamma: f32,
    min_block_size: usize,
    max_block_size: usize,
}

impl SurpriseSegmenter {
    pub fn new(gamma: f32, min_block_size: usize, max_block_size: usize) -> Self {
        Self { gamma, min_block_size, max_block_size }
    }

    /// Compute surprise as L2 distance to prev token (proxy for LLM loss)
    pub fn compute_surprise(&self, embeddings: &[f32], dim: usize) -> Vec<f32> {
        let seq_len = embeddings.len() / dim;
        let mut surprise = vec![0.0; seq_len];

        for i in 1..seq_len {
            let mut dist_sq = 0.0;
            for d in 0..dim {
                let diff = embeddings[i*dim + d] - embeddings[(i-1)*dim + d];
                dist_sq += diff*diff;
            }
            surprise[i] = dist_sq.sqrt();
        }

        surprise
    }

    pub fn initial_segmentation(&self, surprise: &[f32]) -> Vec<usize> {
        let mean = surprise.iter().sum::<f32>() / surprise.len() as f32;
        let var = surprise.iter().map(|s| (s-mean)*(s-mean)).sum::<f32>() / surprise.len() as f32;
        let std = var.sqrt();
        let threshold = mean + self.gamma * std;

        let mut boundaries = vec![0];
        let mut current_size = 0;

        for (i, &s) in surprise.iter().enumerate() {
            current_size += 1;

            if s > threshold && current_size >= self.min_block_size {
                boundaries.push(i);
                current_size = 0;
            } else if current_size >= self.max_block_size {
                boundaries.push(i);
                current_size = 0;
            }
        }

        boundaries.push(surprise.len());
        boundaries.sort();
        boundaries.dedup();
        boundaries
    }

    pub fn segment(&self, embeddings: &[f32], dim: usize) -> (Vec<usize>, Vec<f32>, Vec<Event>) {
        let surprise = self.compute_surprise(embeddings, dim);
        let boundaries = self.initial_segmentation(&surprise);

        let mut events = Vec::new();
        for i in 0..boundaries.len()-1 {
            let start = boundaries[i];
            let end = boundaries[i+1];
            events.push(Event {
                start,
                end,
                representative_tokens: Vec::new(), // would select topk by norm
            });
        }

        (boundaries, surprise, events)
    }
}
