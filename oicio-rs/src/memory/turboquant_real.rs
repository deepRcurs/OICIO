/*!
TurboQuant Real — Walsh-Hadamard Rotation O(n log n) — Real Implementation
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Real TurboQuant uses Walsh-Hadamard Transform (WHT) for rotation, not matrix multiplication.
WHT is O(n log n) with only add/sub, no multiplication, no weights.

This is what makes it data-oblivious and fast: no training, no codebook calibration.
*/

use super::super::core::hadamard::hadamard_transform;

/// Real TurboQuant with FWHT rotation O(n log n)
pub struct TurboQuantReal {
    dim: usize,
    bit_width: usize,
    codebook: Vec<f32>,
}

impl TurboQuantReal {
    pub fn new(dim: usize, bit_width: usize) -> Self {
        assert!(dim.is_power_of_two(), "Dim must be power of 2 for FWHT, got {}", dim);

        let codebook = match bit_width {
            2 => vec![-1.510, -0.4528, 0.4528, 1.510],
            4 => (0..16).map(|i| -2.0 + i as f32 * 4.0 / 15.0).collect(),
            _ => (0..8).map(|i| -2.0 + i as f32 * 4.0 / 7.0).collect(),
        };

        Self { dim, bit_width, codebook }
    }

    /// Compress with REAL FWHT rotation O(n log n), not matrix mul
    /// vectors: [N, D] flattened
    pub fn compress_real(&self, vectors: &[f32], num_vectors: usize) -> (Vec<u8>, Vec<f32>) {
        let dim = self.dim;
        assert_eq!(vectors.len(), num_vectors * dim);

        let mut norms = Vec::with_capacity(num_vectors);
        let mut rotated = vec![0.0; num_vectors * dim];

        // 1. Norms + normalize
        for n in 0..num_vectors {
            let mut norm_sq = 0.0;
            for d in 0..dim {
                norm_sq += vectors[n*dim + d] * vectors[n*dim + d];
            }
            let norm = norm_sq.sqrt().max(1e-8);
            norms.push(norm);

            for d in 0..dim {
                rotated[n*dim + d] = vectors[n*dim + d] / norm;
            }
        }

        // 2. REAL FWHT rotation O(n log n) — only add/sub, no mul, no weights
        // This is the key difference from POC which used matrix multiplication
        // Real TurboQuant: apply Walsh-Hadamard transform to make coordinates Gaussian
        for n in 0..num_vectors {
            let slice = &mut rotated[n*dim..(n+1)*dim];
            hadamard_transform(slice);
        }

        // 3. Lloyd-Max quant
        let mut indices = vec![0u8; num_vectors * dim];
        for n in 0..num_vectors {
            for d in 0..dim {
                let val = rotated[n*dim + d];
                let mut best_idx = 0;
                let mut best_dist = f32::INFINITY;
                for (i, &cb) in self.codebook.iter().enumerate() {
                    let dist = (val - cb).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i;
                    }
                }
                indices[n*dim + d] = best_idx as u8;
            }
        }

        (indices, norms)
    }

    /// Decompress with inverse FWHT (same as forward, orthogonal)
    pub fn decompress_real(&self, codes: &[u8], norms: &[f32]) -> Vec<f32> {
        let dim = self.dim;
        let num_vectors = norms.len();
        let mut dequant = vec![0.0; num_vectors * dim];

        for n in 0..num_vectors {
            for d in 0..dim {
                let idx = codes[n*dim + d] as usize;
                dequant[n*dim + d] = self.codebook[idx];
            }
        }

        // Inverse FWHT (same as forward)
        for n in 0..num_vectors {
            let slice = &mut dequant[n*dim..(n+1)*dim];
            hadamard_transform(slice);
        }

        // Restore norm
        let mut recon = vec![0.0; num_vectors * dim];
        for n in 0..num_vectors {
            for d in 0..dim {
                recon[n*dim + d] = dequant[n*dim + d] * norms[n];
            }
        }

        recon
    }

    pub fn stats(&self, num_vectors: usize) -> String {
        let fp32_bytes = num_vectors * self.dim * 4;
        let packed_bytes = num_vectors * self.dim * self.bit_width / 8 + num_vectors * 4;
        format!(
            "TurboQuant Real FWHT O(n log n): {} vectors {} dim: {:.1}MB -> {:.1}MB ({:.1}x) @ {}-bit, no mul only add/sub",
            num_vectors,
            self.dim,
            fp32_bytes as f32 / 1024.0 / 1024.0,
            packed_bytes as f32 / 1024.0 / 1024.0,
            fp32_bytes as f32 / packed_bytes as f32,
            self.bit_width
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turboquant_real_fwht() {
        let tq = TurboQuantReal::new(8, 4);
        let vectors = vec![1.0; 2*8];
        let (codes, norms) = tq.compress_real(&vectors, 2);
        assert_eq!(codes.len(), 16);
        assert_eq!(norms.len(), 2);

        let recon = tq.decompress_real(&codes, &norms);
        assert_eq!(recon.len(), 16);
    }
}
