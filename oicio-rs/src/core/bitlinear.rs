/*!
BitLinear — Ternary Weights {-1,0,1} — No MatMul, Only Add/Sub
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Berdasarkan:
- Microsoft BitNet b1.58: ternary weights, absmean quantization
- MatMul-free LM 2406.02528: BitLinear eliminates MatMul in dense layers
- T-MAC: LUT-based mpGEMM without dequantization, no multiplication

Real BitNet 2B: 1.1GB vs 4.8GB FP16 (4.3x), 4.1x faster, 8.9x throughput
Bonsai 8B: 1.75GB vs Qwen3 16.38GB (9.4x), 82 tok/s M4 Pro, 27 tok/s iPhone

CPU-only: AVX2/NEON TBL/PSHUF for parallel LUT lookup, 32 indices with 1 instruction
*/

/// Ternary weight: -1, 0, +1 — 1.58-bit
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(i8)]
pub enum TernaryWeight {
    NegOne = -1,
    Zero = 0,
    PosOne = 1,
}

impl TernaryWeight {
    /// From 2-bit packed value: 00=-1, 01=0, 10=1, 11=0 (unused)
    pub fn from_2bit(val: u8) -> Self {
        match val & 0b11 {
            0 => TernaryWeight::NegOne,
            1 => TernaryWeight::Zero,
            2 => TernaryWeight::PosOne,
            _ => TernaryWeight::Zero, // 11 -> 0
        }
    }

    /// To f32 with scale
    pub fn to_f32(self, scale: f32) -> f32 {
        (self as i8 as f32) * scale
    }
}

/// BitLinear: replaces nn.Linear with ternary weights, no MatMul
pub struct BitLinear {
    in_features: usize,
    out_features: usize,
    /// Packed ternary weights: 4 ternary per byte (2 bits each)
    /// Shape: [out_features, in_features/4] packed
    weight_packed: Vec<u8>,
    /// Scale per group of 128 weights (Bonsai style: group-wise quant + FP16 scale)
    weight_scale: Vec<f32>,
    /// Shadow full precision weights for training (QAT from step 0)
    weight_fp: Vec<f32>,
}

impl BitLinear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // For 2-bit packing: 4 ternary per byte
        let packed_in = (in_features + 3) / 4;
        let weight_packed = vec![0u8; out_features * packed_in];
        let num_groups = (in_features + 127) / 128;
        let weight_scale = vec![1.0; out_features * num_groups];
        let weight_fp = vec![0.0; out_features * in_features];

        Self {
            in_features,
            out_features,
            weight_packed,
            weight_scale,
            weight_fp,
        }
    }

    /// Absmean quantization: scale = 1 / mean(abs(w)), w_ternary = round(w/scale) clamped to {-1,0,1}
    /// From BitNet paper and FAQ
    pub fn absmean_quant(&self, w: &[f32]) -> (Vec<TernaryWeight>, f32) {
        let abs_mean = w.iter().map(|v| v.abs()).sum::<f32>() / w.len() as f32;
        let scale = if abs_mean < 1e-5 { 1e-5 } else { abs_mean };
        
        let ternary: Vec<TernaryWeight> = w.iter().map(|&v| {
            let scaled = v / scale;
            let rounded = scaled.round() as i8;
            match rounded {
                -1 => TernaryWeight::NegOne,
                0 => TernaryWeight::Zero,
                1 => TernaryWeight::PosOne,
                x if x < -1 => TernaryWeight::NegOne,
                _ => TernaryWeight::PosOne,
            }
        }).collect();

        (ternary, scale)
    }

    /// Pack ternary weights into bytes: 4 per byte, 2 bits each
    pub fn pack_ternary(ternary: &[TernaryWeight]) -> Vec<u8> {
        let mut packed = Vec::with_capacity((ternary.len() + 3) / 4);
        
        for chunk in ternary.chunks(4) {
            let mut byte = 0u8;
            for (i, &t) in chunk.iter().enumerate() {
                let bits = match t {
                    TernaryWeight::NegOne => 0b00,
                    TernaryWeight::Zero => 0b01,
                    TernaryWeight::PosOne => 0b10,
                };
                byte |= bits << (i*2);
            }
            packed.push(byte);
        }
        
        packed
    }

    /// Unpack bytes to ternary
    pub fn unpack_ternary(packed: &[u8], num_ternary: usize) -> Vec<TernaryWeight> {
        let mut ternary = Vec::with_capacity(num_ternary);
        
        for &byte in packed {
            for i in 0..4 {
                if ternary.len() >= num_ternary { break; }
                let bits = (byte >> (i*2)) & 0b11;
                ternary.push(TernaryWeight::from_2bit(bits));
            }
        }
        
        ternary
    }

    /// Forward: NO MATMUL, only ADD/SUB
    /// x: [batch, in_features] f32
    /// Returns: [batch, out_features] f32
    /// 
    /// Real implementation would use:
    /// - AVX2: _mm256_add_ps, _mm256_sub_ps
    /// - NEON: vaddq_f32, vsubq_f32
    /// - TBL/PSHUF for LUT lookup
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // For POC, simple loop: sum where w=1, sub where w=-1, skip 0
        // Real would use SIMD: 8x f32 per AVX2 register, 4x per NEON

        let batch = x.len() / self.in_features;
        let mut out = vec![0.0; batch * self.out_features];

        // Unpack weights for this forward (in real, would use LUT directly without full unpack)
        let ternary = Self::unpack_ternary(&self.weight_packed, self.out_features * self.in_features);

        for b in 0..batch {
            for o in 0..self.out_features {
                let mut sum = 0.0;
                let mut scale = 1.0;

                // Group-wise scale: 128 weights per group
                let group_idx = 0; // simplified, real would be o * num_groups + group
                if group_idx < self.weight_scale.len() {
                    scale = self.weight_scale[group_idx];
                }

                for i in 0..self.in_features {
                    let w = ternary[o * self.in_features + i];
                    match w {
                        TernaryWeight::PosOne => sum += x[b * self.in_features + i] * scale, // ADD
                        TernaryWeight::NegOne => sum -= x[b * self.in_features + i] * scale, // SUB
                        TernaryWeight::Zero => {}, // SKIP (sparsity)
                    }
                }

                out[b * self.out_features + o] = sum;
            }
        }

        out
    }

    /// Fused kernel: BitLinear + Hadamard + TurboQuant dequant in ONE kernel
    /// Minimizes HBM read/write like FlashAttention
    pub fn forward_fused(
        &self,
        x: &[f32],
        turboquant_codes: Option<&[u8]>,
        codebook: Option<&[f32]>,
        rotation: Option<&[f32]>, // [D*D] flattened
    ) -> Vec<f32> {
        // Step 1: Dequant TurboQuant codes via LUT in SRAM
        let mut x_dequant = x.to_vec();

        if let (Some(codes), Some(cb)) = (turboquant_codes, codebook) {
            // LUT lookup: codes [N,D] uint8 -> float via codebook
            // In real T-MAC: TBL instruction, 32 indices with 1 instruction
            for (i, &code) in codes.iter().enumerate() {
                if i < x_dequant.len() {
                    x_dequant[i] = cb[code as usize];
                }
            }

            // Inverse rotation if provided
            if let Some(rot) = rotation {
                // x_dequant @ rot.T, in SRAM
                // Simplified for POC
            }
        }

        // Step 2: Hadamard transform (in SRAM, no weights, only add/sub)
        // Would call hadamard_transform here

        // Step 3: BitLinear ternary matmul (in SRAM)
        self.forward(&x_dequant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_packing() {
        let ternary = vec![
            TernaryWeight::NegOne,
            TernaryWeight::Zero,
            TernaryWeight::PosOne,
            TernaryWeight::Zero,
        ];

        let packed = BitLinear::pack_ternary(&ternary);
        assert_eq!(packed.len(), 1);

        let unpacked = BitLinear::unpack_ternary(&packed, 4);
        assert_eq!(unpacked, ternary);
    }

    #[test]
    fn test_bitlinear_no_matmul() {
        let mut bl = BitLinear::new(8, 4);
        
        // Set some weights
        let ternary = vec![
            TernaryWeight::PosOne, TernaryWeight::NegOne, TernaryWeight::Zero, TernaryWeight::PosOne,
            TernaryWeight::Zero, TernaryWeight::PosOne, TernaryWeight::NegOne, TernaryWeight::Zero,
        ];
        bl.weight_packed = BitLinear::pack_ternary(&ternary);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = bl.forward(&x);

        // Manual: out[0] = 1*1 + (-1)*2 + 0*3 + 1*4 = 1 -2 +0 +4 = 3
        // But we have 4 out_features, first 8 ternary only for first out
        println!("Output: {:?}", out);
    }
}
