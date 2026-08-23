/*!
OICIO Rust — MatMul-Free CPU-Only
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Paradigma baru total: Tanpa MatMul, Tanpa GPU, Tanpa Python/CUDA
Hanya Addition, Subtraction, Table Lookup, Hadamard Transform

Emergent space via:
- Ternary accumulation (associative memory)
- MLGRU state evolution (temporal compression)
- Hadamard thresholding (sparse coding)
- LUT associative (Hopfield-like)
- Liquid time-constants (inference-time adaptation)

Snapshot-safe: Rust code <1MB, toolchain in .cargo (excluded), model in .cache (excluded)
*/

pub mod core;
pub mod memory;
pub mod harness;
pub mod edge;
pub mod training;
pub mod phase5;

pub const VERSION: &str = "0.6.0";
pub const AUTHOR: &str = "Mzed Imamkh @mzedimamkh";
pub const LAB: &str = "deepRcurs Labs @deeprcurs";

pub fn version() -> String {
    format!("OICIO v{} — {} / {} — MatMul-Free CPU-Only", VERSION, LAB, AUTHOR)
}
