"""
OICIO Training: QAT Ternary Trainer
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Melatih TernarySAN dari scratch dengan Quantization Aware Training (QAT)
- Weights: ternary {-1,0,1} via absmean dari step 0 (bukan post-hoc)
- Activations: 8-bit (target a4.8)
- KV Cache: 2-bit Cactus Quants

Sebagai dataset dan trainer adalah kamu (LLM sumber pengetahuan)
- Generate synthetic data untuk long-context reasoning
- Guru yang membimbing via distillation dari frontier trajectories

POC: Train toy 0.5M param model di CPU 1.9GB RAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/user')
from oicio.core.ternary_san import TernarySAN

class SyntheticOOLONGDataset:
    """
    Generate synthetic OOLONG-like data sebagai guru
    OOLONG: semantic reasoning over thousands of entries, bukan needle retrieval
    """
    def __init__(self, num_samples=1000, seq_len=128, vocab_size=1000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # Generate synthetic data
        self.data = []
        for _ in range(num_samples):
            # Simulate document with entries: user_id, classification
            input_ids = np.random.randint(0, vocab_size, size=seq_len)
            # Label: count of entity entries (simulate)
            # For POC, label = number of tokens > vocab_size//2 (proxy for entity)
            label = np.sum(input_ids > vocab_size//2) % 10
            self.data.append((input_ids, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids, label = self.data[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class QATTrainer:
    def __init__(self, model: TernarySAN, dataset, lr=1e-4, device='cpu'):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        # For language modeling, we use next-token prediction
        # For POC, use classification over last token

    def train_step(self, input_ids, labels):
        self.model.train()
        self.optimizer.zero_grad()

        # Forward
        logits = self.model(input_ids)  # [B, S, V]
        # Take last token logits for classification (POC)
        last_logits = logits[:, -1, :]  # [B, V]
        # For classification, we need to map vocab to 10 classes (entity count)
        # Simulate: project vocab logits to 10 classes via mean pooling
        # For simplicity, use first 10 vocab as classes
        class_logits = last_logits[:, :10]  # [B, 10]

        loss = self.criterion(class_logits, labels)
        loss.backward()

        # Gradient clipping (important for ternary training)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def train(self, epochs=3, batch_size=8):
        print(f"[QAT Trainer] Training {len(self.dataset)} samples, {epochs} epochs, batch {batch_size}")
        print(f"[QAT Trainer] Device: {self.device}, Model: {self.model.count_ternary_params()}")

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            # Mini-batch loop
            for i in range(0, len(self.dataset), batch_size):
                batch_data = [self.dataset[j] for j in range(i, min(i+batch_size, len(self.dataset)))]
                input_ids = torch.stack([d[0] for d in batch_data]).to(self.device)
                labels = torch.stack([d[1] for d in batch_data]).to(self.device)

                loss = self.train_step(input_ids, labels)
                total_loss += loss
                num_batches += 1

                if num_batches % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} Batch {num_batches} Loss {loss:.4f}")

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

            # Check ternary stats
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if hasattr(module, 'weight') and 'BitLinear' in str(type(module)):
                        w = module.weight.data
                        w_ternary, scale = module.absmean_quant(w)
                        # Count distribution
                        unique, counts = torch.unique(w_ternary, return_counts=True)
                        dist = {int(u): int(c) for u, c in zip(unique, counts)}
                        print(f"    {name}: ternary dist {dist}, scale {scale.item():.4f}")
                        break  # just first

        print("[QAT Trainer] Training complete")

    def save_checkpoint(self, path="/home/user/oicio/data/ternary_san_qat.pt"):
        # Save in .cache? No, data is small (<128MB) so save in oicio/data (snapshot-safe)
        # But weights are tiny (0.25MB), so okay
        torch.save(self.model.state_dict(), path)
        print(f"[QAT Trainer] Saved checkpoint to {path}")

# Demo
if __name__ == "__main__":
    print("=== OICIO QAT Trainer POC ===")

    # Create model
    model = TernarySAN(vocab_size=1000, dim=128, num_layers=2, num_heads=4)
    print(f"Model stats: {model.count_ternary_params()}")

    # Create synthetic dataset (LLM as dataset generator)
    dataset = SyntheticOOLONGDataset(num_samples=200, seq_len=64, vocab_size=1000)
    print(f"Dataset: {len(dataset)} synthetic OOLONG samples")

    # Train
    trainer = QATTrainer(model, dataset, lr=1e-3, device='cpu')
    trainer.train(epochs=2, batch_size=8)
    trainer.save_checkpoint()

    print("\n[QAT] POC training done in limited env (1.9GB RAM, CPU)")
    print("[QAT] Real training would be 4T tokens, 2B-8B params, 10x less energy than FP16")
