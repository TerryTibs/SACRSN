# ============================================================
# SACRSN v31-12 (Corrected & Optimized)
# ============================================================

import os
import sys
import time
import random
import requests
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import traceback
from pathlib import Path
from collections import defaultdict, OrderedDict
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Tuple, Optional, List, Dict, Union, NamedTuple

# ==========================================
# 0. Strict Determinism & Setup
# ==========================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "task_type": "shakespeare",
    "seq_len": 64,
    "embedding_dim": 128,
    "n_heads": 8,
    "n_syntax_symbols": 64,
    "n_semantic_symbols": 128,
    "commitment_cost": 0.25,
    "stack_size": 32,
    "weights": {
        "prediction": 1.0,
        "vq": 1.0,
        "entropy": 0.02,
    },
    "epochs": 5,          # Reduced for demo speed
    "batch_size": 64,     # Adjusted for stability
    "learning_rate": 3e-4,
    "grad_clip": 1.0,
    "use_amp": torch.cuda.is_available(), 
}

# ==========================================
# 2. Data & Ordered BPE Tokenizer
# ==========================================

class SimpleBPE:
    def __init__(self, vocab_size=1000):
        self.target_vocab_size = vocab_size
        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {"<PAD>": 256, "<UNK>": 257, "<MASK>": 258}
        self.vocab.update({v: k for k, v in self.special_tokens.items()})
        self.next_id = 259
        self.vocab_size = 259
        self._encode_cache = OrderedDict()
        self.CACHE_SIZE = 5000

    def get_stats(self, ids):
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def train(self, text):
        print("Training BPE Tokenizer...")
        # Use a subset for speed if text is huge
        if len(text) > 1_000_000:
            text = text[:1_000_000]
            
        ids = list(text.encode("utf-8"))
        num_merges = self.target_vocab_size - self.next_id

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = self.next_id
            self.merges.append((pair, idx))
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.next_id += 1

            # Apply merge
            new_ids = []
            skip = False
            for j in range(len(ids)):
                if skip:
                    skip = False
                    continue
                if j < len(ids) - 1 and ids[j] == pair[0] and ids[j + 1] == pair[1]:
                    new_ids.append(idx)
                    skip = True
                else:
                    new_ids.append(ids[j])
            ids = new_ids
            
            if i % 50 == 0:
                print(f"BPE Merge {i}/{num_merges} complete...", end="\r")
        
        self.vocab_size = self.next_id
        print(f"\nFinal Vocab Size: {self.vocab_size}")

    def encode(self, text: str):
        if text in self._encode_cache:
            return self._encode_cache[text]

        ids = list(text.encode("utf-8"))
        for (a, b), idx in self.merges:
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        if len(self._encode_cache) > self.CACHE_SIZE:
            self._encode_cache.popitem(last=False)
        self._encode_cache[text] = ids
        return ids

    def decode(self, ids: list):
        byte_seq = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return byte_seq.decode("utf-8", errors="replace")

def get_shakespeare_loader(file_path: str, seq_len=CONFIG["seq_len"], batch_size=CONFIG["batch_size"]):
    path = Path(file_path)
    if not path.exists():
        print(f"{file_path} not found. Downloading...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            r = requests.get(url)
            with open(path, "w", encoding="utf-8") as f:
                f.write(r.text)
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Increase vocab size slightly for BPE to be effective
    tokenizer = SimpleBPE(vocab_size=CONFIG["n_syntax_symbols"] + CONFIG["n_semantic_symbols"] + 500)
    tokenizer.train(raw_text)

    data_ids = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)

    class ShakespeareDataset(Dataset):
        def __init__(self, data_tensor, seq_len):
            self.data = data_tensor
            self.seq_len = seq_len

        def __len__(self):
            return len(self.data) - self.seq_len

        def __getitem__(self, idx):
            x = self.data[idx : idx + self.seq_len]
            y = self.data[idx + 1 : idx + self.seq_len + 1]
            return x, y

    dataset = ShakespeareDataset(data_ids, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return loader, loader, tokenizer.vocab_size, data_ids, tokenizer

# Initialize dataset
train_loader, val_loader, VOCAB_SIZE, data_tensor, base_tokenizer = get_shakespeare_loader("tinyshakespeare.txt")

# ==========================================
# 4. Complex Primitives
# ==========================================

class ComplexLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.norm_r = nn.LayerNorm(dim, eps=eps)
        self.norm_i = nn.LayerNorm(dim, eps=eps)

    def forward(self, z: torch.Tensor):
        if not torch.is_complex(z):
            z = torch.complex(z, torch.zeros_like(z))
        return torch.complex(self.norm_r(z.real), self.norm_i(z.imag))

class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor):
        if not torch.is_complex(z):
            z = torch.complex(z, torch.zeros_like(z))
        magnitude = torch.abs(z)
        scale = F.relu(magnitude + self.bias) / (magnitude + 1e-6)
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, z: torch.Tensor):
        if not torch.is_complex(z):
            z = torch.complex(z, torch.zeros_like(z))
        r = self.real(z.real) - self.imag(z.imag)
        i = self.real(z.imag) + self.imag(z.real)
        return torch.complex(r, i)

# ==========================================
# 5. Phase-Coupled Attention
# ==========================================
class PhaseCoupledComplexAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        
        self.q_proj = ComplexLinear(dim, dim)
        self.k_proj = ComplexLinear(dim, dim)
        self.v_proj = ComplexLinear(dim, dim)
        self.out_proj = ComplexLinear(dim, dim)

    def forward(self, z):
        B, L, D = z.shape
        q = self.q_proj(z).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(z).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(z).view(B, L, self.n_heads, self.head_dim)

        # Attention scores: (B, L, H, D) @ (B, S, H, D).conj() -> (B, L, H, S) -> permute -> (B, H, L, S)
        # Using einsum for clarity:
        # q: blhd, k: bshd -> bhls
        attn_scores = torch.einsum("blhd,bshd->bhls", q, k.conj())
        attn_scores = attn_scores.real / math.sqrt(self.head_dim)
        
        # Causal Mask
        mask = torch.triu(torch.ones(L, L, device=z.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.complex(attn_probs, torch.zeros_like(attn_probs))

        # Output: bhls * bshd -> blhd
        attn_output = torch.einsum("bhls,bshd->blhd", attn_probs, v)
        attn_output = attn_output.contiguous().view(B, L, D)
        
        return self.out_proj(attn_output)

# ==========================================
# 6. Differentiable Soft-Stack
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, stack_size: int, dim: int):
        super().__init__()
        self.stack_size = stack_size
        self.dim = dim
        # Initialize buffer
        self.register_buffer("stack_buffer", torch.zeros(1, stack_size, dim))

    def forward(self, x: torch.Tensor, push_gate: torch.Tensor, pop_gate: torch.Tensor):
        """
        Iterative soft stack processing. 
        Note: This is O(L) sequential.
        """
        B, L, D = x.shape
        if self.stack_buffer.size(0) != B:
            self.stack_buffer = torch.zeros(B, self.stack_size, D, device=x.device, dtype=x.dtype)
        
        stack = self.stack_buffer.clone() 
        stack_outputs = []

        # Process sequence
        for t in range(L):
            p = push_gate[:, t, :]  # (B, 1)
            o = pop_gate[:, t, :]   # (B, 1)
            val = x[:, t, :]        # (B, D)

            # Soft Push: slide down, put val at top (index 0 or -1 depending on convention)
            # Here convention: last element is top
            # Shift stack left (lose bottom)
            shifted_stack = torch.cat([stack[:, 1:, :], val.unsqueeze(1)], dim=1)
            
            # Interpolate based on push gate
            stack = (1 - p.unsqueeze(1)) * stack + p.unsqueeze(1) * shifted_stack

            # Soft Pop: Interpolate between current and popped state
            # Popped state is roughly shifting right (duplicating bottom or zero fill)
            # Simple approximation: stack * decay
            stack = stack * (1 - o.unsqueeze(1))

            stack_outputs.append(stack[:, -1, :]) # Record Top

        # Save state (detached) for next window if implementing stateful RNN
        # For this training loop (stateless windows), we don't strictly need to save, 
        # but we reset the buffer to detach gradients.
        self.stack_buffer = stack.detach()

        return torch.stack(stack_outputs, dim=1) # (B, L, D)

# ==========================================
# 7. Context Gate
# ==========================================
class EnhancedContextGate(nn.Module):
    def __init__(self, x_dim, context_dim):
        super().__init__()
        self.linear = nn.Linear(x_dim * 2 + context_dim, x_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, context):
        if torch.is_complex(x):
            x = torch.cat([x.real, x.imag], dim=-1)
        
        # Expand context to match seq len
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)

        combined = torch.cat([x, context], dim=-1)
        return self.sigmoid(self.linear(combined))

# ==========================================
# 8. VQ Layer
# ==========================================
class EnhancedContextAwareDualVQ(nn.Module):
    def __init__(self, n_embeddings, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(n_embeddings, dim))

    def forward(self, z):
        B, L, D = z.shape
        flat_enc = z.view(B * L, D)
        
        # Real-valued codebook lookup (using real part of complex signal magnitude or similar)
        # Here we treat complex z as 2*D real vector for distance or just project
        # For simplicity in this fix: use magnitude for matching, or project to real
        
        # Simplified: Use real part for codebook matching to keep dims consistent
        target = z.real 
        flat_target = target.view(B*L, D)
        
        dist = (
            torch.sum(flat_target ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_target, self.embedding.t())
            + torch.sum(self.embedding ** 2, dim=1)
        )
        
        encoding_indices = torch.argmin(dist, dim=1)
        z_quant_real = self.embedding[encoding_indices].view(B, L, D)
        
        # Quantized complex (keep imaginary part, quantize real part)
        z_quant = torch.complex(z_quant_real, z.imag)

        # Loss
        vq_loss = F.mse_loss(z_quant.real.detach(), z.real) + \
                  0.25 * F.mse_loss(z_quant.real, z.real.detach())

        # Straight-through
        z_quant = z + (z_quant - z).detach()
        return z_quant, vq_loss

class StepOutput(NamedTuple):
    z: torch.Tensor
    loss: torch.Tensor

# ==========================================
# 9. Main Cell
# ==========================================
class RecurrentNeuroSymbolicCell(nn.Module):
    def __init__(self, dim, n_heads, stack_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.attention = PhaseCoupledComplexAttention(dim, n_heads)
        self.stack = DifferentiableStack(stack_size, dim)
        self.vq = EnhancedContextAwareDualVQ(CONFIG["n_semantic_symbols"], dim)
        self.norm = ComplexLayerNorm(dim)
        self.modrelu = ModReLU(dim)
        
        # Learnable gates for stack (Push/Pop)
        self.gate_proj = nn.Linear(dim, 2) 
        
        self.lm_head = nn.Linear(dim, vocab_size)
        
    def forward(self, z: torch.Tensor, context: torch.Tensor):
        if z.dim() == 2:
            z = self.embedding(z.long())

        # Make complex
        z = torch.complex(z, torch.zeros_like(z))
        
        # 1. Attention
        z_attn = self.attention(z) # (B, L, D)

        # 2. VQ
        z_quant, vq_loss = self.vq(z_attn)

        # 3. Stack Operations
        # Predict gates from the quantized signal (real part)
        gate_logits = self.gate_proj(z_quant.real)
        gates = torch.sigmoid(gate_logits)
        push_gate = gates[..., 0:1] # (B, L, 1)
        pop_gate = gates[..., 1:2]  # (B, L, 1)

        # Get stack output (sequence)
        stack_out = self.stack(z_quant.real, push_gate, pop_gate) # (B, L, D) real
        stack_out_c = torch.complex(stack_out, torch.zeros_like(stack_out))

        # 4. Combine & Activate
        z_combined = z_quant + stack_out_c
        z_out = self.modrelu(self.norm(z_combined))

        return StepOutput(z=z_out, loss=vq_loss)

# ==========================================
# 10. Training Loop
# ==========================================
model = RecurrentNeuroSymbolicCell(
    CONFIG["embedding_dim"], 
    CONFIG["n_heads"], 
    CONFIG["stack_size"], 
    VOCAB_SIZE
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
scaler = torch.amp.GradScaler(enabled=CONFIG["use_amp"])

def generate_text(model, start_str="To be, ", length=100):
    model.eval()
    ids = base_tokenizer.encode(start_str)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    context = torch.zeros(1, CONFIG["embedding_dim"], device=DEVICE)
    
    # Reset stack for generation
    model.stack.stack_buffer.zero_()

    for _ in range(length):
        # Crop context to seq_len
        input_seq = x[:, -CONFIG["seq_len"]:]
        
        with torch.no_grad():
            out = model(input_seq, context)
            logits = model.lm_head(out.z[:, -1, :].real)
            probs = F.softmax(logits, dim=-1)
            
            # Sample
            next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1)

    return base_tokenizer.decode(x[0].tolist())

def train():
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    try:
        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss = 0
            
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                context = torch.zeros(x.size(0), CONFIG["embedding_dim"], device=DEVICE)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type="cuda", enabled=CONFIG["use_amp"]):
                    out = model(x, context)
                    logits = model.lm_head(out.z.real)
                    
                    # Reshape for Cross Entropy: (B*L, Vocab) vs (B*L)
                    loss_ce = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
                    loss = loss_ce + 0.1 * out.loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1} | Step {i} | Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            print(f"=== Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} ===")
            print(f"Generated: {generate_text(model)}")
            
            torch.save(model.state_dict(), "sacrsn_model.pt")

    except KeyboardInterrupt:
        print("Training interrupted.")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    train()
