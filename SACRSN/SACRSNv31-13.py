# ============================================================
# SACRSN v31-11 (COMPLETE & FIXED)
# ============================================================

import os
import sys
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
from pathlib import Path
from collections import defaultdict, OrderedDict
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Tuple, Optional, List, Dict, Union, NamedTuple

# Optional: Attempt to import requests for auto-downloading data
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ==========================================
# 0. Strict Determinism
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
        "entropy": 0.05,
    },
    "epochs": 20,
    "batch_size": 128,
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
        self.vocab = {i: bytes([i]) for i in range(256)}  # initial byte vocab
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
        ids = list(text.encode("utf-8"))
        # Optimization: limit training data for speed if text is massive
        if len(ids) > 500000:
            ids = ids[:500000]

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
                print(f"BPE Merge {i}/{num_merges}...", end='\r')

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
        byte_seq = b"".join(self.vocab.get(i, b'?') for i in ids)
        return byte_seq.decode("utf-8", errors="replace")

# -------------------------
# Loader Logic
# -------------------------
def get_shakespeare_loader(file_path: str, seq_len=CONFIG["seq_len"], batch_size=CONFIG["batch_size"]):
    path = Path(file_path)
    if not path.exists():
        print(f"File {file_path} not found.")
        if HAS_REQUESTS:
            print("Downloading TinyShakespeare...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            r = requests.get(url)
            with open(path, "w", encoding="utf-8") as f:
                f.write(r.text)
        else:
            raise FileNotFoundError(f"Please download {file_path} manually.")

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Train tokenizer
    tokenizer = SimpleBPE(vocab_size=CONFIG["n_syntax_symbols"] + CONFIG["n_semantic_symbols"] + 200)
    tokenizer.train(raw_text)

    # Encode full text
    data_ids = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)

    class ShakespeareDataset(Dataset):
        def __init__(self, data_tensor, seq_len):
            self.data = data_tensor
            self.seq_len = seq_len

        def __len__(self):
            return len(self.data) - self.seq_len

        def __getitem__(self, idx):
            x = self.data[idx : idx + self.seq_len].clone()
            y = self.data[idx + 1 : idx + self.seq_len + 1].clone()
            return x, y

    dataset = ShakespeareDataset(data_ids, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return loader, tokenizer.vocab_size, base_tokenizer

# Initialize global tokenizer placeholder
base_tokenizer = None

# ==========================================
# 3. Complex Primitives
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
# 4. Phase-Coupled Attention
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

        # Complex Dot Product Attention
        # (Batch, Length, Head, Dim)
        # Score = Q * K_conj
        attn_scores = torch.einsum("blhd,bshd->bhls", q, k.conj())
        attn_scores = attn_scores.real / math.sqrt(self.head_dim)

        # Causal Masking
        mask = torch.triu(torch.ones(L, L, device=z.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        # Re-complexify for V mult
        attn_probs = torch.complex(attn_probs, torch.zeros_like(attn_probs))

        # Output
        attn_output = torch.einsum("bhls,bshd->blhd", attn_probs, v)
        attn_output = attn_output.contiguous().view(B, L, D)

        return self.out_proj(attn_output)

# ==========================================
# 5. Differentiable Soft-Stack (Batch-Aware)
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, stack_size: int, dim: int):
        super().__init__()
        self.stack_size = stack_size
        self.dim = dim
        self.register_buffer("stack_buffer", torch.zeros(1, stack_size, dim))

    def forward(self, x: torch.Tensor, push_gate: torch.Tensor, pop_gate: torch.Tensor):
        """
        x: (B, L, D) - values to push
        push_gate: (B, L, 1) [0,1]
        pop_gate:  (B, L, 1) [0,1]
        """
        B, L, D = x.shape
        if self.stack_buffer.size(0) != B:
            self.stack_buffer = torch.zeros(B, self.stack_size, D, device=x.device)

        stack = self.stack_buffer.clone()
        outputs = []

        # Sequential processing over L
        for t in range(L):
            val = x[:, t, :]       # (B, D)
            p = push_gate[:, t, :] # (B, 1)
            o = pop_gate[:, t, :]  # (B, 1)

            # Soft PUSH:
            # Shift everything down (index 1->2, 0->1), new value at 0 (or stack top)
            # Here, let's treat index 0 as BOTTOM, index -1 as TOP.
            # Shift Left (discard bottom)
            shifted_stack = torch.cat([stack[:, 1:, :], val.unsqueeze(1)], dim=1)
            # Interpolate
            stack = (1 - p.unsqueeze(1)) * stack + p.unsqueeze(1) * shifted_stack

            # Soft POP:
            # We decay the whole stack or shift right? 
            # Simple soft stack usually just suppresses the magnitude or interpolates with a "popped" version (shift right).
            # Let's implement pop as "decay/erase top" for stability in this version.
            stack = stack * (1 - o.unsqueeze(1))

            outputs.append(stack[:, -1, :]) # Top of stack

        # Detach state to prevent infinite backprop across batches in this stateless training loop
        self.stack_buffer = stack.detach()
        
        return torch.stack(outputs, dim=1) # (B, L, D)

# ==========================================
# 6. Context & VQ Layers
# ==========================================

class EnhancedContextGate(nn.Module):
    def __init__(self, x_dim, context_dim):
        super().__init__()
        self.linear = nn.Linear(x_dim * 2 + context_dim, x_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, context):
        # x is complex (B, L, D) -> convert to (B, L, 2D)
        if torch.is_complex(x):
            x_real = torch.cat([x.real, x.imag], dim=-1)
        else:
            x_real = torch.cat([x, x], dim=-1) # Fallback

        if context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)

        combined = torch.cat([x_real, context], dim=-1)
        gate = self.sigmoid(self.linear(combined))
        
        # Apply gate to complex input (magnitude scaling)
        return x * torch.complex(gate, torch.zeros_like(gate))


class EnhancedContextAwareDualVQ(nn.Module):
    def __init__(self, n_embeddings, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(n_embeddings, dim))

    def forward(self, z):
        # z is complex (B, L, D). We quantize the REAL part for codebook.
        B, L, D = z.shape
        flat_input = z.real.reshape(-1, D)
        
        # Distance
        dist = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding.t())
            + torch.sum(self.embedding ** 2, dim=1)
        )
        
        indices = torch.argmin(dist, dim=1)
        quantized_real = self.embedding[indices].view(B, L, D)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized_real.detach(), z.real)
        q_latent_loss = F.mse_loss(quantized_real, z.real.detach())
        loss = q_latent_loss + CONFIG["commitment_cost"] * e_latent_loss
        
        # Straight-through
        quantized_real = z.real + (quantized_real - z.real).detach()
        
        # Recombine with imaginary part
        z_out = torch.complex(quantized_real, z.imag)
        
        return z_out, loss

class StepOutput(NamedTuple):
    z: torch.Tensor
    loss: torch.Tensor
    halt: Optional[torch.Tensor]

# ==========================================
# 7. Main Model
# ==========================================
class RecurrentNeuroSymbolicCell(nn.Module):
    def __init__(self, dim, n_heads, stack_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.attention = PhaseCoupledComplexAttention(dim, n_heads)
        self.stack = DifferentiableStack(stack_size, dim)
        self.vq = EnhancedContextAwareDualVQ(CONFIG["n_semantic_symbols"], dim)
        self.context_gate = EnhancedContextGate(dim, dim)
        self.norm = ComplexLayerNorm(dim)
        self.modrelu = ModReLU(dim)
        
        # Gate Predictor for Stack (FIXED: Added this)
        self.stack_gate_proj = nn.Linear(dim, 2)
        
        self.lm_head = nn.Linear(dim, vocab_size)
        
    def forward(self, z: torch.Tensor, context: torch.Tensor):
        B = z.size(0)
        
        # 1. Embedding
        if z.dim() == 2:
            z = self.embedding(z.long())
        
        # Make complex
        z = torch.complex(z, torch.zeros_like(z))
        
        # 2. Attention
        z_attn = self.attention(z)

        # 3. VQ
        z_quant, vq_loss = self.vq(z_attn)

        # 4. Context Gating
        z_gated = self.context_gate(z_quant, context)

        # 5. Stack Operations (FIXED: Dynamic Gates)
        # Use real part of signal to decide push/pop
        gate_logits = self.stack_gate_proj(z_gated.real)
        gates = torch.sigmoid(gate_logits)
        push_gate = gates[..., 0:1] # (B, L, 1)
        pop_gate = gates[..., 1:2]  # (B, L, 1)

        stack_top = self.stack(z_gated.real, push_gate, pop_gate)
        # Convert stack top to complex
        stack_top_c = torch.complex(stack_top, torch.zeros_like(stack_top))

        # 6. Fusion & Output
        z_out = self.modrelu(self.norm(z_gated + stack_top_c))
        
        return StepOutput(z=z_out, loss=vq_loss, halt=None)

# ==========================================
# 8. Training Utilities
# ==========================================

def generate_sample_text(model, tokenizer, start_prompt="To be, or not to be, ", length=100):
    model.eval()
    ids = tokenizer.encode(start_prompt)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    context = torch.zeros(1, CONFIG["embedding_dim"], device=DEVICE)
    
    # Reset stack for generation
    model.stack.stack_buffer.zero_()

    generated = list(ids)
    
    with torch.no_grad():
        for _ in range(length):
            # Context window
            curr_input = input_tensor[:, -CONFIG["seq_len"]:]
            
            step_out = model(curr_input, context)
            
            # Predict next token from last embedding
            logits = model.lm_head(step_out.z[:, -1, :].real)
            probs = F.softmax(logits, dim=-1)
            
            # Sample
            next_token = torch.multinomial(probs, 1)
            
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            generated.append(next_token.item())
            
    return tokenizer.decode(generated)

# ==========================================
# 9. Main Execution
# ==========================================

def main():
    # 1. Load Data
    print("Initializing Data...")
    train_loader, vocab_size, tokenizer = get_shakespeare_loader("tinyshakespeare.txt")
    
    # 2. Build Model
    print(f"Initializing Model (Vocab: {vocab_size})...")
    model = RecurrentNeuroSymbolicCell(
        CONFIG["embedding_dim"], 
        CONFIG["n_heads"], 
        CONFIG["stack_size"], 
        vocab_size
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    # Handle AMP scaler (Robust check)
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["use_amp"])

    # 3. Training Loop
    print("Starting Training...")
    try:
        for epoch in range(CONFIG["epochs"]):
            model.train()
            epoch_loss = 0.0
            
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                context = torch.zeros(x.size(0), CONFIG["embedding_dim"], device=DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"]):
                    out = model(x, context)
                    
                    logits = model.lm_head(out.z.real)
                    
                    # Flatten for Cross Entropy
                    ce_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                    
                    # Entropy Penalty (from original requirements)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                    entropy_loss = -CONFIG["weights"]["entropy"] * entropy
                    
                    total_loss = ce_loss + CONFIG["weights"]["vq"] * out.loss + entropy_loss

                scaler.scale(total_loss).backward()
                
                # Clip Grads
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += total_loss.item()

                if i % 100 == 0:
                    print(f"Ep {epoch+1} | Batch {i} | Loss: {total_loss.item():.4f} | Ent: {entropy.item():.2f}")

            # End of Epoch Stats
            avg_loss = epoch_loss / len(train_loader)
            print(f"\n=== Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} ===")
            
            # Generation check
            sample = generate_sample_text(model, tokenizer)
            print(f"GENERATED:\n{sample}\n")
            print("-" * 50)
            
            # Save
            torch.save(model.state_dict(), f"sacrsn_epoch_{epoch+1}.pt")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        torch.save(model.state_dict(), "sacrsn_interrupted.pt")
        print("Saved.")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
