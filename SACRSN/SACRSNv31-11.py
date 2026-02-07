# ============================================================
# SACRSN v31-11
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
import gc
import traceback
from pathlib import Path
from collections import defaultdict, OrderedDict
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple, Optional, List, Dict, Union, NamedTuple  # FIX: Added NamedTuple
from torch import amp
torch.autograd.set_detect_anomaly(True)

# Optional Dependency Guard
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: NetworkX not found. Topology plots will be skipped.")

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
    "ablation": {
        "use_stack": True,
        "use_vq": True,
        "use_graph": True,
        "use_complex": True,
    },
    "seq_len": 64,
    "embedding_dim": 128,
    "n_heads": 8,
    "n_syntax_symbols": 64,
    "n_semantic_symbols": 128,
    "commitment_cost": 0.25,
    "decay": 0.99,
    "context_gate_strength": 3.0,
    "max_recursion_depth": 8,
    "act_threshold": 0.99,
    "ponder_penalty": 0.01,
    "use_stack": True,
    "stack_size": 32,
    "graph_bias_scale": 1.0,
    "ortho_max_vocab": 1024,
    "logit_norm_scale": 10.0,
    "graph_warmup_steps": 500,
    "hyper": {
        "adjacency_decay": 0.999,
        "adjacency_clamp": 20.0,
        "stack_temp": 10.0,
        "var_clamp": 1e-5,
        "entropy_ema": 0.99,
        "memory_decay": 0.95,
    },
    "weights": {
        "prediction": 1.0,
        "ponder": 0.01,
        "vq": 1.0,
        "consistency": 0.05,
        "entropy": 0.02,
        "orthogonal": 0.001
    },
    "mask_prob": 0.15,
    "epochs": 15,
    "batch_size": 128,
    "learning_rate": 3e-4,
    "grad_clip": 1.0,
    "clip_floor": 0.05,
    "eps": 1e-5,
    "warmup_epochs": 3,
    "adaptive_scheduler": True,
    "structured_masking": True,
    "use_amp": False,  # FIX: set to False to avoid AMP issues
    "debug_anomaly": False,
}

if CONFIG["debug_anomaly"]:
    torch.autograd.set_detect_anomaly(True)

# ==========================================
# 2. Data & Ordered BPE Tokenizer (Tiny Shakespeare)
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
        self._decode_cache = OrderedDict()
        self._encode_cache = OrderedDict()
        self.CACHE_SIZE = 5000

    # -------------------------
    # Count adjacent pair frequencies
    # -------------------------
    def get_stats(self, ids):
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    # -------------------------
    # Train BPE merges
    # -------------------------
    def train(self, text):
        print("Training BPE Tokenizer (Ordered)...")
        ids = list(text.encode("utf-8"))
        num_merges = self.target_vocab_size - self.next_id

        for _ in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            # Most frequent pair
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

        self.vocab_size = self.next_id
        print(f"Final Vocab Size: {self.vocab_size}")

    # -------------------------
    # Encode text → token IDs
    # -------------------------
    def encode(self, text: str):
        if text in self._encode_cache:
            return self._encode_cache[text]

        ids = list(text.encode("utf-8"))

        # Apply BPE merges
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

        # Cache results
        if len(self._encode_cache) > self.CACHE_SIZE:
            self._encode_cache.popitem(last=False)
        self._encode_cache[text] = ids
        return ids

    # -------------------------
    # Decode token IDs → text
    # -------------------------
    def decode(self, ids: list):
        key = tuple(ids)
        if key in self._decode_cache:
            return self._decode_cache[key]

        byte_seq = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        decoded = byte_seq.decode("utf-8", errors="replace")

        if len(self._decode_cache) > self.CACHE_SIZE:
            self._decode_cache.popitem(last=False)
        self._decode_cache[key] = decoded
        return decoded


# -------------------------
# get_shakespeare_loader
# -------------------------
def get_shakespeare_loader(file_path: str, seq_len=CONFIG["seq_len"], batch_size=CONFIG["batch_size"]):
    # Load text
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found. Download tinyshakespeare.txt first.")
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Train tokenizer
    tokenizer = SimpleBPE(vocab_size=CONFIG["n_syntax_symbols"] + CONFIG["n_semantic_symbols"])
    tokenizer.train(raw_text)

    # Encode full text to IDs using BPE tokenizer
    data_ids = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)

    # Dataset & DataLoader
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, loader, tokenizer.vocab_size, data_ids, tokenizer


# -------------------------
# Initialize dataset
# -------------------------
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
        # If input is real, treat imaginary part as zeros
        if not torch.is_complex(z):
            z = torch.complex(z, torch.zeros_like(z))
        return torch.complex(self.norm_r(z.real), self.norm_i(z.imag))


class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor):
        # If input is real, treat it as complex with 0 imag
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
        # If input is real, treat as complex with 0 imaginary part
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

        # Complex linear projections
        self.q_proj = ComplexLinear(dim, dim)
        self.k_proj = ComplexLinear(dim, dim)
        self.v_proj = ComplexLinear(dim, dim)
        self.out_proj = ComplexLinear(dim, dim)

    def forward(self, z):
        """
        z: complex tensor (B, L, D)
        returns: complex tensor (B, L, D)
        """
        B, L, D = z.shape

        # Linear projections
        q = self.q_proj(z)  # complex
        k = self.k_proj(z)  # complex
        v = self.v_proj(z)  # complex

        # Reshape for multi-head attention
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)

        # Attention scores (complex dot product)
        attn_scores = torch.einsum("blhd,bshd->bhls", q, k.conj())  # B, H, L, S
        attn_scores = attn_scores.real / math.sqrt(self.head_dim)  # scale, take real for softmax

        # Softmax over sequence dimension
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # FIX: convert attn_probs to complex to match v
        attn_probs = torch.complex(attn_probs, torch.zeros_like(attn_probs))

        # Compute attention output
        attn_output = torch.einsum("bhls,bshd->blhd", attn_probs, v)

        # Merge heads
        attn_output = attn_output.contiguous().view(B, L, D)

        # Output projection
        out = self.out_proj(attn_output)

        return out

# ==========================================
# 6. Differentiable Soft-Stack (Batch-Aware)
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, stack_size: int, dim: int):
        super().__init__()
        self.stack_size = stack_size
        self.dim = dim
        # Initialize stack per batch dynamically in forward
        self.register_buffer("stack_buffer", torch.zeros(1, stack_size, dim))

    def forward(self, x: torch.Tensor, push_gate: torch.Tensor, pop_gate: torch.Tensor):
        """
        x: (B, L, D) tensor to push
        push_gate: (B, L, 1) gate
        pop_gate: (B, L, 1) gate
        returns:
            top: (B, D)
        """
        B, L, D = x.shape

        # Initialize per-batch stack if buffer is smaller
        if self.stack_buffer.size(0) != B:
            self.stack_buffer = torch.zeros(B, self.stack_size, D, device=x.device, dtype=x.dtype)

        stack = self.stack_buffer.clone()  # avoid in-place during computation

        for t in range(L):
            p_gate = push_gate[:, t, :]  # (B,1)
            o_gate = pop_gate[:, t, :]   # (B,1)
            push_vec = x[:, t, :]        # (B,D)

            # Soft pop: reduce top proportionally
            stack = stack * (1 - o_gate.unsqueeze(-1))  

            # Soft push: add vector scaled by push_gate
            stack = stack + push_vec.unsqueeze(1) * p_gate.unsqueeze(1)  

        # Take top of stack for each batch (last element)
        top = stack[:, -1, :]  # (B,D)

        # Save stack back
        self.stack_buffer = stack.detach()  # prevent gradient accumulation in buffer

        return top

# ==========================================
# 7. Context-Aware Gate
# ==========================================

class EnhancedContextGate(nn.Module):
    def __init__(self, x_dim, context_dim, out_dim=None):
        """
        x_dim: feature size of x (complex)
        context_dim: feature size of context (real)
        out_dim: output dimension (default = x_dim)
        """
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else x_dim
        # we don't hardcode input features; will compute dynamically
        self.linear = None
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, context):
        # convert complex x to real if needed
        if torch.is_complex(x):
            x = torch.view_as_real(x)  # (B,L,D,2)
            x = x.flatten(-2)          # (B,L,2*D)

        # broadcast context if needed
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)

        # ensure same dtype
        x = x.float()
        context = context.float()

        # concatenate
        combined = torch.cat([x, context], dim=-1)  # (B,L,2*D + D_context)

        # dynamically initialize linear if not already
        if self.linear is None:
            in_features = combined.shape[-1]
            self.linear = nn.Linear(in_features, self.out_dim).to(x.device)

        # linear + sigmoid
        out = self.linear(combined)
        out = self.sigmoid(out)
        return out


# ==========================================
# 8. Vector Quantization Layer
# ==========================================
class EnhancedContextAwareDualVQ(nn.Module):
    def __init__(self, n_embeddings, dim):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.dim = dim
        self.embedding_dim = dim  # backward compatibility
        # Initialize embeddings as real; will convert to complex if needed
        self.embedding = nn.Parameter(torch.randn(n_embeddings, dim))

    def forward(self, z):
        """
        z: (B, L, D) complex or real
        Returns:
            z_quant: quantized tensor, same shape as z
            vq_loss: commitment + codebook loss (real scalar)
        """
        B, L, D = z.shape
        flat_enc = z.view(B * L, D)

        # Convert embeddings to complex if z is complex
        embedding = torch.complex(self.embedding, torch.zeros_like(self.embedding)) if torch.is_complex(flat_enc) else self.embedding

        # Complex-safe squared distance
        dist = (
            torch.sum(torch.abs(flat_enc) ** 2, dim=1, keepdim=True)
            - 2 * torch.real(torch.matmul(flat_enc, embedding.t().conj()))
            + torch.sum(torch.abs(embedding) ** 2, dim=1)
        )

        # Nearest codebook
        encoding_indices = torch.argmin(dist, dim=1)
        z_quant = embedding[encoding_indices].view(B, L, D)

        # VQ loss — ensure real scalar for backward
        vq_loss = torch.mean(torch.abs(z_quant.detach() - z) ** 2) + torch.mean(torch.abs(z_quant - z.detach()) ** 2)

        # Straight-through estimator
        z_quant = z + (z_quant - z).detach()

        return z_quant, vq_loss

# ==========================================
# 9. StepOutput NamedTuple
# ==========================================
class StepOutput(NamedTuple):
    z: torch.Tensor
    memory: torch.Tensor
    ptr: torch.Tensor
    halt: torch.Tensor
    indices: tuple
    loss: torch.Tensor
    divergence: float

# ==========================================
# 10. Forward Pass Skeleton
# ==========================================
class RecurrentNeuroSymbolicCell(nn.Module):
    def __init__(self, dim, n_heads, stack_size, vocab_size=None):
        super().__init__()
        # Embedding for token IDs
        self.embedding = nn.Embedding(vocab_size, dim) if vocab_size else nn.Identity()
        self.attention = PhaseCoupledComplexAttention(dim, n_heads)
        self.stack = DifferentiableStack(stack_size, dim)
        self.context_gate = EnhancedContextGate(dim, dim)
        self.vq = EnhancedContextAwareDualVQ(CONFIG["n_semantic_symbols"], dim)
        self.norm = ComplexLayerNorm(dim)
        self.modrelu = ModReLU(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        
    def forward(self, z: torch.Tensor, context: torch.Tensor):
        B = z.size(0)
        L = z.size(1) if z.dim() >= 2 else 1

        # Embed token IDs if input is 2D (B, L)
        if z.dim() == 2:
            z = self.embedding(z.long())  # → (B, L, D)
            L = z.size(1)

        # Convert to complex dtype for ComplexLinear / attention
        z = torch.complex(z, torch.zeros_like(z))

        # Attention
        z_attn = self.attention(z)

        # Vector Quantization + Loss
        z_quant, vq_loss = self.vq(z_attn)

        # Context Gate
        z_gated = self.context_gate(z_quant, context)

        # FIXED: create push/pop gates with correct shape (B, L, 1)
        push_gate = torch.ones(B, L, 1, device=z_gated.device)
        pop_gate = torch.zeros(B, L, 1, device=z_gated.device)

        # Stack Top
        stack_top = self.stack(
            z_gated,
            push_gate=push_gate,
            pop_gate=pop_gate
        )

        # Activation + Norm
        z_out = self.modrelu(self.norm(z_gated + stack_top.unsqueeze(1)))

        # Divergence placeholder
        divergence = 0.0
        
        logits = self.lm_head(z_out.real)  # (B, L, vocab)

        return StepOutput(
            z=z_out,
            memory=z_out,
            ptr=stack_top,
            halt=torch.zeros(B, device=z.device),
            indices=(0,),
            loss=vq_loss,
            divergence=divergence
        )


# ==========================================
# 11. Model Initialization
# ==========================================
model_dim = CONFIG["embedding_dim"]
n_heads = CONFIG["n_heads"]
stack_size = CONFIG["stack_size"]

model = RecurrentNeuroSymbolicCell(model_dim, n_heads, stack_size, vocab_size=VOCAB_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

# FIX: Use new torch.amp API and handle CPU fallback
scaler = amp.GradScaler(enabled=CONFIG["use_amp"])

# ==========================================
# 12. Training Loop with KeyboardInterrupt
# ==========================================

def generate_sample_text(model, start_prompt="To be, or not to be, ", length=100):
    model.eval()
    
    prompt_ids = torch.tensor(
        [list(start_prompt.encode("utf-8"))],
        dtype=torch.long,
        device=DEVICE
    )

    generated_ids = prompt_ids.clone()
    context = torch.zeros(
        prompt_ids.shape[0],
        prompt_ids.shape[1],
        CONFIG["embedding_dim"],
        device=DEVICE
    )

    with torch.no_grad():
        for _ in range(length):
            step_out = model(generated_ids[:, -CONFIG["seq_len"]:], context)

            next_emb = step_out.z[:, -1, :].real
            logits = model.lm_head(next_emb)

            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1)

            generated_ids = torch.cat(
                [generated_ids, next_token_id.unsqueeze(0)],
                dim=1
            )

    chars = []
    for tid in generated_ids[0].cpu().tolist():
        chars.append(chr(tid) if tid < 256 else "?")

    return "".join(chars)


def train():
    try:
        for epoch in range(CONFIG["epochs"]):
            model.train()
            epoch_loss = 0.0
            epoch_entropy = []

            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                context = torch.zeros(
                    x.size(0),
                    x.size(1),
                    CONFIG["embedding_dim"],
                    device=DEVICE
                )

                with amp.autocast(
                    device_type="cuda",
                    enabled=CONFIG["use_amp"]
                ):
                    # Forward pass
                    step_out = model(x, context)

                    # -----------------------------
                    # STEP 1 — Token Prediction Loss
                    # -----------------------------
                    logits = model.lm_head(step_out.z.real)
                    prediction_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )

                    # -----------------------------
                    # STEP 2 — Entropy Floor
                    # -----------------------------
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                    target_entropy = 2.5
                    entropy_penalty = torch.clamp(target_entropy - entropy, min=0.0)

                    # -----------------------------
                    # Final Combined Loss
                    # -----------------------------
                    loss_final = (
                        CONFIG["weights"]["prediction"] * prediction_loss
                        + CONFIG["weights"]["vq"] * step_out.loss
                        + 0.05 * entropy_penalty
                    )

                # Backward pass with AMP
                scaler.scale(loss_final).backward()

                if CONFIG["grad_clip"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss_final.item()
                epoch_entropy.append(entropy.item())

                # -----------------------------
                # Batch-level Logging
                # -----------------------------
                if batch_idx % 512 == 0:
                    print(
                        f"Epoch {epoch+1}, "
                        f"Batch {batch_idx+1}/{len(train_loader)}, "
                        f"Loss: {loss_final.item():.6f}, "
                        f"Entropy: {entropy.item():.3f}, "
                        f"Entropy Penalty: {entropy_penalty.item():.3f}"
                    )

            # -----------------------------
            # Epoch-level Logging
            # -----------------------------
            avg_loss = epoch_loss / len(train_loader)
            avg_entropy = sum(epoch_entropy) / len(epoch_entropy)
            print(
                f"\nEpoch {epoch+1}/{CONFIG['epochs']} "
                f"Avg Loss: {avg_loss:.6f}, "
                f"Avg Entropy: {avg_entropy:.3f}\n"
            )

            # -----------------------------
            # Generate Sample Text
            # -----------------------------
            sample_text = generate_sample_text(
                model,
                start_prompt="To be, or not to be, ",
                length=200
            )
            print(f"=== Sample After Epoch {epoch+1} ===\n{sample_text}\n")

            # -----------------------------
            # Save Checkpoint
            # -----------------------------
            torch.save(
                model.state_dict(),
                f"checkpoint_epoch{epoch+1}.pt"
            )

        return model

    except KeyboardInterrupt:
        print("\nTraining interrupted — saving checkpoint.")
        torch.save(
            model.state_dict(),
            "trained_sacrsn_model_interrupted.pt"
        )
        return model

    except Exception:
        print("Training error:")
        traceback.print_exc()
        return model

# ==========================================
# 13. Execute Training
# ==========================================
trained_model = train()
print("Training Complete")

# Save the trained model if not already saved by KeyboardInterrupt
if not Path("trained_sacrsn_model.pt").exists():
    torch.save(trained_model.state_dict(), "trained_sacrsn_model.pt")
    print("Model saved to trained_sacrsn_model.pt")


