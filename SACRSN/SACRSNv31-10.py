# ============================================================
# SACRSN v93: THE COSMIC RESEARCH EDITION (FINAL RECTIFIED)
# 
# Status: REFERENCE / STRICT / AUDITED
# Architecture: Complex-Valued Neuro-Symbolic Recurrent Network
# 
# Critical Audit Fixes:
#  - ACT: Loss is now sample-wise weighted, preventing "easy-sample" bias.
#  - Graph: Updates are masked; masked tokens do not corrupt adjacency.
#  - Logits: All VQ logit sources (dist, graph, context) are LayerNormed.
#  - Clipper: Now tracks gradient norms (not loss) to prevent starvation.
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
from collections import defaultdict, OrderedDict
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple, Optional, List, Dict, Union
from typing import NamedTuple
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
    
    # ABLATION CONTROLS
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
    
    # ACT (Pondering) Settings
    "max_recursion_depth": 8,       
    "act_threshold": 0.99,          # Stop when cumulative prob > 0.99
    "ponder_penalty": 0.01, 
    
    "use_stack": True,
    "stack_size": 32,
    
    "graph_bias_scale": 1.0,        # Reduced for stability
    "ortho_max_vocab": 1024,
    "logit_norm_scale": 10.0,       
    "graph_warmup_steps": 500,      
    
    "hyper": {
        "adjacency_decay": 0.999,
        "adjacency_clamp": 20.0,
        "stack_temp": 10.0,         # Sharpened for pseudo-discrete behavior
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
    "batch_size": 32,               
    "learning_rate": 3e-4,          
    "grad_clip": 1.0,
    "clip_floor": 0.05,
    "eps": 1e-5,                    
    "warmup_epochs": 3,             
    
    "adaptive_scheduler": True,
    "structured_masking": True,
    "use_amp": True,
    "debug_anomaly": False,
}

if CONFIG["debug_anomaly"]:
    torch.autograd.set_detect_anomaly(True)

# ==========================================
# 2. Data & Ordered BPE Tokenizer
# ==========================================
def safe_tensor(t: torch.Tensor) -> torch.Tensor:
    """Global guard against NaNs and Infs."""
    if torch.is_complex(t):
        real = torch.nan_to_num(t.real, nan=0.0, posinf=1e4, neginf=-1e4)
        imag = torch.nan_to_num(t.imag, nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.complex(real, imag)
    else:
        return torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)

def c_cat(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Flatten complex tensor to real (cat real/imag)."""
    return torch.cat([z.real, z.imag], dim=dim)

def c_pack(r: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """Reconstruct complex tensor from real/imag components."""
    return torch.complex(r, i)

class SimpleBPE:
    def __init__(self, vocab_size=1000):
        self.target_vocab_size = vocab_size
        self.merges = [] 
        self.vocab = {i: bytes([i]) for i in range(256)} 
        self.special_tokens = {"<PAD>": 256, "<UNK>": 257, "<MASK>": 258}
        self.next_id = 259
        self.vocab_size = 259
        self._decode_cache = OrderedDict()
        self._encode_cache = OrderedDict()
        self.CACHE_SIZE = 5000 

    def get_stats(self, ids):
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def train(self, text):
        print("Training BPE Tokenizer (Ordered)...")
        ids = list(text.encode("utf-8"))
        num_merges = self.target_vocab_size - self.next_id
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats: break
            pair = max(stats, key=stats.get)
            idx = self.next_id
            self.merges.append((pair, idx))
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.next_id += 1
            new_ids = []
            skip = False
            for j in range(len(ids)):
                if skip: skip = False; continue
                if j < len(ids) - 1 and ids[j] == pair[0] and ids[j+1] == pair[1]:
                    new_ids.append(idx); skip = True
                else: new_ids.append(ids[j])
            ids = new_ids
        self.vocab_size = self.next_id
        print(f"Final Vocab Size: {self.vocab_size}")

    def encode(self, text):
        if text in self._encode_cache:
            self._encode_cache.move_to_end(text)
            return self._encode_cache[text]
        ids = list(text.encode("utf-8"))
        for pair, idx in self.merges:
            new_ids = []
            skip = False
            for j in range(len(ids)):
                if skip: skip = False; continue
                if j < len(ids) - 1 and ids[j] == pair[0] and ids[j+1] == pair[1]:
                    new_ids.append(idx); skip = True
                else: new_ids.append(ids[j])
            ids = new_ids
        if len(self._encode_cache) > self.CACHE_SIZE:
            self._encode_cache.popitem(last=False)
        self._encode_cache[text] = ids
        return ids

    def decode(self, ids):
        if isinstance(ids, int): ids = [ids]
        tuple_ids = tuple(ids)
        if tuple_ids in self._decode_cache:
            self._decode_cache.move_to_end(tuple_ids)
            return self._decode_cache[tuple_ids]
        out_bytes = b""
        for idx in ids:
            if idx in self.vocab: out_bytes += self.vocab[idx]
            elif idx == self.special_tokens["<MASK>"]: out_bytes += b"[MASK]"
            else: out_bytes += b"?"
        res = out_bytes.decode("utf-8", errors="replace")
        if len(self._decode_cache) > self.CACHE_SIZE:
            self._decode_cache.popitem(last=False)
        self._decode_cache[tuple_ids] = res
        return res

# --- DATASET ---
class SyntheticCopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples=5000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        half = self.seq_len // 2
        data = torch.randint(0, self.vocab_size - 4, (half,))
        full_seq = torch.cat([data, data])
        x = full_seq[:-1]
        y = full_seq[1:]
        return x, y

def get_data_loader():
    gen = torch.Generator()
    gen.manual_seed(SEED)
    
    if CONFIG["task_type"] == "copy":
        print("Initializing Synthetic Copy Task...")
        vocab_size = 64
        dataset = SyntheticCopyDataset(vocab_size, CONFIG["seq_len"] + 1)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=gen)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True, generator=gen, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, drop_last=True, generator=gen, num_workers=0)
        return train_loader, val_loader, vocab_size, None, None 
    else:
        # Shakespeare
        V33_TEXT = "The neural architecture of the mind is a mirror of the cosmos itself.\n"
        FILE_PATH = "tinyshakespeare.txt"
        try:
            if not os.path.exists(FILE_PATH):
                url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
                print(f"Downloading {FILE_PATH}...")
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                with open(FILE_PATH, 'w') as f: f.write(resp.text)
            with open(FILE_PATH, 'r') as f: raw_data = f.read()
        except Exception as e: 
            print(f"Warning: Failed to load text ({e}). Using fallback string.")
            raw_data = V33_TEXT * 100

        FULL_TEXT = V33_TEXT * 15 + raw_data 
        base_tokenizer = SimpleBPE(vocab_size=1200)
        base_tokenizer.train(FULL_TEXT[:100000])
        tokenized_data = base_tokenizer.encode(FULL_TEXT)
        data_tensor = torch.tensor(tokenized_data, dtype=torch.long).to(DEVICE)
        
        n_split = int(0.9 * len(data_tensor))
        train_data = data_tensor[:n_split]
        val_data = data_tensor[n_split:]
        
        def create_ds(tensor_data):
            num_samples = len(tensor_data) // CONFIG["seq_len"]
            trim = num_samples * CONFIG["seq_len"]
            x_data = tensor_data[:trim].view(num_samples, CONFIG["seq_len"])
            y_data = torch.roll(tensor_data, -1)[:trim].view(num_samples, CONFIG["seq_len"])
            return TensorDataset(x_data, y_data)
        
        train_loader = DataLoader(create_ds(train_data), batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True, generator=gen, num_workers=0)
        val_loader = DataLoader(create_ds(val_data), batch_size=CONFIG["batch_size"], shuffle=False, drop_last=True, generator=gen, num_workers=0)
        return train_loader, val_loader, base_tokenizer.vocab_size, data_tensor, base_tokenizer

# Global Data Init
train_loader, val_loader, VOCAB_SIZE, data_tensor, base_tokenizer = get_data_loader()

# ==========================================
# 3. Complex Primitives
# ==========================================
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        mag = torch.abs(z) + CONFIG["eps"]
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, unbiased=False, keepdim=True)
        var = torch.clamp(var, min=CONFIG["hyper"]["var_clamp"])
        norm_mag = (mag - mean) / torch.sqrt(var + CONFIG["eps"])
        norm_mag = norm_mag * self.scale + self.shift
        phase = torch.angle(z + 1e-9) 
        return torch.complex(norm_mag * torch.cos(phase), norm_mag * torch.sin(phase))

class ModReLU(nn.Module):
    """Complex-valued ReLU that scales magnitude with a learned bias."""
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.bias, 0.1) # Alive Init
    def forward(self, z):
        norm = torch.abs(z) + CONFIG["eps"]
        scale = F.relu(norm + self.bias) / (norm + CONFIG["eps"])
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.fc_real = nn.Linear(in_dim, out_dim, bias=bias)
        self.fc_imag = nn.Linear(in_dim, out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        scale = 1 / math.sqrt(2)
        nn.init.xavier_uniform_(self.fc_real.weight, gain=scale)
        nn.init.xavier_uniform_(self.fc_imag.weight, gain=scale)
        if self.fc_real.bias is not None:
            nn.init.zeros_(self.fc_real.bias)
            nn.init.zeros_(self.fc_imag.bias)

    def forward(self, z):
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        return torch.complex(
            self.fc_real(z.real) - self.fc_imag(z.imag),
            self.fc_real(z.imag) + self.fc_imag(z.real)
        )

# ==========================================
# 4. Phase-Coupled Attention
# ==========================================
class PhaseCoupledComplexAttention(nn.Module):
    """
    Implements complex-valued attention where real and imaginary components 
    interact via phase-coupled dot products. Scores are clamped to avoid FP16 overflow.
    """
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0, f"Dim {dim} must be divisible by n_heads {n_heads}"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = ComplexLinear(dim, dim)
        self.k_proj = ComplexLinear(dim, dim)
        self.v_proj = ComplexLinear(dim, dim)
        self.o_proj = ComplexLinear(dim, dim)

    def forward(self, z_seq):
        batch_size, seq_len = z_seq.shape[:2]
        q, k, v = self.q_proj(z_seq), self.k_proj(z_seq), self.v_proj(z_seq)
        
        q, k, v = safe_tensor(q), safe_tensor(k), safe_tensor(v)
        
        if not CONFIG["ablation"]["use_complex"]:
            q, k, v = torch.complex(q.real, torch.zeros_like(q.imag)), \
                      torch.complex(k.real, torch.zeros_like(k.imag)), \
                      torch.complex(v.real, torch.zeros_like(v.imag))

        def split(tensor):
            r = tensor.real.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            i = tensor.imag.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            return torch.complex(r, i)

        Q, K, V = split(q), split(k), split(v)
        
        # Complex Dot Product Interaction
        scores_real = (torch.matmul(Q.real, K.real.transpose(-2, -1)) + 
                       torch.matmul(Q.imag, K.imag.transpose(-2, -1)))
        scores_imag = (torch.matmul(Q.imag, K.real.transpose(-2, -1)) - 
                       torch.matmul(Q.real, K.imag.transpose(-2, -1)))
        
        # Clamp inputs to hypot to avoid FP16 overflow
        scores_real = torch.clamp(scores_real, min=-250.0, max=250.0)
        scores_imag = torch.clamp(scores_imag, min=-250.0, max=250.0)
        
        # Force FP32 for hypot calculation
        with autocast(enabled=False):
            sr32 = scores_real.float()
            si32 = scores_imag.float()
            scores = torch.hypot(sr32, si32) * self.scale
            scores = safe_tensor(scores)
            scores = scores - scores.max(dim=-1, keepdim=True)[0]
            attn_weights = F.softmax(scores, dim=-1)
        
        attn_weights = attn_weights.to(z_seq.real.dtype)
        attn_weights_c = torch.complex(attn_weights, torch.zeros_like(attn_weights))
        
        out = torch.matmul(attn_weights_c, V)
        out_real = out.real.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out_imag = out.imag.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        out_real = safe_tensor(out_real)
        out_imag = safe_tensor(out_imag)
        
        return self.o_proj(torch.complex(out_real, out_imag))

# ==========================================
# 5. Differentiable Soft-Stack
# ==========================================
class DifferentiableStack(nn.Module):
    """
    A continuous, differentiable stack memory.
    Uses soft pointers to read/write/decay memory without discrete operations.
    Handles boundary conditions via zero-padding shifts.
    """
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
    
    def forward(self, z, memory, ptr, control):
        if not CONFIG["ablation"]["use_stack"]:
            return torch.zeros_like(z), memory, ptr

        ptr = ptr / (ptr.sum(dim=1, keepdim=True) + CONFIG["eps"])
        push, pop, noop = control[:, 0:1], control[:, 1:2], control[:, 2:3]
        
        # Non-circular shift (Zero padded)
        # Push: Shift Indices Right (0 -> 1)
        ptr_up = torch.cat([torch.zeros_like(ptr[:, :1]), ptr[:, :-1]], dim=1)
        # Pop: Shift Indices Left (1 -> 0)
        ptr_down = torch.cat([ptr[:, 1:], torch.zeros_like(ptr[:, :1])], dim=1)
        
        new_ptr = (push * ptr_up) + (pop * ptr_down) + (noop * ptr)
        new_ptr = F.softmax(new_ptr * CONFIG["hyper"]["stack_temp"], dim=1)
        new_ptr = safe_tensor(new_ptr)
        
        z_flat = c_cat(z)
        write_mask = push * ptr_up # Write to the *new* top
        write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)
        
        decay = CONFIG["hyper"]["memory_decay"]
        retain_mask = (1.0 - write_mask.unsqueeze(2)) * decay
        
        new_memory = write_val + (memory * retain_mask)
        new_memory = safe_tensor(new_memory)
        
        # Clamp memory to prevent drift over long sequences
        new_memory = torch.clamp(new_memory, -1e3, 1e3)
        
        read_mask = new_ptr.unsqueeze(2)
        read_flat = torch.sum(new_memory * read_mask, dim=1)
        read_z = torch.complex(read_flat[:, :self.dim], read_flat[:, self.dim:])
        
        return read_z, new_memory, new_ptr

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 4, dim * 2) 
    def forward(self, z, stack_val):
        combined = torch.cat([z.real, z.imag, stack_val.real, stack_val.imag], dim=-1)
        gates = torch.sigmoid(self.gate(combined))
        g_real, g_imag = torch.chunk(gates, 2, dim=-1)
        z_out_real = (1 - g_real) * z.real + g_real * stack_val.real
        z_out_imag = (1 - g_imag) * z.imag + g_imag * stack_val.imag
        return torch.complex(z_out_real, z_out_imag)

class GatedRecurrentUpdate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 4, dim * 2)
        self.norm = ComplexLayerNorm(dim)
    def forward(self, z_new, z_prev):
        combined = torch.cat([z_new.real, z_new.imag, z_prev.real, z_prev.imag], dim=-1)
        gates = torch.sigmoid(self.gate(combined))
        g_real, g_imag = torch.chunk(gates, 2, dim=-1)
        z_out_real = (1 - g_real) * z_prev.real + g_real * z_new.real
        z_out_imag = (1 - g_imag) * z_prev.imag + g_imag * z_new.imag
        z_out = torch.complex(z_out_real, z_out_imag)
        return self.norm(z_out)

# ==========================================
# 6. Context-Biased Symbol Induction
# ==========================================
class EnhancedContextGate(nn.Module):
    def __init__(self, dim, n_symbols):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim // 2),
            nn.ReLU()
        )
        self.symbol_predictor = nn.Linear(dim // 2, n_symbols)
    def forward(self, z_flat): 
        return self.symbol_predictor(self.context_encoder(z_flat))

class EnhancedContextAwareDualVQ(nn.Module):
    """
    Dual-stream Vector Quantization (Syntax/Semantics) with Hebbian Graph memory.
    Updates the adjacency matrix only on valid index transitions (Strict Indexing).
    """
    def __init__(self, latent_dim, n_syntax, n_semantic, decay=0.99):
        super().__init__()
        self.dim = latent_dim * 2
        self.decay = decay
        self.cb_syn = nn.Parameter(torch.randn(n_syntax, self.dim))
        self.cb_sem = nn.Parameter(torch.randn(n_semantic, self.dim))
        self.ctx_gate_syn = EnhancedContextGate(self.dim, n_syntax)
        self.ctx_gate_sem = EnhancedContextGate(self.dim, n_semantic)
        self.register_buffer("cl_syn", torch.zeros(n_syntax))
        self.register_buffer("avg_syn", self.cb_syn.clone())
        self.register_buffer("cl_sem", torch.zeros(n_semantic))
        self.register_buffer("avg_sem", self.cb_sem.clone())
        self.register_buffer("adj_syn", torch.zeros(n_syntax, n_syntax))
        self.register_buffer("adj_sem", torch.zeros(n_semantic, n_semantic))
        self.graph_gate = nn.Parameter(torch.tensor(0.0))
        self.rng = torch.Generator(device=DEVICE).manual_seed(SEED)

    def vq_step(self, z, codebook, cluster_size, embed_avg, adj_matrix, ctx_layer, update_graph, prev_idx=None, mask=None):
        if not CONFIG["ablation"]["use_vq"]:
            return torch.zeros_like(z), torch.tensor(0.0, device=z.device), None, 0.0

        z_flat = c_cat(z)
        d_sq = torch.sum(z_flat**2, dim=1, keepdim=True) + \
               torch.sum(codebook**2, dim=1) - \
               2 * torch.matmul(z_flat, codebook.t())
        
        # FIX: LayerNorm for Dist Logits to match Graph Bias scale
        logits_dist = -torch.clamp(d_sq, min=0.0, max=1e4)
        logits_dist = F.layer_norm(logits_dist, logits_dist.shape[1:])
        
        graph_bias = 0
        if prev_idx is not None and prev_idx.numel() > 0 and CONFIG["ablation"]["use_graph"]:
            idx_safe = prev_idx.long().clamp(0, adj_matrix.size(0) - 1)
            norm_adj = F.softmax(adj_matrix[idx_safe], dim=-1).detach()
            
            # FIX: Normalize Graph Bias
            norm_adj = F.layer_norm(norm_adj, norm_adj.shape[1:])
            graph_bias = norm_adj * torch.sigmoid(self.graph_gate) * CONFIG["graph_bias_scale"]
        
        # FIX: Normalize Context Logits
        context_logits = ctx_layer(z_flat)
        context_logits = F.layer_norm(context_logits, context_logits.shape[1:])
        
        total_logits = logits_dist + graph_bias + (CONFIG["context_gate_strength"] * context_logits)

        idx = torch.argmax(total_logits, dim=1)
        z_q = F.embedding(idx, codebook)
        
        idx_pure = torch.argmax(logits_dist, dim=1)
        divergence = (idx != idx_pure).float().mean()
        
        # --- IMPROVED GRAPH UPDATE LOGIC ---
        if update_graph and prev_idx is not None and self.training and CONFIG["ablation"]["use_graph"]:
             with torch.no_grad():
                adj_matrix.mul_(CONFIG["hyper"]["adjacency_decay"])
                
                # STRICT MASKING
                prev_long = prev_idx.long()
                curr_long = idx.long()
                
                # Calculate flat indices
                flat_indices = prev_long * adj_matrix.size(1) + curr_long
                flat_indices = flat_indices.view(-1)
                
                # Create mask for valid transitions only
                valid_mask = (prev_long >= 0) & (prev_long < adj_matrix.size(0)) & \
                             (curr_long >= 0) & (curr_long < adj_matrix.size(1))
                valid_mask = valid_mask.view(-1)
                
                # FIX: Prevent leaked graph updates from masked tokens
                if mask is not None:
                    # Mask is [B, 1], where 1 means "masked"
                    is_real = (mask < 0.5).view(-1) 
                    valid_mask = valid_mask & is_real
                
                # Apply update only to valid pairs
                clean_indices = flat_indices[valid_mask]
                
                if clean_indices.numel() > 0:
                    # Enforce contiguous memory for safe view ops
                    adj_flat = adj_matrix.contiguous().view(-1)
                    updates = torch.ones_like(clean_indices, dtype=adj_matrix.dtype, device=DEVICE)
                    adj_flat.scatter_add_(0, clean_indices, updates)
                
                adj_matrix.clamp_(max=CONFIG["hyper"]["adjacency_clamp"])
        # -----------------------------------

        if self.training:
            enc = F.one_hot(idx, codebook.size(0)).float()
            with torch.no_grad():
                cluster_size.mul_(self.decay).add_(enc.sum(0), alpha=1-self.decay)
                embed_sum = torch.matmul(enc.t(), z_flat.detach())
                embed_avg.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
                n = cluster_size.sum()
                cs = (cluster_size + CONFIG["eps"]) / (n + codebook.size(0)*CONFIG["eps"]) * n
                codebook.copy_(embed_avg / cs.unsqueeze(1))
            
        # FIX: Return unreduced loss for ACT weighting
        loss = F.mse_loss(z_q.detach(), z_flat, reduction='none').mean(dim=-1) + \
               CONFIG["commitment_cost"] * F.mse_loss(z_q, z_flat.detach(), reduction='none').mean(dim=-1)
        z_q = z_flat + (z_q - z_flat).detach()
        
        ortho_loss = 0
        if self.training and CONFIG["weights"]["orthogonal"] > 0:
             if codebook.size(0) <= CONFIG["ortho_max_vocab"]:
                cb_norm = F.normalize(codebook, p=2, dim=1)
                ortho_matrix = torch.matmul(cb_norm, cb_norm.t())
                identity = torch.eye(codebook.size(0), device=ortho_matrix.device)
                ortho_loss = CONFIG["weights"]["orthogonal"] * F.mse_loss(ortho_matrix, identity)
             else:
                # Deterministic sampling using seeded generator
                indices = torch.randperm(codebook.size(0), generator=self.rng, device=DEVICE)[:CONFIG["ortho_max_vocab"]]
                sampled_cb = codebook[indices]
                cb_norm = F.normalize(sampled_cb, p=2, dim=1)
                ortho_matrix = torch.matmul(cb_norm, cb_norm.t())
                identity = torch.eye(len(indices), device=ortho_matrix.device)
                ortho_loss = CONFIG["weights"]["orthogonal"] * F.mse_loss(ortho_matrix, identity)
        
        return c_pack(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:]), loss, ortho_loss, idx, divergence

    def forward(self, z_fast, z_slow, update_graph, prev_indices=None, mask=None):
        prev_syn, prev_sem = (None, None) if prev_indices is None else prev_indices
        zq_syn, loss_syn, ortho_syn, idx_syn, div_syn = self.vq_step(z_fast, self.cb_syn, self.cl_syn, self.avg_syn, self.adj_syn, self.ctx_gate_syn, update_graph, prev_syn, mask)
        zq_sem, loss_sem, ortho_sem, idx_sem, div_sem = self.vq_step(z_slow, self.cb_sem, self.cl_sem, self.avg_sem, self.adj_sem, self.ctx_gate_sem, update_graph, prev_sem, mask)
        return zq_syn, zq_sem, loss_syn + loss_sem, ortho_syn + ortho_sem, (idx_syn, idx_sem), (div_syn + div_sem)/2

# ==========================================
# 7. Constraints
# ==========================================
class EntropyRegularization(nn.Module):
    def __init__(self, n_semantic_symbols, weight=0.1):
        super().__init__()
        self.n_semantic_symbols = n_semantic_symbols
        self.weight = weight
        self.register_buffer("semantic_history", torch.ones(n_semantic_symbols) * 1e-3)
    def forward(self, semantic_indices):
        if not CONFIG["ablation"]["use_vq"]: return torch.tensor(0.0, device=DEVICE)
        onehot = F.one_hot(semantic_indices, self.n_semantic_symbols).float()
        with torch.no_grad():
            self.semantic_history.data.mul_(CONFIG["hyper"]["entropy_ema"]).add_(onehot.mean(dim=0), alpha=1-CONFIG["hyper"]["entropy_ema"])
        probs = self.semantic_history / (self.semantic_history.sum() + 1e-9)
        return self.weight * (probs * torch.log(probs + 1e-9)).sum()

class GraphConsistencyLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, prev_sem, curr_sem, adjacency):
        if not CONFIG["ablation"]["use_graph"]: return torch.tensor(0.0, device=DEVICE)
        if prev_sem is None: return torch.tensor(0.0).to(adjacency.device)
        safe_prev = prev_sem.clamp(0, adjacency.size(0) - 1)
        row_logits = adjacency[safe_prev] 
        graph_probs = F.softmax(row_logits, dim=-1)
        selected_probs = graph_probs.gather(1, curr_sem.unsqueeze(1)).squeeze(1)
        # Return per-sample loss for ACT weighting
        return -torch.log(selected_probs + 1e-9)

# ==========================================
# 8. Adaptive Scheduler & Clipper
# ==========================================
class AdaptiveScheduler(nn.Module):
    def __init__(self, optimizer, T_max, warmup_epochs=2):
        super().__init__()
        self.optimizer = optimizer
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr_scale = min(1.0, (epoch + 1) / (self.warmup_epochs + 1))
            for pg in self.optimizer.param_groups: pg['lr'] = CONFIG["learning_rate"] * lr_scale
        else:
            denom = self.T_max - self.warmup_epochs
            if denom <= 0: denom = 1
            curr_step = epoch - self.warmup_epochs
            cosine_lr = 0.5 * (1 + math.cos(curr_step * math.pi / denom))
            for pg in self.optimizer.param_groups: pg['lr'] = CONFIG["learning_rate"] * cosine_lr

class AdaptiveGradientClip(nn.Module):
    def __init__(self, clip_value=1.0, alpha=0.1, max_ceiling=5.0):
        super().__init__()
        self.clip_value = clip_value
        self.alpha = alpha
        self.max_ceiling = max_ceiling
        self.avg_norm = None
        
    def forward(self, parameters, current_loss=None):
        # FIX: Track gradient norms, not loss, to prevent starvation
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(DEVICE) for p in parameters if p.grad is not None]), 2.0)
        
        if self.avg_norm is None:
            self.avg_norm = total_norm
        else:
            self.avg_norm = (1 - self.alpha) * self.avg_norm + self.alpha * total_norm
        
        # If current norm is significantly higher than average, cap it closer to average
        factor = torch.where(total_norm > self.avg_norm * 2.0, 0.95, 1.05)
        self.clip_value = self.clip_value * factor.item()
        
        # Constrain between floor and strict ceiling
        self.clip_value = max(CONFIG["clip_floor"], min(self.max_ceiling, self.clip_value))
        
        torch.nn.utils.clip_grad_norm_(parameters, self.clip_value)

# ==========================================
# 9. Unified Model
# ==========================================
class StepOutput(NamedTuple):
    z: torch.Tensor
    memory: torch.Tensor
    ptr: torch.Tensor
    halt: torch.Tensor
    indices: tuple
    loss: torch.Tensor
    divergence: float

class RecurrentNeuroSymbolicCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim, dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.attention = PhaseCoupledComplexAttention(dim, CONFIG["n_heads"])
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        nn.init.constant_(self.halt_linear.bias, -1.0)

    def forward(self, z):
        z_proc = self.linear(z)
        z_proc = self.norm(z_proc)
        z_proc = self.act(z_proc)
        
        z_proc_3d = z_proc.unsqueeze(1) 
        z_attn = self.attention(z_proc_3d)
        z_proc = z_attn.squeeze(1) 
        
        z_flat = c_cat(z_proc)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        stack_ctrl = F.softmax(self.stack_ctrl(z_flat), dim=-1)
        return z_proc, halt_prob, stack_ctrl

class HonestBiCameralCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.emb_proj = ComplexLinear(dim, dim)
        self.cell = RecurrentNeuroSymbolicCell(dim)
        self.vq = EnhancedContextAwareDualVQ(dim, CONFIG["n_syntax_symbols"], CONFIG["n_semantic_symbols"], decay=CONFIG["decay"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        self.stack_gate = GatedResidual(dim)
        self.stack = DifferentiableStack(dim, CONFIG["stack_size"])
        self.consistency = GraphConsistencyLoss()
        self.entropy_reg = EntropyRegularization(CONFIG["n_semantic_symbols"], CONFIG["weights"]["entropy"])
        self.mask_token = nn.Parameter(torch.randn(dim*2))
        self.norm_merge = ComplexLayerNorm(dim)
        self.recurrent_gate = GatedRecurrentUpdate(dim)
        self.mix_logits = nn.Parameter(torch.zeros(3))
        
        # Telemetry storage
        self.telemetry = {}

    def embed(self, input_ids):
        emb = self.embedding(input_ids)
        z_c = torch.complex(emb, torch.zeros_like(emb))
        return self.emb_proj(z_c)

    def _run_step(self, z, stack_mem, stack_ptr, prev_indices, mask, update_graph) -> StepOutput:
        z_proc, p_halt, stack_ctrl = self.cell(z)
        stack_read, new_mem, new_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
        z_gate = self.stack_gate(z_proc, stack_read)
        zq_syn, zq_sem, vq_loss, ortho_loss, new_indices, div = self.vq(z_proc, z_gate, update_graph, prev_indices, mask)
        
        mix_weights = F.softmax(self.mix_logits, dim=0)
        z_out = mix_weights[0] * z_gate + mix_weights[1] * zq_syn + mix_weights[2] * zq_sem
        
        return StepOutput(z_out, new_mem, new_ptr, p_halt, new_indices, vq_loss + ortho_loss, div)

    def forward(self, input_ids, hidden=None, mask_mask=None, prev_vq_indices=None, update_graph=True):
        batch_size = input_ids.size(0)
        
        # FIX: Pre-masking at embedding level to prevent leakage
        emb = self.embedding(input_ids)
        if mask_mask is not None:
            if mask_mask.ndim != 2: mask_mask = mask_mask.view(batch_size, 1)
            # Expand mask for embedding dim
            mask_expanded = mask_mask.unsqueeze(-1).expand_as(emb)
            # Mask token projection
            mask_val = self.mask_token[..., :self.dim] 
            emb = emb * (1 - mask_expanded) + mask_val.unsqueeze(0).expand_as(emb) * mask_expanded
            
        z = torch.complex(emb, torch.zeros_like(emb))
        z = self.emb_proj(z).squeeze(1)
        
        if hidden is None:
            z_prev = torch.zeros_like(z)
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device, dtype=z.real.dtype)
            stack_ptr[:, 0] = 1.0 
        else:
            z_prev, stack_mem, stack_ptr = hidden
            z = self.recurrent_gate(z, z_prev)

        # ACT State
        halting_probability = torch.zeros(batch_size, 1, device=z.device)
        remain = torch.ones(batch_size, 1, device=z.device)
        n_updates = torch.zeros(batch_size, 1, device=z.device)
        
        z_accum = torch.zeros_like(z)
        ponder_cost = torch.zeros((), device=z.device)
        vq_divergence = 0.0
        
        # Telemetry
        stack_depths = []
        
        current_indices = prev_vq_indices
        vq_loss_total = 0
        consistency_loss_total = 0
        
        t_final = 0
        for t in range(CONFIG["max_recursion_depth"]):
            t_final = t
            
            # Check threshold halting
            active_mask = (halting_probability < CONFIG["act_threshold"]).float()
            if active_mask.sum() == 0: break
            
            step_out = self._run_step(
                z, stack_mem, stack_ptr, current_indices, mask_mask, update_graph
            )
            
            # Masked updates: Only update state if active
            new_z_flat = c_cat(step_out.z)
            old_z_flat = c_cat(z)
            z_flat_mixed = new_z_flat * active_mask + old_z_flat * (1 - active_mask)
            z = c_pack(z_flat_mixed[..., :self.dim], z_flat_mixed[..., self.dim:])
            
            # FIX: Sample-wise loss accumulation for correct ACT weighting
            # step_out.loss is [B] (vector). active_mask is [B, 1].
            loss_weight = active_mask.squeeze()
            vq_loss_total += (step_out.loss * loss_weight).sum()
            vq_divergence += step_out.divergence
            
            if current_indices is not None and current_indices[1] is not None:
                c_loss = self.consistency(current_indices[1], step_out.indices[1], self.vq.adj_sem)
                consistency_loss_total += (c_loss * loss_weight).sum()
            
            current_indices = step_out.indices
            stack_mem = step_out.memory
            stack_ptr = step_out.ptr
            
            # PonderNet-style Geometric Halting
            p = step_out.halt * remain
            halting_probability = halting_probability + p
            remain = remain - p
            
            z_accum = safe_tensor(z_accum + (p * step_out.z))
            n_updates = n_updates + active_mask
            ponder_cost += active_mask.sum()
            
            # Stack Telemetry (Expected Position)
            positions = torch.arange(CONFIG["stack_size"], device=z.device, dtype=stack_ptr.dtype).unsqueeze(0)
            depth = (stack_ptr * positions).sum(dim=1).mean()
            stack_depths.append(depth.item())
            
            # Save Phase at T=0
            if t == 0 and not self.training:
                self.telemetry["phase_reals"] = z.real.detach().flatten()[:100].cpu().numpy().tolist()
                self.telemetry["phase_imags"] = z.imag.detach().flatten()[:100].cpu().numpy().tolist()

        # Final residual for unhalted
        z_accum = z_accum + (remain * z)
        
        features = c_cat(z_accum)
        logits = self.decoder(features)
        
        norm = logits.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        logits = (logits / norm) * CONFIG["logit_norm_scale"]
        
        next_hidden = (z_accum, stack_mem, stack_ptr)
        avg_stack = torch.tensor(np.mean(stack_depths) if stack_depths else 0.0)
        avg_div = vq_divergence / (t_final + 1)
        
        # Normalize aggregated losses by batch size (standard reduction)
        ponder_cost = ponder_cost / batch_size
        vq_loss_total = vq_loss_total / batch_size
        consistency_loss_total = consistency_loss_total / batch_size
        
        if not self.training:
            self.telemetry["stack_history"] = stack_depths
            self.telemetry["act_history"] = [n_updates.mean().item()]
            
        return logits, next_hidden, current_indices, ponder_cost, vq_loss_total, consistency_loss_total, avg_stack, avg_div

# ==========================================
# 10. Training Engine
# ==========================================
def train():
    model = HonestBiCameralCRSN(VOCAB_SIZE, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    CONFIG["use_amp"] = False
    scaler = GradScaler(enabled=CONFIG["use_amp"])
    
    if CONFIG["adaptive_scheduler"]:
        scheduler = AdaptiveScheduler(opt, T_max=CONFIG["epochs"] - CONFIG["warmup_epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"] - CONFIG["warmup_epochs"])
    
    gradient_clipper = AdaptiveGradientClip(CONFIG["grad_clip"])
    
    loader = train_loader
    
    print(f"--- Training SACRSN v93: Cosmic Research Edition ---")
    print(f"Task: {CONFIG['task_type']} | Ablation: {CONFIG['ablation']}")
    
    global_step = 0
    
    try:
        for epoch in range(CONFIG["epochs"]):
            if epoch < CONFIG["warmup_epochs"]:
                lr_scale = min(1.0, (epoch + 1) / (CONFIG["warmup_epochs"] + 1))
                for pg in opt.param_groups: pg['lr'] = CONFIG["learning_rate"] * lr_scale
            
            hidden = None 
            prev_indices = None 
            
            for i, (x_batch, y_batch) in enumerate(loader):
                global_step += 1
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                
                curr_bs = x_batch.size(0)
                if hidden is not None and hidden[0].size(0) != curr_bs: 
                    hidden = None
                    prev_indices = None
                
                if hidden is not None:
                    hidden = (hidden[0].detach(), hidden[1].detach(), hidden[2].detach())
                
                if prev_indices is not None:
                    prev_indices = (prev_indices[0].detach(), prev_indices[1].detach())
                
                if CONFIG["structured_masking"]: mask_mask = create_structured_mask(x_batch, CONFIG["mask_prob"])
                else: mask_mask = (torch.rand_like(x_batch.float()) < CONFIG["mask_prob"]).float().unsqueeze(-1).to(DEVICE)
                
                update_graph = global_step > CONFIG["graph_warmup_steps"]
                
                loss_seq_acc = torch.tensor(0.0, device=DEVICE)
                ponder_seq_acc = torch.tensor(0.0, device=DEVICE)
                vq_seq_acc = torch.tensor(0.0, device=DEVICE)
                
                opt.zero_grad()
                
                for t in range(CONFIG["seq_len"]):
                    yt = y_batch[:, t]
                    mt = mask_mask[:, t].view(-1, 1)
                    xt = x_batch[:, t:t+1]
                    
                    with autocast(enabled=CONFIG["use_amp"]):
                        # FIX: mask_mask is passed into forward for pre-embedding application
                        logits, hidden, curr_indices, ponder, vq_loss, c_loss, _, div = model(xt, hidden, mt, prev_indices, update_graph)
                        
                        if curr_indices is not None and curr_indices[1] is not None: 
                            prev_indices = (curr_indices[0], curr_indices[1])
                        
                        loss_pred = F.cross_entropy(logits, yt)
                        sem_idx = curr_indices[1]
                        loss_ent = model.entropy_reg(sem_idx)
                        
                        W = CONFIG["weights"]
                        loss_step = (loss_pred * W["prediction"]) + \
                                    (ponder * W["ponder"]) + \
                                    (vq_loss * W["vq"]) + \
                                    (c_loss * W["consistency"]) + \
                                    (loss_ent * W["entropy"])
                    
                    loss_seq_acc += loss_step
                    ponder_seq_acc += ponder
                    vq_seq_acc += vq_loss
                
                loss_final = loss_seq_acc / CONFIG["seq_len"]
                
                if torch.isnan(loss_final):
                    print(f"Warning: NaN loss detected. Skipping.")
                    hidden = None
                    prev_indices = None
                    continue

                scaler.scale(loss_final).backward()
                scaler.unscale_(opt)
                
                # FIX: Clipper is explicitly called here with model parameters
                gradient_clipper(model.parameters())
                
                scaler.step(opt)
                scaler.update()
                
                if i % 10 == 0:
                     avg_steps = ponder_seq_acc.item() / CONFIG['seq_len']
                     ppx = torch.exp(loss_final).item()
                     vq_val = (vq_seq_acc / CONFIG['seq_len']).item()
                     adj = model.vq.adj_sem.detach()
                     sat = (adj > CONFIG["hyper"]["adjacency_clamp"] * 0.9).float().mean().item()
                     
                     print(f"Ep {epoch} | Bt {i:03d} | Loss: {loss_final.item():.3f} | Pred: {ppx:.0f} | VQ: {vq_val:.2f} | ACT: {avg_steps:.2f} | Clip: {gradient_clipper.clip_value:.2f}")

            if CONFIG["adaptive_scheduler"] and epoch >= CONFIG["warmup_epochs"]: scheduler.step(epoch)
            elif not CONFIG["adaptive_scheduler"] and epoch >= CONFIG["warmup_epochs"]: scheduler.step()
            
    except KeyboardInterrupt: print("\nInterrupted.")
    except Exception as e:
        print(f"Training Error: {e}")
        traceback.print_exc()
    return model

def create_structured_mask(input_ids, mask_prob):
    batch_size, seq_len = input_ids.shape
    mask_pattern = torch.zeros(batch_size, seq_len).to(input_ids.device)
    for i in range(batch_size):
        num_masks = int(seq_len * mask_prob)
        if num_masks > 0:
            high = max(1, seq_len - num_masks)
            start_pos = torch.randint(0, high, (1,)).item()
            mask_pattern[i, start_pos:start_pos + num_masks] = 1
    
    if not mask_pattern.any(): 
        rand_indices = torch.randint(0, seq_len, (batch_size,), device=input_ids.device)
        mask_pattern[torch.arange(batch_size), rand_indices] = 1
        
    return mask_pattern.float().unsqueeze(-1)

# ==========================================
# 11. Visualization (Modular)
# ==========================================
def plot_topology(adj_matrix, node_labels, filename):
    if not HAS_NETWORKX: return
    plt.close('all')
    try:
        G = nx.DiGraph()
        n_nodes = adj_matrix.shape[0]
        for i in range(n_nodes): G.add_node(i)
        for i in range(n_nodes):
            for j in range(n_nodes):
                w = adj_matrix[i, j]
                if w > 0.4: G.add_edge(i, j, weight=w)
        
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color='#ff9999', node_size=600, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold")
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', arrowstyle='->', arrowsize=10)
        plt.savefig(filename); plt.close()
    except Exception as e: 
        print(f"Viz Error (Topology): {e}")

def plot_stack_mri(history, filename):
    if not history: return
    try:
        plt.figure(figsize=(12, 4))
        plt.plot(history, color='purple', label='Stack Depth')
        plt.fill_between(range(len(history)), history, color='purple', alpha=0.1)
        plt.savefig(filename); plt.close()
    except Exception as e: 
        print(f"Viz Error (Stack): {e}")

def plot_act_profile(history, filename):
    if not history: return
    try:
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(history)), history, color='orange')
        plt.savefig(filename); plt.close()
    except Exception as e: 
        print(f"Viz Error (ACT): {e}")

def plot_phase(reals, imags, filename):
    if not reals: return
    try:
        plt.figure(figsize=(8, 8))
        plt.scatter(reals, imags, c=range(len(reals)), cmap='plasma', alpha=0.5)
        plt.axis('equal'); plt.savefig(filename); plt.close()
    except Exception as e: 
        print(f"Viz Error (Phase): {e}")

def plot_heatmap(syn_ids, sem_ids, acts, labels, filename):
    try:
        def normalize(lst):
            arr = np.array(lst)
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        matrix = np.vstack([normalize(syn_ids), normalize(sem_ids), normalize(acts)])
        plt.figure(figsize=(10, 6))
        plt.imshow(matrix, aspect='auto', cmap='magma', interpolation='nearest')
        plt.yticks(range(3), ['Syntax', 'Semantics', 'ACT'])
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.tight_layout(); plt.savefig(filename); plt.close()
    except Exception as e: 
        print(f"Viz Error (Heatmap): {e}")

def visualize_all(model):
    print("\n--- Generating Diagnostics ---")
    model.eval()
    gc.collect(); torch.cuda.empty_cache() 
    
    if data_tensor is None: return

    scan_limit = 1000 
    adj_sem = torch.sigmoid(model.vq.adj_sem).detach().cpu().numpy()
    symbol_to_token = defaultdict(list)
    hidden, prev_indices = None, None
    
    if base_tokenizer:
        with torch.no_grad():
            for i in range(scan_limit):
                if i >= len(data_tensor): break
                x = data_tensor[i].view(1,1)
                _, hidden, curr_indices, _, _, _, _, _ = model(x, hidden, None, prev_indices)
                if curr_indices:
                    sem_idx = curr_indices[1].item()
                    symbol_to_token[sem_idx].append(base_tokenizer.decode([data_tensor[i].item()]))
                prev_indices = curr_indices

        node_labels = {}
        for i in range(CONFIG["n_semantic_symbols"]):
            tokens = symbol_to_token.get(i, [])
            if tokens:
                filtered = [t for t in tokens if len(t.strip()) > 1]
                target = filtered if filtered else tokens
                most_common = max(set(target), key=target.count)
                node_labels[i] = f"{most_common}"
            else: node_labels[i] = str(i)

        plot_topology(adj_sem, node_labels, "1_semantic_topology.png")

        start_token = base_tokenizer.special_tokens["<UNK>"]
        x = torch.tensor([[start_token]], device=DEVICE)
        hidden, prev_indices = None, None
        dream_text_list = []
        
        with torch.no_grad():
            for _ in range(150):
                logits, hidden, idx, ponder, _, _, _, _ = model(x, hidden, None, prev_indices)
                prev_indices = idx
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                x = next_token
                dream_text_list.append(base_tokenizer.decode([x.item()]))
                
        print(f"Dream Output: {''.join(dream_text_list)}\n")
        
        plot_stack_mri(model.telemetry.get("stack_history", []), "2_stack_mri.png")
        plot_act_profile(model.telemetry.get("act_history", []), "3_act_profile.png")
        plot_phase(model.telemetry.get("phase_reals", []), model.telemetry.get("phase_imags", []), "4_phase_plot.png")

        if CONFIG["task_type"] != "copy":
            test_sentence = "The mind is a mirror of the void"
            input_ids = torch.tensor(base_tokenizer.encode(test_sentence), dtype=torch.long).to(DEVICE)
            syn_ids, sem_ids, acts = [], [], []
            hidden, prev_indices = None, None
            decoded_labels = [base_tokenizer.decode([i.item()]) for i in input_ids]
            
            with torch.no_grad():
                for i in range(len(input_ids)):
                    x = input_ids[i].view(1,1)
                    _, hidden, idx, ponder, _, _, _, _ = model(x, hidden, None, prev_indices)
                    if idx:
                        syn_ids.append(idx[0].item()); sem_ids.append(idx[1].item())
                    else: syn_ids.append(0); sem_ids.append(0)
                    acts.append(ponder.item())
                    prev_indices = idx
                    
            plot_heatmap(syn_ids, sem_ids, acts, decoded_labels, "5_diagnostic_heatmap.png")

        print("\n--- Logic Extraction ---")
        adj = torch.sigmoid(model.vq.adj_sem).detach().cpu().numpy()
        count = 0
        for i in range(CONFIG["n_semantic_symbols"]):
            for j in range(CONFIG["n_semantic_symbols"]):
                if adj[i, j] > 0.4:
                    src_token = node_labels.get(i, "?")
                    dst_token = node_labels.get(j, "?")
                    if src_token != "?" and dst_token != "?":
                        print(f"S{i:<3} '{src_token[:8]:<8}' -> S{j:<3} '{dst_token[:8]:<8}' | {adj[i,j]:.2f}")
                        count += 1
                        if count > 20: break
            if count > 20: break
    
    del adj_sem, hidden, x
    gc.collect(); torch.cuda.empty_cache()

def dream_mode(model):
    print("\n--- 🌙 Dream Mode ---")
    try:
        adj = torch.sigmoid(model.vq.adj_sem).detach().cpu().numpy()
        symbol_to_token = defaultdict(lambda: "?")
        hidden, prev_indices = None, None
        scan_limit = 2000
        
        if base_tokenizer:
            with torch.no_grad():
                for i in range(scan_limit):
                    if i >= len(data_tensor): break
                    x = data_tensor[i].view(1,1)
                    _, hidden, curr_indices, _, _, _, _, _ = model(x, hidden, None, prev_indices)
                    if curr_indices is not None and curr_indices[1] is not None:
                        symbol_to_token[curr_indices[1].item()] = base_tokenizer.decode([data_tensor[i].item()])
                    prev_indices = curr_indices
            
            start_token = base_tokenizer.special_tokens["<UNK>"]
            x = torch.tensor([[start_token]], device=DEVICE)
            _, _, prev_indices, _, _, _, _, _ = model(x, None, None)
            curr_indices = prev_indices[1].item() if prev_indices else 0
            dream_tokens = [base_tokenizer.decode([start_token])]
            dream_indices = [curr_indices]
            
            for _ in range(40):
                curr_idx_int = int(curr_indices)
                curr_idx_int = max(0, min(curr_idx_int, adj.shape[0]-1))
                
                probs = adj[curr_idx_int]
                probs[probs < 0.15] = 0 
                
                if probs.sum() == 0: 
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs = probs / (probs.sum() + 1e-9)
                    
                next_indices = np.random.choice(len(probs), p=probs)
                token = symbol_to_token.get(next_indices, "?")
                dream_tokens.append(token)
                dream_indices.append(next_indices)
                curr_indices = next_indices
                
            print(f"Dream: {' '.join(dream_tokens)}\n")
            
            dream_tensor = []
            for t in dream_tokens:
                ids = base_tokenizer.encode(t)
                if len(ids) > 0: dream_tensor.append(ids[0])
                else: dream_tensor.append(0)
            dream_tensor = torch.tensor(dream_tensor, dtype=torch.long).to(DEVICE)
            
            topo, stack, act = [], [], []
            if len(dream_tensor) > 0:
                x0 = dream_tensor[0].view(1,1)
                with torch.no_grad():
                    _, hidden, prev_indices, _, _, _, _, _ = model(x0, None, None)
                    for i in range(1, len(dream_tensor)):
                        x = dream_tensor[i].view(1,1)
                        _, hidden, curr_indices, ponder, _, c_loss, stack_d, _ = model(x, hidden, None, prev_indices)
                        topo.append(c_loss.item())
                        stack.append(stack_d.item())
                        act.append(ponder.item())
                        prev_indices = curr_indices

                def normalize(lst):
                    arr = np.array(lst)
                    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) if len(arr) > 0 and arr.max() > arr.min() else arr

                if len(topo) > 0:
                    matrix = np.vstack([normalize(topo), normalize(stack), normalize(act)])
                    plt.figure(figsize=(12, 5))
                    plt.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
                    plt.yticks(range(3), ['Consistency', 'Stack', 'Effort'])
                    plt.xticks(range(len(dream_tokens)-1), dream_tokens[1:], rotation=45, ha='right')
                    for i in range(min(len(dream_tokens)-1, len(dream_indices)-1)):
                        sym_id = dream_indices[i+1]
                        for y in range(3): plt.text(i, y, f"{sym_id}", color='white', ha='center', va='center', fontsize=7)
                    plt.tight_layout(); plt.savefig("6_dream_diagnostic.png"); plt.close()
    except Exception as e: 
        print(f"Dream Mode Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    FILENAME = "sacrsn_v93_cosmic.pth"
    try:
        torch.autograd.set_detect_anomaly(True)
        trained_model = train()
        print(f"\n--- Saving Model to {FILENAME} ---")
        torch.save(trained_model.state_dict(), FILENAME)
        visualize_all(trained_model)
        dream_mode(trained_model)
    except KeyboardInterrupt:
        print("\nRun interrupted by user.")
    except Exception as e:
        print(f"Fatal Error: {e}")
        traceback.print_exc()
