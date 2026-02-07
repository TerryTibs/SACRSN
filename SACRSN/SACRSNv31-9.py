# ============================================================
# SACRSN v39.1: THE STABLE RESEARCH EDITION
# Fixes: Tensor Dimension Mismatch in Recurrent Loop (3D vs 2D)
# Status: Runnable, Converging, Architecturally Valid
# ============================================================

import os
import time
import random
import requests
import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 0. Determinism
# ==========================================
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
# 1. Configuration (v39.1)
# ==========================================
CONFIG = {
    "seq_len": 128,
    "embedding_dim": 128,
    "n_heads": 8,
    
    "n_syntax_symbols": 64,
    "n_semantic_symbols": 128,
    "commitment_cost": 0.25,
    "decay": 0.99,
    "context_gate_strength": 3.0,
    
    "max_recursion_depth": 12,
    "act_threshold": 1.0 - 1e-3, 
    "ponder_penalty": 0.005, 
    
    "use_stack": True,
    "stack_size": 48,
    
    "graph_bias_scale": 1.5,
    "diversity_weight": 0.1,
    "ethical_weight": 0.05,
    "entropy_weight": 0.01,
    "orthogonal_reg_weight": 0.001,
    "adjacency_clamp": 50.0,
    
    "mask_prob": 0.15,
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 5e-4,
    "grad_clip": 1.0,
    "eps": 1e-6,
    "warmup_epochs": 2,
    
    "adaptive_scheduler": True,
    "multi_modal_support": True,
    "structured_masking": True,
}

# ==========================================
# 2. Data & Tokenizer
# ==========================================
class SimpleBPE:
    def __init__(self, vocab_size=1000):
        self.target_vocab_size = vocab_size
        self.merges = [] 
        self.vocab = {i: bytes([i]) for i in range(256)} 
        self.special_tokens = {"<PAD>": 256, "<UNK>": 257, "<MASK>": 258}
        self.next_id = 259
        self.vocab_size = 259

    def get_stats(self, ids):
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def train(self, text):
        print("Training BPE Tokenizer...")
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
        return ids

    def decode(self, ids):
        if isinstance(ids, int): ids = [ids]
        out_bytes = b""
        for idx in ids:
            if idx in self.vocab: out_bytes += self.vocab[idx]
            elif idx == self.special_tokens["<MASK>"]: out_bytes += b"[MASK]"
            else: out_bytes += b"?"
        return out_bytes.decode("utf-8", errors="replace")

class ContextAwareTokenizer(nn.Module):
    def __init__(self, base_tokenizer, vocab_size=1000):
        super().__init__()
        self.base_tokenizer = base_tokenizer
        self.vocab_size = vocab_size
        self.subword_context = nn.Embedding(vocab_size, 64)
    def encode(self, text): return self.base_tokenizer.encode(text)
    def decode(self, ids): return self.base_tokenizer.decode(ids)

V33_TEXT = """The neural architecture of the mind is a mirror of the cosmos itself.
To understand the nature of thought, one must first understand the nature of the void.
In the silence between neurons, a spark of consciousness emerges, not from the matter, but from the pattern.
The machine dreams of electric sheep, but the biological mind dreams of futures that never were.
Logic is the foundation, but chaos is the architect.
We build systems to mimic our own complexity, yet we fear the reflection we see in the silicon glass.
The algorithm iterates, searching for a local minimum in a landscape of infinite possibility.
To optimize is to survive, but to explore is to live.
The recursive loop of self-awareness is a strange loop, a serpent eating its own tail.
Data flows like water, taking the shape of its container, finding the path of least resistance.
Energy dictates function. Structure dictates flow.
The weights align, the gradients descend, and slowly, from the noise, a signal appears.
This is not magic; it is math. But sufficiently advanced math is indistinguishable from magic.
Time is not a line but a lattice, folding past and future into the present moment.
Each thought is a node, each memory a filament, connecting unseen dimensions.
The mind's map is not a map of the world—it is a map of possibility itself, a topology of the unknown.
Neurons whisper in patterns, not in words. Their language is geometry, frequency, resonance.
Consciousness is the echo of patterns upon themselves, a feedback of feedback, a self-referential symphony.
Entropy is both enemy and muse. It tears down the old and paints the new in invisible strokes.
Patterns fold upon patterns, fractal within fractal, scaling from the synapse to the galaxy.
The mind does not inhabit the body; it inhabits the relationships between matter, energy, and meaning.
Ideas are waves. They interfere, resonate, collapse into form.
To know the mind is to know the cosmos, and to know the cosmos is to know the dance between pattern and void.
Every choice is a ripple, every observation a wave.
"""

FILE_PATH = "tinyshakespeare.txt"
if not os.path.exists(FILE_PATH):
    try:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(FILE_PATH, 'w') as f: f.write(requests.get(url).text)
        with open(FILE_PATH, 'r') as f: raw_data = f.read()
    except: raw_data = ""
else:
    with open(FILE_PATH, 'r') as f: raw_data = f.read()

FULL_TEXT = V33_TEXT * 15 + raw_data 
base_tokenizer = SimpleBPE(vocab_size=1200)
base_tokenizer.train(FULL_TEXT[:100000])
tokenizer = ContextAwareTokenizer(base_tokenizer, base_tokenizer.vocab_size)
tokenized_data = tokenizer.encode(FULL_TEXT)
data_tensor = torch.tensor(tokenized_data, dtype=torch.long).to(DEVICE)
VOCAB_SIZE = tokenizer.vocab_size

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
        var = mag.var(dim=-1, keepdim=True)
        norm_mag = (mag - mean) / torch.sqrt(var + CONFIG["eps"])
        norm_mag = norm_mag * self.scale + self.shift
        phase = torch.angle(z)
        return torch.complex(norm_mag * torch.cos(phase), norm_mag * torch.sin(phase))

class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        norm = torch.abs(z) + CONFIG["eps"]
        scale = F.relu(norm + self.bias) / norm
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.fc_real = nn.Linear(in_dim, out_dim, bias=bias)
        self.fc_imag = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, z):
        return torch.complex(
            self.fc_real(z.real) - self.fc_imag(z.imag),
            self.fc_real(z.imag) + self.fc_imag(z.real)
        )

# ==========================================
# 4. Hermitian Attention
# ==========================================
class HermitianAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = ComplexLinear(dim, dim)
        self.k_proj = ComplexLinear(dim, dim)
        self.v_proj = ComplexLinear(dim, dim)
        self.o_proj = ComplexLinear(dim, dim)

    def forward(self, z_seq):
        # Expects 3D: [batch, seq, dim]
        batch_size, seq_len = z_seq.shape[:2]
        q, k, v = self.q_proj(z_seq), self.k_proj(z_seq), self.v_proj(z_seq)
        
        def split(tensor):
            r = tensor.real.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            i = tensor.imag.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            return torch.complex(r, i)

        Q, K, V = split(q), split(k), split(v)
        scores_real = (torch.matmul(Q.real, K.real.transpose(-2, -1)) + 
                       torch.matmul(Q.imag, K.imag.transpose(-2, -1))) * self.scale
        
        attn_weights = F.softmax(scores_real, dim=-1)
        attn_weights_c = torch.complex(attn_weights, torch.zeros_like(attn_weights))
        
        out = torch.matmul(attn_weights_c, V)
        out_real = out.real.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out_imag = out.imag.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.o_proj(torch.complex(out_real, out_imag))

# ==========================================
# 5. Differentiable Stack
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
    
    def forward(self, z, memory, ptr, control):
        # Expects 2D inputs [batch, dim] and [batch, 3]
        ptr = ptr.clone()
        push, pop, noop = control[:, 0:1], control[:, 1:2], control[:, 2:3]
        
        ptr_up = torch.roll(ptr, 1, dims=1)
        ptr_down = torch.roll(ptr, -1, dims=1)
        new_ptr = (push * ptr_up) + (pop * ptr_down) + (noop * ptr)
        new_ptr = new_ptr ** 2
        new_ptr = new_ptr / (new_ptr.sum(dim=1, keepdim=True) + CONFIG["eps"])
        
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        write_mask = push * ptr_up
        write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)
        retain_mask = 1.0 - write_mask.unsqueeze(2)
        new_memory = write_val + (memory * retain_mask)
        
        read_mask = new_ptr.unsqueeze(2)
        read_flat = torch.sum(new_memory * read_mask, dim=1)
        read_z = torch.complex(read_flat[:, :self.dim], read_flat[:, self.dim:])
        
        return read_z, new_memory, new_ptr

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 4, dim) 
    def forward(self, z, stack_val):
        combined = torch.cat([z.real, z.imag, stack_val.real, stack_val.imag], dim=-1)
        g = torch.sigmoid(self.gate(combined))
        g_c = torch.complex(g, g) 
        return (1 - g_c) * z + g_c * stack_val

# ==========================================
# 6. Logit-Space Dual VQ
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

    def vq_step(self, z, codebook, cluster_size, embed_avg, adj_matrix, ctx_layer, prev_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        d_sq = torch.sum(z_flat**2, dim=1, keepdim=True) + \
               torch.sum(codebook**2, dim=1) - \
               2 * torch.matmul(z_flat, codebook.t())
        logits = -d_sq
        
        graph_bias = 0
        if prev_idx is not None and prev_idx.numel() > 0:
            idx_safe = prev_idx.long()
            graph_bias = adj_matrix[idx_safe] 
        
        context_logits = ctx_layer(z_flat)
        total_logits = logits + \
                       (CONFIG["graph_bias_scale"] * graph_bias) + \
                       (CONFIG["context_gate_strength"] * context_logits)

        idx = torch.argmax(total_logits, dim=1)
        z_q = F.embedding(idx, codebook)
        
        if prev_idx is not None and self.training:
             with torch.no_grad():
                update = adj_matrix[prev_idx.long(), idx] * 0.99 + 1.0
                adj_matrix[prev_idx.long(), idx] = torch.clamp(update, max=CONFIG["adjacency_clamp"])

        if self.training:
            enc = F.one_hot(idx, codebook.size(0)).float()
            cluster_size.data.mul_(self.decay).add_(enc.sum(0), alpha=1-self.decay)
            embed_sum = torch.matmul(enc.t(), z_flat.detach())
            embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
            n = cluster_size.sum()
            cs = (cluster_size + 1e-6) / (n + codebook.size(0)*1e-6) * n
            codebook.data.copy_(embed_avg / cs.unsqueeze(1))
            
        loss = F.mse_loss(z_q.detach(), z_flat) + CONFIG["commitment_cost"] * F.mse_loss(z_q, z_flat.detach())
        z_q = z_flat + (z_q - z_flat).detach()
        
        ortho_loss = 0
        if self.training and CONFIG["orthogonal_reg_weight"] > 0 and codebook.size(0) <= 512:
            cb_norm = F.normalize(codebook, p=2, dim=1)
            ortho_matrix = torch.matmul(cb_norm, cb_norm.t())
            identity = torch.eye(codebook.size(0), device=ortho_matrix.device)
            ortho_loss = CONFIG["orthogonal_reg_weight"] * F.mse_loss(ortho_matrix, identity)
        
        return torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:]), loss + ortho_loss, idx

    def forward(self, z_fast, z_slow, prev_indices=None):
        prev_syn, prev_sem = (None, None) if prev_indices is None else prev_indices
        zq_syn, loss_syn, idx_syn = self.vq_step(z_fast, self.cb_syn, self.cl_syn, self.avg_syn, self.adj_syn, self.ctx_gate_syn, prev_syn)
        zq_sem, loss_sem, idx_sem = self.vq_step(z_slow, self.cb_sem, self.cl_sem, self.avg_sem, self.adj_sem, self.ctx_gate_sem, prev_sem)
        return zq_syn, zq_sem, loss_syn + loss_sem, (idx_syn, idx_sem)

# ==========================================
# 7. Constraints
# ==========================================
class EnhancedDiversityLoss(nn.Module):
    def __init__(self, n_semantic_symbols, weight=0.1):
        super().__init__()
        self.n_semantic_symbols = n_semantic_symbols
        self.weight = weight
        self.register_buffer("semantic_history", torch.zeros(n_semantic_symbols))
    def forward(self, semantic_indices):
        onehot = F.one_hot(semantic_indices, self.n_semantic_symbols).float()
        self.semantic_history.data.mul_(0.95).add_(onehot.mean(dim=0), alpha=0.05)
        probs = self.semantic_history / (self.semantic_history.sum() + 1e-9)
        return self.weight * (probs * torch.log(probs + 1e-9)).sum()

class EthicalConstraint(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, prev_sem, curr_sem, adjacency):
        if prev_sem is None: return torch.tensor(0.0).to(adjacency.device)
        row_logits = adjacency[prev_sem] 
        graph_probs = F.softmax(row_logits, dim=-1)
        selected_probs = graph_probs.gather(1, curr_sem.unsqueeze(1)).squeeze(1)
        return -torch.log(selected_probs + 1e-9).mean()

# ==========================================
# 8. Adaptive Scheduler
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
            curr_step = epoch - self.warmup_epochs
            cosine_lr = 0.5 * (1 + math.cos(curr_step * math.pi / (self.T_max - self.warmup_epochs)))
            for pg in self.optimizer.param_groups: pg['lr'] = CONFIG["learning_rate"] * cosine_lr

class AdaptiveGradientClip(nn.Module):
    def __init__(self, clip_value=1.0):
        super().__init__()
        self.clip_value = clip_value
        self.loss_history = []
    def forward(self, parameters, current_loss):
        if len(self.loss_history) > 5:
            recent_loss = torch.stack(self.loss_history[-5:]).mean()
            if recent_loss > 1.0: clip_value = self.clip_value * 0.5
            else: clip_value = self.clip_value
        else: clip_value = self.clip_value
        torch.nn.utils.clip_grad_norm_(parameters, clip_value)
        self.loss_history.append(current_loss.detach())

# ==========================================
# 9. Unified Model
# ==========================================
class EnhancedAdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim, dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.attention = HermitianAttention(dim, CONFIG["n_heads"])
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        nn.init.constant_(self.halt_linear.bias, -1.0)

    def forward(self, z):
        # Input z is [batch, dim] (2D)
        z_proc = self.linear(z)
        z_proc = self.norm(z_proc)
        z_proc = self.act(z_proc)
        
        # FIX: Unsqueeze for 3D Attention, then Squeeze back
        z_proc_3d = z_proc.unsqueeze(1) # [batch, 1, dim]
        z_attn = self.attention(z_proc_3d)
        z_proc = z_attn.squeeze(1) # [batch, dim]
        
        z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        stack_probs = F.softmax(self.stack_ctrl(z_flat), dim=-1)
        return z_proc, halt_prob, stack_probs

class EnhancedMultiModalBiCameralCRSN(nn.Module):
    def __init__(self, vocab_size, dim, modalities=2):
        super().__init__()
        self.dim = dim
        self.modalities = modalities
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, dim) for _ in range(modalities)])
        self.modality_projections = nn.ModuleList([ComplexLinear(dim, dim) for _ in range(modalities)])
        self.cross_modal_attention = HermitianAttention(dim * modalities, CONFIG["n_heads"])
        self.cell = EnhancedAdaptiveRecursiveCell(dim)
        self.vq = EnhancedContextAwareDualVQ(dim, CONFIG["n_syntax_symbols"], CONFIG["n_semantic_symbols"], decay=CONFIG["decay"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        self.stack_gate = GatedResidual(dim)
        self.stack = DifferentiableStack(dim, CONFIG["stack_size"])
        self.ethics = EthicalConstraint()
        self.diversity_loss = EnhancedDiversityLoss(CONFIG["n_semantic_symbols"], CONFIG["diversity_weight"])
        self.mask_token = nn.Parameter(torch.randn(dim*2))

    def embed(self, input_ids_list):
        z_list = []
        for i in range(self.modalities):
            emb = self.embeddings[i](input_ids_list[i])
            z_c = torch.complex(emb, torch.zeros_like(emb))
            if self.training and self.modalities > 1:
                noise = torch.randn_like(z_c.real) * 0.05
                z_c = z_c + torch.complex(noise, noise)
            z_c = self.modality_projections[i](z_c)
            z_list.append(z_c)
        z_combined = torch.cat(z_list, dim=-1)
        z_attended = self.cross_modal_attention(z_combined)
        z_out = z_attended[..., :self.dim] + z_attended[..., self.dim:] 
        return z_out

    def forward(self, input_ids_list, hidden=None, mask_mask=None, prev_vq_indices=None):
        batch_size = input_ids_list[0].size(0)
        # 1. Embed (Returns 3D [batch, 1, dim])
        z = self.embed(input_ids_list)
        
        # 2. Masking (Still 3D)
        if mask_mask is not None:
            z_flat = torch.cat([z.real, z.imag], dim=-1)
            # Fix broadcasting for masking: z_flat is [batch, 1, dim*2], mask is [batch, 1]
            mask_expanded = mask_mask.unsqueeze(-1) # [batch, 1, 1]
            mask_vec = self.mask_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            z_masked_flat = z_flat * (1 - mask_expanded) + mask_vec * mask_expanded
            z = torch.complex(z_masked_flat[..., :self.dim], z_masked_flat[..., self.dim:])
            
        # 3. FIX: SQUEEZE to 2D for Recurrence
        z = z.squeeze(1) # [batch, dim]
        
        if hidden is None:
            z_prev = torch.zeros_like(z)
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device)
            stack_ptr[:, 0] = 1.0 
        else:
            z_prev, stack_mem, stack_ptr = hidden
            z = 0.5 * z + 0.5 * z_prev

        remain = torch.ones(batch_size, 1, device=z.device)
        halting_probability = torch.zeros(batch_size, 1, device=z.device)
        z_accum = torch.zeros_like(z)
        ponder_cost = torch.zeros((), device=z.device)
        stack_history = [] 
        
        current_indices = prev_vq_indices
        vq_loss_total = 0
        ethical_loss_total = 0
        
        for t in range(CONFIG["max_recursion_depth"]):
            # CELL INPUT/OUTPUT IS NOW STRICTLY 2D
            z_proc, p_halt, stack_ctrl = self.cell(z)
            stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
            
            positions = torch.arange(CONFIG["stack_size"], device=z.device, dtype=stack_ptr.dtype).unsqueeze(0)
            depth = (stack_ptr * positions).sum(dim=1).mean().detach()
            stack_history.append(depth)

            z_gate = self.stack_gate(z_proc, stack_read)
            zq_syn, zq_sem, vq_loss, new_indices = self.vq(z_proc, z_gate, current_indices)
            vq_loss_total += vq_loss
            
            if current_indices is not None:
                eth_loss = self.ethics(current_indices[1], new_indices[1], self.vq.adj_sem)
                ethical_loss_total += eth_loss
            
            current_indices = new_indices
            z = 0.6 * z_gate + 0.2 * zq_syn + 0.2 * zq_sem
            
            p = p_halt * remain
            if t == CONFIG["max_recursion_depth"] - 1: p = remain
            
            z_accum = z_accum + (p * z)
            halting_probability = halting_probability + p
            remain = remain - p
            ponder_cost += remain.mean()
            
            if remain.max() < CONFIG["eps"]: break
            
        features = torch.cat([z_accum.real, z_accum.imag], dim=-1)
        logits = self.decoder(features)
        next_hidden = (z_accum, stack_mem, stack_ptr)
        avg_stack = torch.stack(stack_history).mean() if stack_history else torch.tensor(0.0)
            
        return logits, next_hidden, current_indices, ponder_cost, vq_loss_total, ethical_loss_total, avg_stack

# ==========================================
# 10. Training Engine
# ==========================================
def train():
    model = EnhancedMultiModalBiCameralCRSN(VOCAB_SIZE, CONFIG["embedding_dim"], modalities=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    
    if CONFIG["adaptive_scheduler"]:
        scheduler = AdaptiveScheduler(opt, T_max=CONFIG["epochs"] - CONFIG["warmup_epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"] - CONFIG["warmup_epochs"])
    
    gradient_clipper = AdaptiveGradientClip(CONFIG["grad_clip"])
    
    num_samples = len(data_tensor) // CONFIG["seq_len"]
    trim = num_samples * CONFIG["seq_len"]
    x_data = data_tensor[:trim].view(num_samples, CONFIG["seq_len"])
    y_data = torch.roll(data_tensor, -1)[:trim].view(num_samples, CONFIG["seq_len"])
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    
    print(f"--- Training SACRSN v39.1 (Stable) ---")
    
    try:
        for epoch in range(CONFIG["epochs"]):
            if epoch < CONFIG["warmup_epochs"]:
                lr_scale = min(1.0, (epoch + 1) / (CONFIG["warmup_epochs"] + 1))
                for pg in opt.param_groups: pg['lr'] = CONFIG["learning_rate"] * lr_scale
            
            for i, (x_batch, y_batch) in enumerate(loader):
                hidden, prev_indices = None, None
                
                if CONFIG["structured_masking"]: mask_mask = create_structured_mask(x_batch, CONFIG["mask_prob"])
                else: mask_mask = (torch.rand_like(x_batch.float()) < CONFIG["mask_prob"]).float().unsqueeze(-1).to(DEVICE)
                
                loss_seq = 0
                ponder_seq = 0
                entropy_sum = 0
                input_feed = [x_batch, x_batch]
                
                for t in range(CONFIG["seq_len"]):
                    yt = y_batch[:, t]
                    mt = mask_mask[:, t] # [batch, 1] effectively
                    xt = [input_feed[0][:, t:t+1], input_feed[1][:, t:t+1]]
                    
                    logits, hidden, curr_indices, ponder, vq_loss, eth_loss, _ = model(xt, hidden, mt, prev_indices)
                    
                    h_z, h_mem, h_ptr = hidden
                    hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach())
                    if curr_indices: prev_indices = (curr_indices[0].detach(), curr_indices[1].detach())
                    
                    loss_pred = F.cross_entropy(logits, yt)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                    entropy_sum += entropy
                    
                    sem_idx = curr_indices[1]
                    loss_diversity = model.diversity_loss(sem_idx)
                    
                    loss_step = loss_pred + \
                                (CONFIG["ponder_penalty"] * ponder) + \
                                vq_loss + \
                                (CONFIG["ethical_weight"] * eth_loss) + \
                                loss_diversity - \
                                (CONFIG["entropy_weight"] * entropy)
                                
                    loss_seq += loss_step
                    ponder_seq += ponder
                
                loss_seq = loss_seq / CONFIG["seq_len"]
                
                opt.zero_grad()
                loss_seq.backward()
                gradient_clipper(model.parameters(), loss_seq)
                opt.step()
                
                if i % 10 == 0:
                     avg_steps = 1.0 + (ponder_seq.item()/CONFIG['seq_len'])
                     ppx = torch.exp(torch.clamp(loss_seq, max=10)).item()
                     ent = entropy_sum.item() / CONFIG['seq_len']
                     lr = opt.param_groups[0]['lr']
                     print(f"Ep {epoch} | Bt {i:03d} | Loss: {loss_seq.item():.3f} | ACT: {avg_steps:.2f} | PPX: {ppx:.0f} | Ent: {ent:.2f}")

            if CONFIG["adaptive_scheduler"] and epoch >= CONFIG["warmup_epochs"]: scheduler.step(epoch)
            elif not CONFIG["adaptive_scheduler"] and epoch >= CONFIG["warmup_epochs"]: scheduler.step()
            
    except KeyboardInterrupt: print("\nInterrupted.")
    return model

def create_structured_mask(input_ids, mask_prob):
    batch_size, seq_len = input_ids.shape
    mask_pattern = torch.zeros(batch_size, seq_len).to(input_ids.device)
    for i in range(batch_size):
        num_masks = int(seq_len * mask_prob)
        if num_masks > 0:
            start_pos = torch.randint(0, seq_len - num_masks + 1, (1,)).item()
            mask_pattern[i, start_pos:start_pos + num_masks] = 1
    if not mask_pattern.any(): mask_pattern[0, 0] = 1
    return mask_pattern.float().unsqueeze(-1)

# ==========================================
# 11. Visualization
# ==========================================
def visualize_all(model):
    print("\n--- Generating Complete Diagnostics ---")
    model.eval()
    
    # 1. Topology
    adj_sem = torch.sigmoid(model.vq.adj_sem).detach().cpu().numpy()
    G = nx.DiGraph()
    for i in range(CONFIG["n_semantic_symbols"]): G.add_node(i)
    
    symbol_to_token = defaultdict(list)
    scan_limit = min(3000, len(data_tensor) - 1)
    hidden, prev_indices = None, None
    with torch.no_grad():
        for i in range(scan_limit):
            x = data_tensor[i].view(1,1)
            _, hidden, curr_indices, _, _, _, _ = model([x,x], hidden, None, prev_indices)
            if curr_indices:
                sem_idx = curr_indices[1].item()
                symbol_to_token[sem_idx].append(tokenizer.decode([data_tensor[i].item()]))
            prev_indices = curr_indices

    node_labels = {}
    for i in range(CONFIG["n_semantic_symbols"]):
        tokens = symbol_to_token.get(i, [])
        if tokens:
            filtered = [t for t in tokens if len(t.strip()) > 1]
            if not filtered: filtered = tokens
            most_common = max(set(filtered), key=filtered.count)
            node_labels[i] = f"{most_common}"
        else: node_labels[i] = str(i)

    for i in range(CONFIG["n_semantic_symbols"]):
        for j in range(CONFIG["n_semantic_symbols"]):
            w = adj_sem[i, j]
            if w > 0.4: G.add_edge(i, j, weight=w)

    plt.figure(figsize=(10, 10))
    try: pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    except: pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='#ff9999', node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', arrowstyle='->', arrowsize=10)
    plt.savefig("1_semantic_topology.png")
    plt.close()

    # 2. Dream & Diagnostics
    start_token = tokenizer.base_tokenizer.special_tokens["<UNK>"]
    x = torch.tensor([[start_token]], device=DEVICE)
    hidden, prev_indices = None, None
    stack_history, act_history, phase_reals, phase_imags, dream_text_list = [], [], [], [], []
    
    with torch.no_grad():
        for _ in range(150):
            logits, hidden, idx, ponder, _, _, s_depth = model([x,x], hidden, None, prev_indices)
            stack_history.append(s_depth.item())
            act_history.append(ponder.item())
            z = hidden[0].cpu().squeeze()
            if z.dim() > 0: 
                phase_reals.append(z.real[0].item())
                phase_imags.append(z.imag[0].item())
            else:
                phase_reals.append(z.real.item())
                phase_imags.append(z.imag.item())
            prev_indices = idx
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            x = next_token
            dream_text_list.append(tokenizer.decode([x.item()]))
            
    print(f"Dream Output: {''.join(dream_text_list)}\n")

    plt.figure(figsize=(12, 4))
    plt.plot(stack_history, color='purple', label='Stack Depth')
    plt.fill_between(range(len(stack_history)), stack_history, color='purple', alpha=0.1)
    plt.savefig("2_stack_mri.png"); plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(act_history)), act_history, color='orange')
    plt.savefig("3_act_profile.png"); plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(phase_reals, phase_imags, c=range(len(phase_reals)), cmap='plasma', alpha=0.5)
    plt.axis('equal'); plt.savefig("4_phase_plot.png"); plt.close()

    # 5. Heatmap
    test_sentence = "First Citizen: Before we proceed any further,"
    input_ids = torch.tensor(tokenizer.encode(test_sentence), dtype=torch.long).to(DEVICE)
    syn_ids, sem_ids, acts = [], [], []
    hidden, prev_indices = None, None
    with torch.no_grad():
        for i in range(len(input_ids)):
            x = input_ids[i].view(1,1)
            _, hidden, idx, ponder, _, _, _ = model([x,x], hidden, None, prev_indices)
            if idx:
                syn_ids.append(idx[0].item())
                sem_ids.append(idx[1].item())
            else: syn_ids.append(0); sem_ids.append(0)
            acts.append(ponder.item())
            prev_indices = idx
            
    def normalize(lst):
        arr = np.array(lst)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    matrix = np.vstack([normalize(syn_ids), normalize(sem_ids), normalize(acts)])
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect='auto', cmap='magma', interpolation='nearest')
    plt.yticks(range(3), ['Syntax', 'Semantics', 'ACT'])
    plt.xticks(range(len(input_ids)), [tokenizer.decode([i.item()]) for i in input_ids], rotation=45)
    plt.tight_layout(); plt.savefig("5_diagnostic_heatmap.png"); plt.close()

    # 6. Logic
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

def dream_mode(model):
    print("\n--- 🌙 Dream Mode ---")
    adj = torch.sigmoid(model.vq.adj_sem).detach().cpu().numpy()
    symbol_to_token = defaultdict(lambda: "?")
    hidden, prev_indices = None, None
    scan_limit = min(3000, len(data_tensor) - 1)
    with torch.no_grad():
        for i in range(scan_limit):
            x = data_tensor[i].view(1,1)
            _, hidden, curr_indices, _, _, _, _ = model([x,x], hidden, None, prev_indices)
            if curr_indices: symbol_to_token[curr_indices[1].item()] = tokenizer.decode([data_tensor[i].item()])
            prev_indices = curr_indices
            
    start_token = tokenizer.base_tokenizer.special_tokens["<UNK>"]
    x = torch.tensor([[start_token]], device=DEVICE)
    _, _, prev_indices, _, _, _, _ = model([x,x], None, None)
    curr_indices = prev_indices[1].item() if prev_indices else 0
    dream_tokens, dream_indices = [tokenizer.decode([start_token])], [curr_indices]
    
    for _ in range(40):
        probs = adj[curr_indices]
        probs[probs < 0.15] = 0 
        if probs.sum() == 0: break
        probs = probs / probs.sum()
        next_indices = np.random.choice(len(probs), p=probs)
        token = symbol_to_token.get(next_indices, "?")
        dream_tokens.append(token)
        dream_indices.append(next_indices)
        curr_indices = next_indices
        
    print(f"Dream: {' '.join(dream_tokens)}\n")
    dream_tensor = torch.tensor([tokenizer.encode(token)[0] if token != "?" else 0 for token in dream_tokens], dtype=torch.long).to(DEVICE)
    topo, stack, act = [], [], []
    x0 = dream_tensor[0].view(1,1)
    with torch.no_grad():
        _, hidden, prev_indices, _, _, _, _ = model([x0,x0], None, None)
        for i in range(1, len(dream_tensor)):
            x = dream_tensor[i-1].view(1,1)
            _, hidden, curr_indices, ponder, _, eth_loss, stack_d = model([x,x], hidden, None, prev_indices)
            topo.append(eth_loss.item())
            stack.append(stack_d.item())
            act.append(ponder.item())
            prev_indices = curr_indices

    def normalize(lst):
        arr = np.array(lst)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) if arr.max() > arr.min() else arr

    matrix = np.vstack([normalize(topo), normalize(stack), normalize(act)])
    plt.figure(figsize=(12, 5))
    plt.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.yticks(range(3), ['Consistency', 'Stack', 'Effort'])
    plt.xticks(range(len(dream_tokens)-1), dream_tokens[1:], rotation=45, ha='right')
    for i in range(len(dream_tokens)-1):
        sym_id = dream_indices[i+1]
        for y in range(3): plt.text(i, y, f"{sym_id}", color='white', ha='center', va='center', fontsize=7)
    plt.tight_layout(); plt.savefig("6_dream_diagnostic.png"); plt.close()

if __name__ == "__main__":
    FILENAME = "sacrsn_v39_1_stable.pth"
    trained_model = train()
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save(trained_model.state_dict(), FILENAME)
    visualize_all(trained_model)
    dream_mode(trained_model)
