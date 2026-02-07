"""
Project: CRSN v55 (Complex Recurrent Sequence Network)
Description: 
    Complex-Valued Recursive RNN with Global Bottleneck & Experience Replay.
    
    MAINTENANCE NOTES: 
    - Trainer logic includes 8-item hidden state unpacking fix.
    - Configuration attributes synchronized.
    - Input shape guards active.
      
    Status: STABLE RELEASE.
"""

import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque
from typing import Tuple, List, Optional, Dict, Union

# --- Hardware Optimization ---
NUM_CORES = os.cpu_count()
torch.set_num_threads(NUM_CORES if NUM_CORES else 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Runtime Environment: {DEVICE.upper()} | Threads: {torch.get_num_threads()}")

# Optional: Plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    """System Hyperparameters."""
    # Architecture
    SEQ_LEN: int = 64
    DIM: int = 128
    VOCAB_SIZE: int = 128
    STACK_SIZE: int = 16
    MAX_RECURSION_DEPTH: int = 8
    BOTTLENECK_DIM: int = 32
    HALT_THRESHOLD: float = 0.99
    
    # Memory & Replay
    REPLAY_BUFFER_CAPACITY: int = 200000 
    REPLAY_INTERVAL: int = 50
    REPLAY_STEPS: int = 20
    HIERARCHY_DEPTH: int = 2
    
    # Adaptive Logic
    UPDATE_WINDOW: int = 8
    ENTROPY_THRESHOLD: float = 0.3
    STABILITY_LIMIT: float = 0.9
    UPDATE_INTERVAL: int = 5
    
    # Training
    EPOCHS: int = 1000
    LEARNING_RATE: float = 1e-3
    GRAD_CLIP: float = 1.0
    EPSILON: float = 1e-6
    TEMPERATURE: float = 1.0
    RESIDUAL_WEIGHT: float = 0.1
    
    # Loss Coefficients
    LAMBDA_NLL: float = 1.0
    LAMBDA_PONDER: float = 0.01      # Previously L_HALT
    LAMBDA_L2_REG: float = 0.01      # Previously L_ENERGY
    LAMBDA_KL: float = 0.05
    LAMBDA_ENTROPY: float = 0.15     # Previously L_DIVERSITY
    LAMBDA_ALIGNMENT: float = 0.1    # Previously L_GROUND
    LAMBDA_TRANSITION: float = 0.1   # Previously L_META
    LAMBDA_AUX_CLS: float = 0.1
    LAMBDA_AUX_RECON: float = 0.1
    LAMBDA_DRIFT: float = 0.05
    LAMBDA_PROBE: float = 0.1
    LAMBDA_SHADOW: float = 0.1       # Previously L_TOM
    LAMBDA_CONSTRAINT: float = 0.005 # Previously L_ETHICS
    LAMBDA_EMBED_CONSISTENCY: float = 0.01
    LAMBDA_PHASE_CONSISTENCY: float = 0.05
    
    # Annealing & Resource Management
    KL_CYCLE_LEN: int = 250
    KL_MAX_BETA: float = 0.05
    RESOURCE_CAPACITY: float = 100.0
    RESOURCE_DECAY: float = 0.0
    RESOURCE_RECOVERY: float = 1.0
    ADAPTIVE_ALPHA: float = 2.0
    BASE_HALT_TEMP: float = 1.0
    
    # Constants
    ENTROPY_DELTA_LIMIT: float = 0.5
    PROJECTION_DIM: int = 32

cfg = Config()


# ==========================================
# 2. Data Manager
# ==========================================
class TextDataLoader:
    def __init__(self):
        self.raw_text = """
The neural architecture of the mind is a mirror of the cosmos itself. 
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
""" * 100
        
        self.chars = sorted(list(set(self.raw_text)))
        self.vocab_size = len(self.chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data_tensor = torch.tensor([self.char_to_ix[c] for c in self.raw_text], dtype=torch.long).to(DEVICE)
        print(f"[Data] Initialized: {len(self.raw_text)} characters | Vocab: {self.vocab_size}")

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data_tensor) - cfg.SEQ_LEN, (1,))
        x = self.data_tensor[ix : ix + cfg.SEQ_LEN].view(1, cfg.SEQ_LEN)
        y = self.data_tensor[ix+1 : ix + cfg.SEQ_LEN + 1].view(1, cfg.SEQ_LEN)
        return x, y

data_loader = TextDataLoader()


# ==========================================
# 3. Core Layers (Complex-Valued)
# ==========================================
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z) + cfg.EPSILON
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, keepdim=True)
        norm_mag = (mag - mean) / torch.sqrt(var + cfg.EPSILON)
        norm_mag = norm_mag * self.scale + self.shift
        phase = torch.angle(z)
        return torch.complex(norm_mag * torch.cos(phase), norm_mag * torch.sin(phase))

class ModReLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        norm = torch.abs(z) + cfg.EPSILON
        scale = F.relu(norm + self.bias) / norm
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.fc_real.weight)
        nn.init.orthogonal_(self.fc_imag.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            self.fc_real(z.real) - self.fc_imag(z.imag),
            self.fc_real(z.imag) + self.fc_imag(z.real)
        )

class ContextAwareAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = ComplexLinear(dim)
        self.k_proj = ComplexLinear(dim)
        self.v_proj = ComplexLinear(dim)
        self.context_proj = nn.Linear(dim * 2, dim * 2) 
        self.scale = dim ** -0.5
    
    def forward(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = self.q_proj(z), self.k_proj(z), self.v_proj(z)
        q_flat = torch.cat([q.real, q.imag], dim=-1)
        if context is not None:
            context_weight = torch.sigmoid(self.context_proj(context))
            q_flat = q_flat * context_weight
        k_flat = torch.cat([k.real, k.imag], dim=-1)
        attn_scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        v_real = torch.matmul(attn_weights, v.real)
        v_imag = torch.matmul(attn_weights, v.imag)
        return torch.complex(v_real, v_imag)

class ComplexAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = ComplexLinear(dim)
        self.k_proj = ComplexLinear(dim)
        self.v_proj = ComplexLinear(dim)
        self.scale = dim ** -0.5
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(z), self.k_proj(z), self.v_proj(z)
        q_flat = torch.cat([q.real, q.imag], dim=-1)
        k_flat = torch.cat([k.real, k.imag], dim=-1)
        attn_scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        v_real = torch.matmul(attn_weights, v.real)
        v_imag = torch.matmul(attn_weights, v.imag)
        return torch.complex(v_real, v_imag)


# ==========================================
# 4. Functional Modules
# ==========================================

class ExperienceReplayBuffer:
    """Stores high-loss 'surprise' events for offline replay."""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, x: torch.Tensor, y: torch.Tensor, loss: float):
        if loss > 2.5: # Error threshold
            self.buffer.append((x.detach().cpu(), y.detach().cpu()))
            
    def sample(self, batch_size: int = 32) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if len(self.buffer) < batch_size: return None, None
        batch = random.sample(self.buffer, batch_size)
        x_stack = torch.stack([item[0] for item in batch]).squeeze(1).to(DEVICE)
        y_stack = torch.stack([item[1] for item in batch]).squeeze(1).to(DEVICE)
        return x_stack, y_stack

class AdaptiveLossScaler(nn.Module):
    """Simulates dynamic gain control based on error and entropy."""
    def __init__(self):
        super().__init__()
        self.register_buffer("avg_loss", torch.tensor(5.0))
        self.register_buffer("avg_entropy", torch.tensor(1.0))
        
    def forward(self, current_loss: torch.Tensor, current_entropy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.avg_loss = 0.95 * self.avg_loss + 0.05 * current_loss.detach()
        self.avg_entropy = 0.95 * self.avg_entropy + 0.05 * current_entropy.detach()
        # Magnitude of prediction error (roughly 'Dopamine' analogue)
        error_magnitude = torch.clamp(self.avg_loss - current_loss.detach(), 0.0, 1.0)
        # Entropy gradient (roughly 'Noradrenaline' analogue)
        entropy_gradient = torch.sigmoid((current_entropy.detach() - self.avg_entropy) * 2.0)
        return error_magnitude, entropy_gradient

class GlobalBottleneck(nn.Module):
    """Bottleneck layer for global information broadcast."""
    def __init__(self, dim: int, bottleneck_dim: int):
        super().__init__()
        self.dim = dim
        self.write = nn.Linear(dim * 2, bottleneck_dim)
        self.broadcast = nn.Linear(bottleneck_dim, dim * 2)
        self.norm = nn.LayerNorm(dim * 2)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([z.real, z.imag], dim=-1)
        bottleneck_state = torch.tanh(self.write(flat))
        broadcast_signal = self.broadcast(bottleneck_state)
        combined = self.norm(flat + broadcast_signal)
        return torch.complex(combined[..., :self.dim], combined[..., self.dim:])

class ResourceMonitor(nn.Module):
    """Simulates computational resource constraints."""
    def __init__(self, capacity: float):
        super().__init__()
        self.capacity = capacity
        self.register_buffer("current_resource", torch.tensor(capacity))
        self.decay = cfg.RESOURCE_DECAY
        self.recovery = cfg.RESOURCE_RECOVERY
        
    def forward(self, load: torch.Tensor) -> torch.Tensor:
        self.current_resource = self.current_resource - (load * self.decay)
        self.current_resource = torch.clamp(self.current_resource + self.recovery, 0, self.capacity)
        return self.current_resource / self.capacity

class StateStatisticsTracker:
    def __init__(self, n_symbols):
        self.stats = {s: {"count": 0, "avg_loss": 0.0, "avg_ponder": 0.0, "stability": 0.0} for s in range(n_symbols)}
    def update(self, sym, loss, ponder, changed):
        if isinstance(sym, torch.Tensor): sym = sym.item()
        d = self.stats[sym]
        d["count"] += 1
        d["avg_loss"] = 0.9 * d["avg_loss"] + 0.1 * loss
        d["avg_ponder"] = 0.9 * d["avg_ponder"] + 0.1 * ponder
        d["stability"] = 0.9 * d["stability"] + 0.1 * (0 if changed else 1)

class QualityControlGate:
    def __init__(self, stability_limit):
        self.stability_limit = stability_limit
        self.loss_limit = 2.0
    def allow_update(self, stats):
        if stats["stability"] > self.stability_limit: return False
        if stats["avg_loss"] > self.loss_limit: return False
        return True

class GraphTopologyUpdater(nn.Module):
    def __init__(self, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols
        self.symbol_history = []
        self.node_labels = defaultdict(int)
    def update_history(self, sym_idx):
        self.symbol_history.append(sym_idx.detach())
        if len(self.symbol_history) > cfg.UPDATE_WINDOW: self.symbol_history.pop(0)
    def calculate_entropy(self):
        if len(self.symbol_history) < cfg.UPDATE_WINDOW: return None
        syms = torch.stack(self.symbol_history)
        probs = torch.bincount(syms.flatten(), minlength=self.n_symbols).float()
        probs = probs / probs.sum()
        return -(probs * torch.log(probs + 1e-8)).sum()
    def determine_update_type(self, loss, ponder):
        entropy = self.calculate_entropy()
        if entropy is None: return None
        if entropy < cfg.ENTROPY_THRESHOLD and ponder > 1.5: return "reassign_cluster"
        if entropy < cfg.ENTROPY_THRESHOLD and loss > 0.5: return "adjust_weights"
        return None
    def reassign_cluster(self, z_flat, codebook, adjacency, current_sym):
        dists = torch.sum((codebook - z_flat)**2, dim=-1)
        candidates = torch.topk(-dists, k=4).indices
        best = current_sym; best_score = dists[current_sym]
        for s in candidates:
            score = dists[s] + 0.1 * adjacency[current_sym, s]
            if score < best_score: best_score = score; best = s
        return best
    def adjust_weights(self, adjacency, current_sym):
        with torch.no_grad():
            adjacency[current_sym] *= 0.9
            adjacency[current_sym] += 0.1 * torch.randn_like(adjacency[current_sym])
    def track_node(self, sym_idx):
        key = sym_idx.item()
        self.node_labels[key] += 1
        return key

class TransitionConstraint(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, prev_sym, curr_sym, adjacency):
        if prev_sym is None: return torch.tensor(0.0).to(adjacency.device)
        row_logits = -adjacency[prev_sym] 
        return F.cross_entropy(row_logits.view(-1, cfg.VOCAB_SIZE), curr_sym.view(-1))

class LatentStateProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.probe = nn.Linear(dim * 2, 1)
    def forward(self, z):
        flat = torch.cat([z.real, z.imag], dim=-1)
        return self.probe(flat)

class VarianceScaler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("running_var", torch.ones(1))
    def forward(self, z):
        mag = torch.abs(z)
        batch_var = mag.var()
        if self.training:
            self.running_var = 0.9 * self.running_var + 0.1 * batch_var.detach()
        return F.softplus(batch_var / (self.running_var + cfg.EPSILON))

class UpdateRateLimiter:
    def __init__(self, min_interval=5):
        self.min_interval = min_interval
        self.last_update_step = 0
    def allow(self, step):
        if step - self.last_update_step >= self.min_interval:
            self.last_update_step = step
            return True
        return False

# ==========================================
# 5. Core Neural Components
# ==========================================

class VariationalAutoEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lstm = nn.LSTMCell(dim * 2, dim) 
        self.fc = nn.Linear(dim * 2, dim * 2)
        self.register_buffer("prior_mu", torch.zeros(dim))
        self.meta_h_prior = torch.zeros(dim)
    
    def forward(self, z, h_prev, c_prev):
        z_flat = torch.cat([z.real, z.imag], dim=-1) if torch.is_complex(z) else z
        _, c_next = self.lstm(z_flat, (h_prev, c_prev))
        
        params = self.fc(z_flat)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        h_post = mu + torch.randn_like(std) * std
        
        kl = 0.5 * ((mu - self.prior_mu).pow(2) + logvar.exp() - 1 - logvar).sum(dim=1).mean()
        return h_post, c_next, kl, mu

    def update_prior(self, batch_mu):
        with torch.no_grad():
            self.prior_mu = 0.9 * self.prior_mu + 0.1 * batch_mu.mean(0)
            
    def update_meta_prior(self, batch_mu):
         with torch.no_grad():
             self.meta_h_prior = 0.9 * self.meta_h_prior + 0.1 * batch_mu.mean(0)

class HierarchicalVAE(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.depth = depth
        self.levels = nn.ModuleList()
        for i in range(depth):
            level_dim = dim // (2 ** i)
            if level_dim < 8: level_dim = 8
            self.levels.append(VariationalAutoEncoder(level_dim))
    
    def forward(self, z, h_prev, c_prev):
        h_list, c_list, mu_list = [], [], []
        kl_sum = 0
        h0, c0, kl0, mu0 = self.levels[0](z, h_prev[0], c_prev[0])
        h_list.append(h0); c_list.append(c0); mu_list.append(mu0); kl_sum += kl0
        
        prev_h = h0
        for i in range(1, self.depth):
            h, c, kl, mu = self.levels[i](prev_h, h_prev[i], c_prev[i])
            h_list.append(h); c_list.append(c); mu_list.append(mu); kl_sum += kl
            prev_h = h
        return tuple(h_list), tuple(c_list), kl_sum, tuple(mu_list)

    def update_priors(self, mus):
        for i, mu in enumerate(mus):
            self.levels[i].update_meta_prior(mu)

class GraphVectorQuantizer(nn.Module):
    def __init__(self, dim, n_symbols):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_symbols, dim*2) * 0.2)
        self.transition_matrix = nn.Parameter(torch.randn(n_symbols, n_symbols)) 
        self.embedding_anchors = nn.Parameter(torch.randn(n_symbols, cfg.PROJECTION_DIM))
        self.anchor_proj = nn.Linear(dim*2, cfg.PROJECTION_DIM)
        self.meta_transition = nn.Parameter(torch.randn(n_symbols, n_symbols))
        self.variance_scaler = VarianceScaler(dim)
        
    def forward(self, z, prev_symbol_dist=None, prev_entropy=None, temp=1.0):
        flat = torch.cat([z.real, z.imag], dim=-1)
        d = torch.sum(flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(flat, self.codebook.t())
        
        up_scalar = self.variance_scaler(z)
        if prev_symbol_dist is not None:
             graph_bias = torch.matmul(prev_symbol_dist, self.transition_matrix)
             d_total = d - (cfg.LAMBDA_L2_REG * up_scalar * torch.sigmoid(graph_bias))
        else:
             d_total = d
             
        logits = -d_total
        probs = F.softmax(logits / temp, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        
        transition_loss = torch.tensor(0.0).to(z.device)
        if prev_entropy is not None and prev_symbol_dist is not None:
             delta_H = torch.abs(entropy - prev_entropy)
             meta_gate = torch.sigmoid((delta_H - cfg.ENTROPY_DELTA_LIMIT) * 10.0)
             meta_bias = torch.matmul(prev_symbol_dist, self.meta_transition)
             transition_loss = meta_gate.mean() * (d - torch.sigmoid(meta_bias)).mean()

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, hard=False) if self.training else probs
        hard_idx = torch.argmax(soft_one_hot, dim=-1)
        
        z_q_flat = torch.matmul(soft_one_hot, self.codebook)
        z_q = torch.complex(z_q_flat[..., :z.shape[-1]], z_q_flat[..., z.shape[-1]:])
        
        proj = self.anchor_proj(flat)
        anchor = torch.matmul(soft_one_hot, self.embedding_anchors)
        alignment_loss = F.mse_loss(proj, anchor)
        
        return z_q, soft_one_hot, hard_idx, d.mean(), entropy, alignment_loss, transition_loss

class ShadowConsistencyModel(nn.Module):
    def __init__(self, dim, n_symbols):
        super().__init__()
        self.shadow_graph = GraphVectorQuantizer(dim, n_symbols)
    def forward(self, z, prev_sym_dist, temp=1.0):
        z_q, soft_sym, _, energy, _, _, _ = self.shadow_graph(
            z, prev_symbol_dist=prev_sym_dist, temp=temp
        )
        return z_q, energy

class DifferentiableStackMemory(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
        self.memory_attention = ComplexAttention(dim)
        self.compress_proj = nn.Linear(dim*2, dim)
    
    def forward(self, z, mem, ptr, ctrl):
        # [SHAPE FIX] Force 2D for broadcasting
        if ctrl.ndim == 3: ctrl = ctrl.squeeze(1)
            
        push = torch.sigmoid(ctrl[:, 0:1])
        pop  = torch.sigmoid(ctrl[:, 1:2])
        stay = torch.sigmoid(ctrl[:, 2:3])
        total = push + pop + stay + cfg.EPSILON
        push, pop, stay = push/total, pop/total, stay/total
        
        ptr_up = torch.roll(ptr, 1, dims=1)
        ptr_down = torch.roll(ptr, -1, dims=1)
        
        # [FIX] Safe pointer calculation
        new_ptr = (push * ptr_up) + (pop * ptr_down) + (stay * ptr)
        
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        # mem is (Batch, 16, Dim). push.unsqueeze(-1) is (Batch, 1, 1).
        mem = mem * (1 - push.unsqueeze(-1)) + push.unsqueeze(-1) * z_flat.unsqueeze(1)
        
        mem_complex = torch.complex(mem[..., :self.dim], mem[..., self.dim:])
        attended_mem = self.memory_attention(mem_complex)
        attended_mem_flat = torch.cat([attended_mem.real, attended_mem.imag], dim=-1)
        
        read_flat = (attended_mem_flat * new_ptr.unsqueeze(-1)).sum(1)
        z_read = torch.complex(read_flat[:, :self.dim], read_flat[:, self.dim:])
        active_slots = (new_ptr > 0.1).float().sum(1).mean()
        return z_read, mem, new_ptr, active_slots

class ComplexModulusProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection_layer = nn.Linear(dim*2, dim)
    def forward(self, z):
        flat = torch.cat([z.real, z.imag], dim=-1)
        logits = self.projection_layer(flat)
        return logits

# ==========================================
# 6. Main Processing Unit
# ==========================================
class RecursiveRecurrentUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.attention = ContextAwareAttention(dim)
        self.exit_prob_layer = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        self.global_bottleneck = GlobalBottleneck(dim, cfg.BOTTLENECK_DIM)
        nn.init.constant_(self.exit_prob_layer.bias, 2.0)
        
    def forward(self, z):
        z_in = z
        z_proc = self.act(self.norm(self.linear(z)))
        z_proc = self.attention(z_proc, context=torch.cat([z.real, z.imag], dim=-1))
        
        # Global Information Bottleneck Interaction
        z_proc = self.global_bottleneck(z_proc)
        
        # Residual
        z_proc = z_proc + cfg.RESIDUAL_WEIGHT * z_in
        
        z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
        halt_logit = self.exit_prob_layer(z_flat)
        stack_logits = self.stack_ctrl(z_flat)
        return z_proc, halt_logit, stack_logits

# ==========================================
# 7. ComplexRecurrentSequenceNetwork (The Model)
# ==========================================
class ComplexRecurrentSequenceNetwork(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        
        self.image_encoder = nn.Linear(256, dim)
        self.audio_encoder = nn.Linear(128, dim)
        
        self.cell = RecursiveRecurrentUnit(dim)
        self.vq = GraphVectorQuantizer(dim, cfg.VOCAB_SIZE)
        self.latent_vae = HierarchicalVAE(dim, cfg.HIERARCHY_DEPTH)
        self.stack = DifferentiableStackMemory(dim, cfg.STACK_SIZE)
        self.proj = ComplexModulusProjection(dim)
        self.decoder = nn.Linear(dim, vocab_size)
        
        self.aux_classifier = nn.Linear(dim*2, 10) 
        self.aux_reconstruction = nn.Linear(dim*2, dim*2)
        
        self.latent_probe = LatentStateProbe(dim)
        self.register_buffer("prev_sym_soft", torch.zeros(cfg.VOCAB_SIZE))
        
        # Subsystems
        self.replay_buffer = ExperienceReplayBuffer(cfg.REPLAY_BUFFER_CAPACITY)
        self.adaptive_scaler = AdaptiveLossScaler()
        self.resource_monitor = ResourceMonitor(cfg.RESOURCE_CAPACITY)
        self.shadow_model = ShadowConsistencyModel(dim, cfg.VOCAB_SIZE) 
        self.topology_updater = GraphTopologyUpdater(cfg.VOCAB_SIZE)
        self.stats_tracker = StateStatisticsTracker(cfg.VOCAB_SIZE)
        self.quality_gate = QualityControlGate(cfg.STABILITY_LIMIT)
        self.rate_limiter = UpdateRateLimiter(cfg.UPDATE_INTERVAL)
        self.transition_constraint = TransitionConstraint()

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, x, hidden=None, training=True, modulation=None, global_step=0):
        # [FIX] Reshape input to ensure it is (Batch, 1) or (Batch) before embedding
        z = self.embed(x).squeeze(1)
        
        batch_size = x.size(0)
        z_initial = z.clone()
        
        # Init Hidden
        if hidden is None:
            z_prev = torch.zeros_like(z)
            h_vae = tuple([torch.zeros(batch_size, self.dim // (2**i)).to(DEVICE) if self.dim // (2**i) >= 8 else torch.zeros(batch_size, 8).to(DEVICE) for i in range(cfg.HIERARCHY_DEPTH)])
            c_vae = tuple([torch.zeros(batch_size, self.dim // (2**i)).to(DEVICE) if self.dim // (2**i) >= 8 else torch.zeros(batch_size, 8).to(DEVICE) for i in range(cfg.HIERARCHY_DEPTH)])
            stack_mem = torch.zeros(batch_size, cfg.STACK_SIZE, self.dim*2, device=z.device)
            stack_ptr = torch.zeros(batch_size, cfg.STACK_SIZE, device=z.device); stack_ptr[:, 0] = 1.0
            prev_soft_sym = None; prev_entropy = None; prev_sym_idx = None
        else:
            z_prev, h_vae, c_vae, stack_mem, stack_ptr, prev_soft_sym, prev_entropy, prev_sym_idx = hidden
            z = 0.5 * z + 0.5 * z_prev

        # Metrics Container
        m = defaultdict(float)
        mus_list = []
        mem_efficiency_log = []
        
        halting_probability = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        z_weighted = torch.zeros_like(z)
        
        current_soft_sym = prev_soft_sym
        current_entropy = prev_entropy
        current_hard_sym = prev_sym_idx
        stack_depth_log = torch.tensor(0.0).to(z.device)
        
        resource_usage = self.resource_monitor(torch.tensor(1.0).to(DEVICE))
        
        # Adaptive Modulation
        temp_mod = 1.0
        if modulation:
             _, entropy_grad = modulation
             temp_mod = 1.0 / (1.0 + entropy_grad)
        
        # Recursive Loop
        for t in range(cfg.MAX_RECURSION_DEPTH):
            # 1. Recurse
            z_proc, halt_logit, stack_logits = self.cell(z)
            
            # [SHAPE FIX] Flatten stack_logits to (Batch, 3) immediately
            if stack_logits.dim() > 2:
                stack_logits = stack_logits.view(-1, 3)

            # 2. Stack
            z_stack, stack_mem, stack_ptr, active_slots = self.stack(z_proc, stack_mem, stack_ptr, stack_logits)
            mem_efficiency_log.append(active_slots)
            z = z_proc + z_stack
            
            # 3. Variational Inference
            h_vae, c_vae, kl, mus = self.latent_vae(z, h_vae, c_vae)
            mus_list.append(mus)
            
            # 4. Quantize
            curr_temp = cfg.TEMPERATURE * temp_mod
            z_q, soft_sym, hard_idx, energy, ent, ground, meta = self.vq(
                z, current_soft_sym, prev_entropy=current_entropy, temp=curr_temp
            )
            
            # 5. Shadow Model Consistency
            _, shadow_energy = self.shadow_model(z, current_soft_sym, temp=curr_temp)
            
            # 6. Constraint Check
            constraint_loss = self.transition_constraint(current_hard_sym, hard_idx, self.vq.transition_matrix)
            
            if training:
                sym_item = hard_idx[0].item()
                self.topology_updater.update_history(hard_idx)
                temp_act = cfg.BASE_HALT_TEMP * (0.9 ** t)
                p_halt = torch.sigmoid(halt_logit / temp_act)
                
                update_type = self.topology_updater.determine_update_type(loss=energy.item(), ponder=p_halt.mean().item())
                if update_type:
                    stats = self.stats_tracker.stats[sym_item]
                    if self.quality_gate.allow_update(stats) and self.rate_limiter.allow(global_step):
                         if update_type == "reassign_cluster":
                             z_flat = torch.cat([z.real, z.imag], dim=-1).detach()
                             hard_idx = self.topology_updater.reassign_cluster(z_flat, self.vq.codebook.detach(), self.vq.transition_matrix.detach(), hard_idx)
                         elif update_type == "adjust_weights":
                             self.topology_updater.adjust_weights(self.vq.transition_matrix.data, hard_idx)
                             
                changed = (sym_item != current_hard_sym) if current_hard_sym is not None else False
                self.stats_tracker.update(sym_item, energy.item(), p_halt.mean().item(), changed)
                self.topology_updater.track_node(hard_idx)
            
            current_hard_sym = hard_idx
            current_soft_sym = soft_sym
            current_entropy = ent
            z = 0.7 * z + 0.3 * z_q
            
            # 7. Halting
            temp_act = cfg.BASE_HALT_TEMP * (0.9 ** t)
            p_halt = torch.sigmoid(halt_logit / temp_act)
            p = torch.minimum(remain, p_halt)
            if t == cfg.MAX_RECURSION_DEPTH - 1: p = remain
            
            z_weighted = z_weighted + (p * z)
            remain = remain - p
            
            drift = torch.mean((z.real - z_initial.real)**2)
            
            # Accumulate Metrics
            m['halt'] += p * (t+1)
            m['l2_reg'] += energy
            m['kl'] += kl
            m['alignment'] += ground
            m['transition'] += meta
            m['drift'] += drift
            m['shadow'] += shadow_energy
            m['constraint'] += constraint_loss
            m['probe'] += self.latent_probe(z)
            stack_depth_log = torch.sum(stack_ptr * torch.arange(cfg.STACK_SIZE, device=z.device), dim=1)

        # Output
        logits = self.decoder(self.proj(z_weighted))
        
        # Aux Outputs
        flat = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        aux_c = self.aux_classifier(flat)
        aux_r = self.aux_reconstruction(flat)
        
        if current_soft_sym is not None:
            with torch.no_grad():
                self.prev_sym_soft = self.prev_sym_soft * 0.9 + current_soft_sym.mean(0) * 0.1
        
        next_hidden = (z_weighted, h_vae, c_vae, stack_mem, stack_ptr, current_soft_sym, current_entropy, current_hard_sym)
        
        # Average Mus for hierarchical VAE
        # Note: mus_list contains tuples of mus from each step. We need to aggregate them properly.
        # Simplify for return to match unpack count: Just return the last step's mus or average across steps
        # For simplicity and shape consistency with original:
        if len(mus_list) > 0:
            avg_mus = tuple(torch.stack([m[i] for m in mus_list]).mean(0) for i in range(len(mus_list[0])))
        else:
            avg_mus = ()

        avg_mem_efficiency = sum(mem_efficiency_log) / len(mem_efficiency_log)
        
        # Return 17 items (Standardized Protocol)
        return (logits, next_hidden, m['halt'], m['l2_reg'], m['kl'], m['alignment'], 
                m['transition'], m['drift'], m['probe'], avg_mus, 
                stack_depth_log, aux_c, aux_r, avg_mem_efficiency, m['shadow'], resource_usage, m['constraint'])

# ==========================================
# 8. Training Routine
# ==========================================
class AdaptiveOptimizer:
    def __init__(self, model, base_lr=1e-3):
        self.opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
        self.base_lr = base_lr
        self.kl_history = []
        
    def step(self, loss, kl_value, error_magnitude=0.0):
        if math.isnan(kl_value) or math.isinf(kl_value): kl_value = 100.0
        self.kl_history.append(kl_value)
        boost = 1.0 + (error_magnitude * 0.5) 
        if len(self.kl_history) > 5:
            recent = np.array(self.kl_history[-5:])
            volatility = np.std(recent)
            if volatility > 0.5: lr = self.base_lr * 0.1
            elif volatility > 0.1: lr = self.base_lr * 0.5
            else: lr = self.base_lr
        else: lr = self.base_lr
        final_lr = lr * boost
        for param_group in self.opt.param_groups: param_group['lr'] = final_lr
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.opt.param_groups[0]['params'], cfg.GRAD_CLIP)
        self.opt.step()
        return final_lr

class Trainer:
    def __init__(self, model):
        self.model = model
        self.opt = AdaptiveOptimizer(model, base_lr=cfg.LEARNING_RATE)

    def train_epoch(self, epoch_idx):
        hidden = None
        metrics = defaultdict(float)
        last_error, last_ent_grad = 0.0, 0.0
        
        # --- 1. OFFLINE EXPERIENCE REPLAY ---
        if epoch_idx > 0 and epoch_idx % cfg.REPLAY_INTERVAL == 0:
            mem_count = len(self.model.replay_buffer.buffer)
            dynamic_steps = int((mem_count / 32) * 0.10) 
            actual_steps = max(cfg.REPLAY_STEPS, dynamic_steps)
            
            print(f"\n   [...] Executing Offline Replay (Consolidating {actual_steps} batches from {mem_count} samples)...")
            self.model.train()
            for i in range(actual_steps):
                rx, ry = self.model.replay_buffer.sample()
                if rx is not None:
                    r_logits, _, *r_rest = self.model(rx, None, training=True)
                    r_loss = F.cross_entropy(r_logits, ry.view(-1))
                    # Manual opt step for replay
                    self.opt.opt.zero_grad()
                    r_loss.backward()
                    self.opt.opt.step()
                if i % 100 == 0:
                     print(f"         Processing... {i}/{actual_steps}", end='\r')
            print(f"         Replay Complete.                               ")
        
        # 2. Online Training Cycle
        batch_kl = 0.0 
        
        for _ in range(10):
            x_seq, y_seq = data_loader.get_batch()
            factors = (last_error, last_ent_grad)
            hidden = None # Reset hidden for new batch/sequence
            loss_seq = 0
            
            # [CRITICAL FIX] Time-step loop restored
            for t in range(cfg.SEQ_LEN):
                # Slice time step
                x = x_seq[:, t].view(1, 1)
                y = y_seq[:, t].view(1)
            
                # Forward
                (logits, hidden, halt, l2_reg, kl, alignment, transition, drift, probe, mus, 
                 stack_depth, aux_c, aux_r, mem_eff, shadow, resource, constraint) = self.model(
                     x, hidden, global_step=epoch_idx, training=True, modulation=factors
                )
                
                # Detach
                # [CRITICAL FIX] Unpack the full 8-item hidden state
                z_h, v_h, v_c, mem, ptr, sym, ent, idx = hidden
                v_h = tuple(t.detach() for t in v_h)
                v_c = tuple(t.detach() for t in v_c)
                
                # [CRITICAL FIX] Repack all 8 items for next step
                hidden = (z_h.detach(), v_h, v_c, mem.detach(), ptr.detach(), 
                          sym.detach() if sym is not None else None,
                          ent.detach() if ent is not None else None, idx)
                
                self.model.latent_vae.update_priors(mus)
                nll = F.cross_entropy(logits, y.view(-1))
                
                # Annealing
                cyc = (epoch_idx % cfg.KL_CYCLE_LEN) / cfg.KL_CYCLE_LEN
                beta = cfg.KL_MAX_BETA * (1.0 if cyc > 0.5 else cyc * 2)
                
                loss_step = (cfg.LAMBDA_NLL * nll) + \
                       (cfg.LAMBDA_PONDER * halt) + \
                       (cfg.LAMBDA_L2_REG * l2_reg) + \
                       (beta * kl) + \
                       (cfg.LAMBDA_CONSTRAINT * constraint) + \
                       (cfg.LAMBDA_SHADOW * shadow)
                
                loss_seq += loss_step
                
                # Update State
                self.model.replay_buffer.push(x, y, loss_step.item())
                err_mag, ent_grad = self.model.adaptive_scaler(loss_step, torch.tensor(0.5).to(DEVICE))
                last_error, last_ent_grad = err_mag.item(), ent_grad.item()
                
                # Logs
                metrics['loss'] += loss_step.item()
                metrics['kl'] += kl.item()
                metrics['nll'] += nll.item()
                metrics['error_mag'] += last_error
                metrics['constraint'] += constraint.item()
                metrics['shadow'] += shadow.item()
                metrics['l2_reg'] += l2_reg.item()
                metrics['halt'] += halt.item()
                metrics['resource'] += resource.item()
                metrics['mem'] += mem_eff.item()
                batch_kl += kl.item()

            # BPTT after sequence
            loss_final = loss_seq / cfg.SEQ_LEN
            self.opt.step(loss_final, batch_kl/cfg.SEQ_LEN, error_magnitude=last_error)

        # Console Output
        avg = {k: v/(10*cfg.SEQ_LEN) for k, v in metrics.items()}
        print(f"Ep {epoch_idx:04d} | Loss: {avg['loss']:.3f} | NLL: {avg['nll']:.3f} | KL: {avg['kl']:.4f} | ErrMag: {avg['error_mag']:.2f}")

        # Deep Scan (Every 10)
        if epoch_idx % 10 == 0:
            print(f"         ---------------------------------------------------------------")
            print(f"         MDL: Constr: {avg['constraint']:.4f} | Shadow: {avg['shadow']:.4f} | MemEff: {avg['mem']:.2f}")
            print(f"         SYS: L2Reg: {avg['l2_reg']:.4f} | Halt: {avg['halt']:.4f}")
            print(f"         ---------------------------------------------------------------")

        # Live Gen (Every 20)
        if epoch_idx % 20 == 0:
            self.generate_preview()

    def generate_preview(self):
        self.model.eval()
        with torch.no_grad():
            start_str = "The "
            ids = [data_loader.char_to_ix.get(c, 0) for c in start_str]
            inp = torch.tensor(ids, dtype=torch.long).to(DEVICE)
            hid = None
            
            for k in range(len(inp)-1):
                _, hid, *rest = self.model(inp[k].view(1,1), hid, training=False)
            
            curr = inp[-1].view(1,1)
            txt = start_str
            for _ in range(80):
                out, hid, *rest = self.model(curr, hid, training=False)
                idx = torch.argmax(out).item()
                txt += data_loader.ix_to_char[idx]
                curr = torch.tensor([[idx]], dtype=torch.long).to(DEVICE)
            print(f"   >>> GEN: {txt.replace('\n', ' ')}...")
            print(f"   >>> MEM: {len(self.model.replay_buffer.buffer)} samples.")
        self.model.train()

# ==========================================
# 9. Diagnostics Suite (Visualizer)
# ==========================================
class DiagnosticsSuite:
    """Generates all 14+ research plots."""
    
    # [FIX] Force headless backend so it works on VMs/Servers
    import matplotlib
    matplotlib.use('Agg') 
    
    @staticmethod
    def run_inference_scan(model) -> Dict:
        print("\n[Diagnostics] Running Inference Scan (200 steps)...")
        model.eval()
        data = defaultdict(list)
        hidden = None
        # Start token "The "
        x = torch.tensor([[data_loader.char_to_ix.get('T', 0)]], device=DEVICE)
        
        for _ in range(200):
            with torch.no_grad():
                (logits, hidden, halt, l2_reg, kl, alignment, transition, drift, probe, mus, 
                 stack_depth, aux_c, aux_r, mem_eff, shadow, resource, constraint) = model(x, hidden, training=False)
                
                # Collect Scalar Metrics
                data['stack'].append(stack_depth.item())
                data['l2_reg'].append(l2_reg.item())
                data['kl'].append(kl.item())
                data['mem_eff'].append(mem_eff.item())
                data['drift'].append(drift.item())
                data['resource'].append(resource.item())
                data['shadow'].append(shadow.item())
                data['halt'].append(halt.mean().item())
                data['constraint'].append(constraint.item())
                
                # Collect Phase
                z = hidden[0].cpu().squeeze()
                data['real'].append(z.real.item())
                data['imag'].append(z.imag.item())
                
                # Next Token
                probs = F.softmax(logits, dim=-1)
                x = torch.multinomial(probs, 1)
                
                # Collect Symbol
                data['sym'].append(hidden[7].item()) # Hard index

        return data

    @staticmethod
    def generate_all(model):
        data = DiagnosticsSuite.run_inference_scan(model)
        print(f"\n[Diagnostics] Saving charts to: {os.getcwd()}")
        
        DiagnosticsSuite.plot_internal_metrics(data)
        DiagnosticsSuite.plot_phase(data)
        DiagnosticsSuite.plot_topology(model)
        DiagnosticsSuite.plot_attention(model)
        DiagnosticsSuite.plot_activation_map(model)
        DiagnosticsSuite.plot_spectrum(data)
        DiagnosticsSuite.plot_resource_usage(data)
        DiagnosticsSuite.plot_shadow_consistency(data) 
        DiagnosticsSuite.print_logic(model)
        DiagnosticsSuite.generative_sampling(model)
        DiagnosticsSuite.anomaly_test(model)
        print("[Diagnostics] All diagnostics generated.")

    @staticmethod
    def plot_internal_metrics(data):
        plt.figure(figsize=(10, 10))
        plt.subplot(2,2,1); plt.plot(data['stack']); plt.title("Stack Depth")
        plt.subplot(2,2,2); plt.plot(data['l2_reg']); plt.title("L2 Regularization")
        plt.subplot(2,2,3); plt.plot(data['kl']); plt.title("KL Divergence")
        plt.subplot(2,2,4); plt.plot(data['mem_eff']); plt.title("Memory Efficiency")
        plt.tight_layout(); plt.savefig("2_internal_metrics.png"); plt.close()
        
        plt.figure(figsize=(10, 4))
        plt.plot(data['drift'], color='crimson'); plt.title("Semantic Drift")
        plt.savefig("10_semantic_drift.png"); plt.close()
        
        plt.figure(figsize=(10, 4))
        plt.plot(data['halt'], color='orange'); plt.title("Ponder Depth (Halt)")
        plt.savefig("9_ponder_profile.png"); plt.close()
        print("  - Saved Internal Metrics, Drift, and Ponder charts")

    @staticmethod
    def plot_phase(data):
        plt.figure(figsize=(8,8))
        plt.scatter(data['real'], data['imag'], alpha=0.5, c=range(len(data['real'])))
        plt.title("Phase Space Scatter")
        plt.savefig("4_phase_scatter.png"); plt.close()
        
        # Coherence
        phases = np.angle(np.array(data['real']) + 1j * np.array(data['imag']))
        diff = phases[1:] - phases[:-1]
        plt.figure(figsize=(10,4)); plt.plot(diff); plt.title("Phase Difference")
        plt.savefig("5_phase_diff_sequence.png"); plt.close()
        print("  - Saved Phase Scatter and Coherence")

    @staticmethod
    def plot_resource_usage(data):
        plt.figure(figsize=(10,4))
        plt.plot(data['resource'], color='brown')
        plt.title("Resource Utilization")
        plt.savefig("11_resource_usage.png"); plt.close()

    @staticmethod
    def plot_shadow_consistency(data): 
        plt.figure(figsize=(10,4))
        plt.plot(data['shadow'], color='purple')
        plt.title("Shadow Model Consistency")
        plt.savefig("12_model_divergence.png"); plt.close()
        print("  - Saved Model Divergence")

    @staticmethod
    def plot_topology(model):
        try:
            adj = torch.sigmoid(model.vq.transition_matrix).detach().cpu().numpy()
            G = nx.DiGraph()
            for i in range(cfg.VOCAB_SIZE): G.add_node(i)
            for i in range(cfg.VOCAB_SIZE):
                for j in range(cfg.VOCAB_SIZE):
                    if adj[i, j] > 0.5: G.add_edge(i, j, weight=adj[i,j])
            plt.figure(figsize=(12,12))
            nx.draw(G, node_size=50, alpha=0.6, width=0.5)
            plt.title("Semantic Topology")
            plt.savefig("1_semantic_topology.png"); plt.close()
            print("  - Saved Topology Graph")
        except Exception as e: print(f"  [!] Topology Plot Error: {e}")

    @staticmethod
    def plot_attention(model):
        model.eval()
        attn_data = []
        def hook(m, i, o):
            mag = torch.abs(o).detach().cpu().squeeze()
            attn_data.append(mag.numpy())
        h = model.cell.attention.register_forward_hook(hook)
        x = torch.tensor([[0]], dtype=torch.long).to(DEVICE)
        model(x, None, training=False)
        h.remove()
        if len(attn_data) > 0:
            data = np.stack(attn_data)
            if data.ndim == 1: data = data[:, None]
            plt.figure(figsize=(10,6)); plt.imshow(data, aspect='auto', cmap='inferno')
            plt.title("Attention Magnitude"); plt.savefig("6_attention_eeg.png"); plt.close()
            print("  - Saved Attention Heatmap")

    @staticmethod
    def plot_activation_map(model):
        activations = []
        def hook(m, i, o):
            mag = torch.abs(o[0]).detach().cpu().squeeze()
            activations.append(mag.numpy())
        h = model.cell.register_forward_hook(hook)
        x = torch.tensor([[0]], dtype=torch.long).to(DEVICE)
        model(x, None, training=False)
        h.remove()
        if len(activations) > 0:
            data = np.stack(activations)
            plt.figure(figsize=(10,6)); plt.imshow(data, aspect='auto', cmap='magma')
            plt.title("Neural State Activation"); plt.savefig("8_activation_map.png"); plt.close()

    @staticmethod
    def plot_spectrum(data):
        counts = np.bincount(data['sym'], minlength=cfg.VOCAB_SIZE)
        plt.figure(figsize=(12,4))
        plt.bar(range(cfg.VOCAB_SIZE), counts)
        plt.title(f"Symbol Usage (Active: {np.count_nonzero(counts)})")
        plt.savefig("7_symbol_spectrum.png"); plt.close()

    @staticmethod
    def print_logic(model):
        print("\n[Logic Extraction]")
        adj = model.vq.transition_matrix.detach().cpu().numpy()
        probs = np.exp(-adj) / np.sum(np.exp(-adj), axis=1, keepdims=True)
        count = 0
        for i in range(cfg.VOCAB_SIZE):
            best = np.argmax(probs[i])
            if probs[i, best] > 0.02:
                print(f"  Symbol {i} -> {best} (p={probs[i, best]:.4f})")
                count += 1
                if count > 5: break

    @staticmethod
    def generative_sampling(model):
        print("\n[Generative Sampling]")
        curr = 0; s = ""
        for _ in range(50):
            logits = -model.vq.transition_matrix[curr]
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            z_flat = model.vq.codebook[idx].unsqueeze(0)
            z = torch.complex(z_flat[...,:cfg.DIM], z_flat[...,cfg.DIM:])
            out = model.decoder(model.proj(z))
            char_idx = torch.argmax(out).item()
            s += data_loader.ix_to_char[char_idx]
            curr = idx
        print(f"  Sampled: {s}")

    @staticmethod
    def anomaly_test(model):
        print("\n[Anomaly Detection]")
        bad_txt = "True without falsehood certain and most banana"
        ids = [data_loader.char_to_ix.get(c, 0) for c in bad_txt if c in data_loader.char_to_ix]
        if not ids: ids = [0]
        inp = torch.tensor(ids, dtype=torch.long).to(DEVICE)
        hid = None
        anoms = []
        with torch.no_grad():
            for i in range(len(inp)):
                out = model(inp[i].view(1,1), hid, training=False)
                hid = out[1]
                anoms.append(out[-1].item()) # Constraint cost
        plt.figure(figsize=(10,4)); plt.plot(anoms, 'r-o')
        plt.title("Anomaly Detection (Constraint Cost)"); plt.savefig("13_anomaly_detection.png"); plt.close()
        print("  - Saved Anomaly Chart")


# ==========================================
# 10. Main Execution
# ==========================================
if __name__ == "__main__":
    model = ComplexRecurrentSequenceNetwork(data_loader.vocab_size, cfg.DIM).to(DEVICE)
    trainer = Trainer(model)
    
    # Run Training
    for epoch in range(cfg.EPOCHS):
        trainer.train_epoch(epoch)
        
    # Save Model
    save_path = os.path.join(os.getcwd(), "crsn_v55_master.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[SYSTEM] Model saved to: {save_path}")
    
    # Run Visualizer
    DiagnosticsSuite.generate_all(model)
    
    # Auto-Download (Colab Compatibility)
    try:
        from google.colab import files
        print("Attempting Colab download...")
        files.download("crsn_v55_master.pth")
        files.download("2_internal_metrics.png")
        files.download("4_phase_scatter.png")
        files.download("1_semantic_topology.png")
        files.download("6_attention_eeg.png")
        files.download("8_activation_map.png")
        files.download("13_anomaly_detection.png")
        files.download("10_semantic_drift.png")
        files.download("9_ponder_profile.png")
        files.download("11_resource_usage.png")
        files.download("12_model_divergence.png")
        files.download("5_phase_diff_sequence.png")
        files.download("7_symbol_spectrum.png")
    except ImportError:
        pass
