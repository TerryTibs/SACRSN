# ============================================================
# SACRSN v54-CPU-Pocket: FINAL FIXED EDITION
# ------------------------------------------------------------
# 1. FIX: "return model" indentation fixed (Runs all 1000 epochs).
# 2. LOGGING: Prints every epoch so you know it's working.
# 3. DATA: Offline "Pocket Library" active.
# ============================================================

import os
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# CPU Optimization for VM
torch.set_num_threads(os.cpu_count() if os.cpu_count() else 1)

# Optional: Plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not found. Dashboard will be skipped.")

# ==========================================
# 0. Determinism & Setup
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE} (Threads: {torch.get_num_threads()})")

# ==========================================
# 1. Configuration (CPU Mode)
# ==========================================
CONFIG = {
    # Architecture
    "seq_len": 64,
    "dim": 128,
    "symbols": 128,
    "stack_size": 16,
    "max_depth": 8,
    "act_threshold": 0.99,
    
    # Training
    "epochs": 1000,
    "lr": 1e-3,
    "grad_clip": 1.0,
    "eps": 1e-6,
    "temp": 1.0,
    "residual_weight": 0.1,
    
    # Loss Weights
    "lambda_nll": 1.0,
    "lambda_halt": 0.01,
    "lambda_energy": 0.01,
    "lambda_kl": 0.05,
    "lambda_diversity": 0.1,
    "lambda_ground": 0.1,
    "lambda_meta_energy": 0.1,
    "lambda_aux_class": 0.1,
    "lambda_aux_recon": 0.1,
    "lambda_drift": 0.05,
    "lambda_probe": 0.1,
    "lambda_tom": 0.1,
    "ethical_weight": 0.005,
    "symbol_consistency_weight": 0.01,
    "phase_consistency_weight": 0.05,
    
    # Annealing
    "kl_cycle_len": 200, 
    "kl_max_beta": 0.05,
    
    # Meta / Psych
    "meta_alpha": 0.1,
    "meta_lr": 0.01,
    "ground_dim": 32,
    "entropy_delta_thresh": 0.5,
    "meta_hierarchy_depth": 2,
    "reframe_window": 8,
    "entropy_threshold": 0.3,
    "rigidity_limit": 0.9,
    "pacing_interval": 5,     
    
    # Physiology
    "metabolic_capacity": 100.0,
    "metabolic_decay": 0.0,
    "metabolic_recovery": 1.0,  
    "eaft_alpha": 2.0,
    "act_temp_base": 1.0,     
}

# ==========================================
# 2. Data (Offline Pocket Library)
# ==========================================
TEXT_DATA = """
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

print(f"Pocket Library Loaded: {len(TEXT_DATA)} characters.")

chars = sorted(list(set(TEXT_DATA)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(DEVICE)

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
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.fc_real.weight)
        nn.init.orthogonal_(self.fc_imag.weight)
    def forward(self, z):
        return torch.complex(
            self.fc_real(z.real) - self.fc_imag(z.imag),
            self.fc_real(z.imag) + self.fc_imag(z.real)
        )

class ContextAwareAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = ComplexLinear(dim)
        self.k_proj = ComplexLinear(dim)
        self.v_proj = ComplexLinear(dim)
        self.context_proj = nn.Linear(dim*2, dim*2) 
        self.scale = dim ** -0.5
    
    def forward(self, z, context=None):
        q = self.q_proj(z); k = self.k_proj(z); v = self.v_proj(z)
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
    def __init__(self, dim):
        super().__init__()
        self.q_proj = ComplexLinear(dim)
        self.k_proj = ComplexLinear(dim)
        self.v_proj = ComplexLinear(dim)
        self.scale = dim ** -0.5
    def forward(self, z):
        q = self.q_proj(z); k = self.k_proj(z); v = self.v_proj(z)
        q_flat = torch.cat([q.real, q.imag], dim=-1)
        k_flat = torch.cat([k.real, k.imag], dim=-1)
        attn_scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        v_real = torch.matmul(attn_weights, v.real)
        v_imag = torch.matmul(attn_weights, v.imag)
        return torch.complex(v_real, v_imag)

# ==========================================
# 4. Advanced Modules
# ==========================================
class HomeostaticRegulator(nn.Module):
    def __init__(self, capacity=100.0):
        super().__init__()
        self.capacity = capacity
        self.register_buffer("energy", torch.tensor(capacity))
        self.decay = CONFIG["metabolic_decay"]
        self.recovery = CONFIG["metabolic_recovery"]
    def forward(self, effort):
        self.energy = self.energy - (effort * self.decay)
        self.energy = torch.clamp(self.energy + self.recovery, 0, self.capacity)
        fatigue = self.energy / self.capacity
        return fatigue

class MirrorSystem(nn.Module):
    def __init__(self, dim, n_symbols):
        super().__init__()
        self.other_graph = GroundedMetaGraphVQ(dim, n_symbols)
    def forward(self, z, prev_sym_dist, temp=1.0):
        z_q, soft_sym, _, energy, _, _, _ = self.other_graph(
            z, prev_symbol_dist=prev_sym_dist, temp=temp
        )
        return z_q, energy

class MetaBeliefTracker:
    def __init__(self, n_symbols):
        self.stats = {s: {"count": 0, "avg_loss": 0.0, "avg_ponder": 0.0, "rigidity": 0.0} for s in range(n_symbols)}
    def update(self, sym, loss, ponder, changed):
        if isinstance(sym, torch.Tensor): sym = sym.item()
        d = self.stats[sym]
        d["count"] += 1
        d["avg_loss"] = 0.9 * d["avg_loss"] + 0.1 * loss
        d["avg_ponder"] = 0.9 * d["avg_ponder"] + 0.1 * ponder
        d["rigidity"] = 0.9 * d["rigidity"] + 0.1 * (0 if changed else 1)

class EthicalGate:
    def __init__(self, rigidity_limit=0.9, loss_limit=2.0):
        self.rigidity_limit = rigidity_limit
        self.loss_limit = loss_limit
    def allow(self, meta):
        if meta["rigidity"] > self.rigidity_limit: return False
        if meta["avg_loss"] > self.loss_limit: return False
        return True

class TherapeuticPacer:
    def __init__(self, min_interval=5):
        self.min_interval = min_interval
        self.last_reframe_step = 0
    def allow(self, step):
        if step - self.last_reframe_step >= self.min_interval:
            self.last_reframe_step = step
            return True
        return False

class BeliefReframer(nn.Module):
    def __init__(self, n_symbols, window=8, entropy_threshold=0.3):
        super().__init__()
        self.n_symbols = n_symbols
        self.window = window
        self.entropy_threshold = entropy_threshold
        self.symbol_history = []
        self.belief_labels = defaultdict(int)
    def update_history(self, sym_idx):
        self.symbol_history.append(sym_idx.detach())
        if len(self.symbol_history) > self.window: self.symbol_history.pop(0)
    def belief_entropy(self):
        if len(self.symbol_history) < self.window: return None
        syms = torch.stack(self.symbol_history)
        probs = torch.bincount(syms.flatten(), minlength=self.n_symbols).float()
        probs = probs / probs.sum()
        return -(probs * torch.log(probs + 1e-8)).sum()
    def classify_reframe(self, loss, ponder):
        entropy = self.belief_entropy()
        if entropy is None: return None
        if entropy < self.entropy_threshold and ponder > 1.5: return "meaning"
        if entropy < self.entropy_threshold and loss > 0.5: return "context"
        return None
    def meaning_reframe(self, z_flat, codebook, adjacency, current_sym):
        dists = torch.sum((codebook - z_flat)**2, dim=-1)
        candidates = torch.topk(-dists, k=4).indices
        best = current_sym; best_score = dists[current_sym]
        for s in candidates:
            score = dists[s] + 0.1 * adjacency[current_sym, s]
            if score < best_score: best_score = score; best = s
        return best
    def context_reframe(self, adjacency, current_sym):
        with torch.no_grad():
            adjacency[current_sym] *= 0.9
            adjacency[current_sym] += 0.1 * torch.randn_like(adjacency[current_sym])
    def label_belief(self, sym_idx):
        key = sym_idx.item()
        self.belief_labels[key] += 1
        return key

class EthicalConstraint(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, prev_sym, curr_sym, adjacency):
        if prev_sym is None: return torch.tensor(0.0).to(adjacency.device)
        row_logits = -adjacency[prev_sym] 
        return F.cross_entropy(row_logits.view(-1, CONFIG["symbols"]), curr_sym.view(-1))

class ComplexityModulator(nn.Module):
    def __init__(self, dim, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_var", torch.ones(1))
    def forward(self, z):
        mag = torch.abs(z)
        batch_var = mag.var()
        if self.training:
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var.detach()
        modulation = F.softplus(batch_var / (self.running_var + CONFIG["eps"]))
        return modulation

class AmortizedHallucinationProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.probe = nn.Linear(dim * 2, 1)
    def forward(self, z):
        flat = torch.cat([z.real, z.imag], dim=-1)
        return self.probe(flat)

class MetaIntrospectionTracker(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.prior_lstm = nn.LSTMCell(dim * 2, dim) 
        self.posterior_fc = nn.Linear(dim * 2, dim * 2)
        self.register_buffer("meta_h_prior", torch.zeros(dim)) 
    def forward(self, z, h_prev, c_prev):
        if torch.is_complex(z): z_flat = torch.cat([z.real, z.imag], dim=-1)
        else: z_flat = z 
        h_step_prior_raw, c_next = self.prior_lstm(z_flat, (h_prev, c_prev))
        h_step_prior = h_step_prior_raw.detach() 
        alpha = CONFIG["meta_alpha"]
        h_prior_mixed = alpha * h_step_prior + (1 - alpha) * self.meta_h_prior
        params = self.posterior_fc(z_flat)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        h_posterior = mu + eps * std
        prior_var = torch.ones_like(mu)
        kl_div = 0.5 * ((mu - h_prior_mixed).pow(2) / prior_var + logvar.exp() / prior_var - 1 - logvar).sum(dim=1)
        return h_posterior, c_next, kl_div.mean(), mu
    def update_meta_prior(self, batch_mu):
        with torch.no_grad():
            target = batch_mu.mean(0)
            self.meta_h_prior = (1 - CONFIG["meta_lr"]) * self.meta_h_prior + CONFIG["meta_lr"] * target

class HierarchicalMetaTracker(nn.Module):
    def __init__(self, dim, depth=2):
        super().__init__()
        self.depth = depth
        self.levels = nn.ModuleList()
        for i in range(depth):
            level_dim = dim // (2 ** i)
            if level_dim < 8: level_dim = 8
            self.levels.append(MetaIntrospectionTracker(level_dim))
    def forward(self, z, h_prev_tuple, c_prev_tuple):
        h1_prev, h2_prev = h_prev_tuple
        c1_prev, c2_prev = c_prev_tuple
        h1, c1, kl1, mu1 = self.levels[0](z, h1_prev, c1_prev)
        if len(self.levels) > 1:
            h2, c2, kl2, mu2 = self.levels[1](h1, h2_prev, c2_prev)
            combined_kl = kl1 + kl2
            return (h1, h2), (c1, c2), combined_kl.mean(), (mu1, mu2)
        else:
            return (h1, h1), (c1, c1), kl1.mean(), (mu1, mu1)
            
    def update_meta_prior(self, mus_tuple):
        self.levels[0].update_meta_prior(mus_tuple[0])
        if len(self.levels) > 1:
            self.levels[1].update_meta_prior(mus_tuple[1])

class GroundedMetaGraphVQ(nn.Module):
    def __init__(self, dim, n_symbols):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_symbols, dim*2) * 0.2)
        self.adjacency_energy = nn.Parameter(torch.randn(n_symbols, n_symbols)) 
        self.symbol_ground = nn.Parameter(torch.randn(n_symbols, CONFIG["ground_dim"]))
        self.grounding_proj = nn.Linear(dim*2, CONFIG["ground_dim"])
        self.meta_adjacency = nn.Parameter(torch.randn(n_symbols, n_symbols))
        self.complexity_mod = ComplexityModulator(dim)
    def forward(self, z, prev_symbol_dist=None, prev_entropy=None, temp=1.0):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        d_content = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
                    torch.sum(self.codebook**2, dim=-1) - \
                    2 * torch.matmul(z_flat, self.codebook.t())
        d_content = d_content / z_flat.shape[-1]
        up_scalar = self.complexity_mod(z)
        if prev_symbol_dist is not None:
            graph_bias = torch.matmul(prev_symbol_dist, self.adjacency_energy)
            energy_weight = CONFIG["lambda_energy"] * up_scalar
            d_total = d_content - energy_weight * torch.sigmoid(graph_bias)
        else: d_total = d_content
        logits = -d_total
        probs = F.softmax(logits / temp, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        meta_E = torch.tensor(0.0).to(z.device)
        if prev_entropy is not None and prev_symbol_dist is not None:
            delta_H = torch.abs(entropy - prev_entropy)
            meta_gate = torch.sigmoid((delta_H - CONFIG["entropy_delta_thresh"]) * 10.0)
            meta_bias = torch.matmul(prev_symbol_dist, self.meta_adjacency)
            meta_E = meta_gate.mean() * (d_content - torch.sigmoid(meta_bias)).mean()
        if self.training: soft_one_hot = F.gumbel_softmax(logits, tau=temp, hard=False)
        else: soft_one_hot = probs
        hard_idx = torch.argmax(soft_one_hot, dim=-1)
        z_q_flat = torch.matmul(soft_one_hot, self.codebook)
        z_q = torch.complex(z_q_flat[..., :z.shape[-1]], z_q_flat[..., z.shape[-1]:])
        z_proj = self.grounding_proj(z_flat) 
        expected_anchor = torch.matmul(soft_one_hot, self.symbol_ground)
        grounding_loss = F.mse_loss(z_proj, expected_anchor)
        return z_q, soft_one_hot, hard_idx, d_total.mean(), entropy, grounding_loss, meta_E

class EnhancedMemoryStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
        self.memory_attention = ComplexAttention(dim)
        self.compress_proj = nn.Linear(dim*2, dim)
    def forward(self, z, mem, ptr, ctrl):
        push = torch.sigmoid(ctrl[:, 0:1])
        pop  = torch.sigmoid(ctrl[:, 1:2])
        stay = torch.sigmoid(ctrl[:, 2:3])
        total = push + pop + stay + CONFIG["eps"]
        push, pop, stay = push/total, pop/total, stay/total
        ptr_up = torch.roll(ptr, 1, dims=1)
        ptr_down = torch.roll(ptr, -1, dims=1)
        new_ptr = (push * ptr_up) + (pop * ptr_down) + (stay * ptr)
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        mem = mem * (1 - push.unsqueeze(-1)) + push.unsqueeze(-1) * z_flat.unsqueeze(1)
        mem_complex = torch.complex(mem[..., :self.dim], mem[..., self.dim:])
        attended_mem = self.memory_attention(mem_complex)
        attended_mem_flat = torch.cat([attended_mem.real, attended_mem.imag], dim=-1)
        compressed_mem = self.compress_proj(attended_mem_flat)
        read_flat = (attended_mem_flat * new_ptr.unsqueeze(-1)).sum(1)
        z_read = torch.complex(read_flat[:, :self.dim], read_flat[:, self.dim:])
        active_slots = (new_ptr > 0.1).float().sum(1).mean()
        return z_read, mem, new_ptr, active_slots

# ==========================================
# 5. Core Processor
# ==========================================
class ResidualRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.attention = ContextAwareAttention(dim)
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        self.residual_proj = nn.Linear(dim * 2, dim * 2) 
        nn.init.constant_(self.halt_linear.bias, 2.0)
    def forward(self, z):
        z_in = z
        z_proc = self.act(self.norm(self.linear(z)))
        z_proc = self.attention(z_proc, context=torch.cat([z.real, z.imag], dim=-1))
        residual_input = torch.cat([z.real, z.imag], dim=-1)
        residual_output = self.residual_proj(residual_input)
        z_proc = z_proc + CONFIG["residual_weight"] * torch.complex(residual_output[..., :self.dim], residual_output[..., self.dim:])
        z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
        halt_logit = self.halt_linear(z_flat)
        stack_logits = self.stack_ctrl(z_flat)
        return z_proc, halt_logit, stack_logits

class QuantumProbabilityLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.probability_layer = nn.Linear(dim*2, dim)
    def forward(self, z):
        flat = torch.cat([z.real, z.imag], dim=-1)
        logits = self.probability_layer(flat)
        return logits

# ==========================================
# 6. Master Model
# ==========================================
class EnhancedUberCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        
        self.image_encoder = nn.Linear(256, dim)
        self.audio_encoder = nn.Linear(128, dim)
        
        self.cell = ResidualRecursiveCell(dim)
        self.vq = GroundedMetaGraphVQ(dim, CONFIG["symbols"])
        self.hallucination_vae = HierarchicalMetaTracker(dim, CONFIG["meta_hierarchy_depth"])
        self.stack = EnhancedMemoryStack(dim, CONFIG["stack_size"])
        self.quantum_layer = QuantumProbabilityLayer(dim)
        self.decoder = nn.Linear(dim, vocab_size)
        
        self.aux_classifier = nn.Linear(dim*2, 10) 
        self.aux_reconstruction = nn.Linear(dim*2, dim*2)
        
        self.hallucination_probe = AmortizedHallucinationProbe(dim)
        self.register_buffer("prev_sym_soft", torch.zeros(CONFIG["symbols"]))
        
        self.reframer = BeliefReframer(CONFIG["symbols"])
        self.meta_tracker = MetaBeliefTracker(CONFIG["symbols"])
        self.ethical_gate = EthicalGate(rigidity_limit=CONFIG["rigidity_limit"])
        self.pacer = TherapeuticPacer(min_interval=CONFIG["pacing_interval"])
        self.ethics = EthicalConstraint()
        
        self.homeostasis = HomeostaticRegulator(capacity=CONFIG["metabolic_capacity"])
        self.mirror_system = MirrorSystem(dim, CONFIG["symbols"])

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, input_ids, hidden=None, injected_thought=None, 
                image_input=None, audio_input=None, global_step=0, training=True):
        
        z = self.embed(input_ids).squeeze(1)
        batch_size = input_ids.size(0)
        z_initial = z.clone()
        
        if image_input is not None: z = z + self.image_encoder(image_input)
        if audio_input is not None: z = z + self.audio_encoder(audio_input)
        if injected_thought is not None: z = z + injected_thought

        if hidden is None:
            z_prev = torch.zeros_like(z)
            h_vae = (torch.zeros(batch_size, self.dim).to(z.device), 
                     torch.zeros(batch_size, self.dim//2).to(z.device))
            c_vae = (torch.zeros(batch_size, self.dim).to(z.device), 
                     torch.zeros(batch_size, self.dim//2).to(z.device))
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device); stack_ptr[:, 0] = 1.0
            prev_soft_sym = None; prev_entropy = None; prev_sym_idx = None
        else:
            z_prev, h_vae, c_vae, stack_mem, stack_ptr, prev_soft_sym, prev_entropy, prev_sym_idx = hidden
            z = 0.5 * z + 0.5 * z_prev

        halt_cost, energy_cost, kl_cost, ground_cost, meta_cost, drift_cost, probe_logit, tom_cost, ethics_cost = 0, 0, 0, 0, 0, 0, 0, 0, 0
        mem_efficiency_log = []; posterior_mus_l1, posterior_mus_l2 = [], []
        halting_probability = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        z_weighted = torch.zeros_like(z)
        
        current_soft_sym = prev_soft_sym
        current_entropy = prev_entropy
        stack_depth_log = torch.tensor(0.0).to(z.device)
        current_hard_sym = prev_sym_idx
        
        fatigue = self.homeostasis(torch.tensor(1.0).to(DEVICE))
        
        for t in range(CONFIG["max_depth"]):
            z_proc, halt_logit, stack_logits = self.cell(z)
            z_stack, stack_mem, stack_ptr, active_slots = self.stack(z_proc, stack_mem, stack_ptr, stack_logits)
            mem_efficiency_log.append(active_slots)
            z = z_proc + z_stack
            
            h_vae, c_vae, kl_div, mus = self.hallucination_vae(z, h_vae, c_vae)
            posterior_mus_l1.append(mus[0]);
            if len(mus) > 1: posterior_mus_l2.append(mus[1])
            else: posterior_mus_l2.append(mus[0])
            
            probe_logit += self.hallucination_probe(z)
            
            current_temp = CONFIG["temp"] 
            
            z_q, soft_sym, hard_idx, energy, ent, ground, meta = self.vq(
                z, current_soft_sym, prev_entropy=current_entropy, temp=current_temp
            )
            
            _, mirror_energy = self.mirror_system(z, current_soft_sym, temp=current_temp)
            tom_cost += mirror_energy
            
            if training:
                sym_item = hard_idx[0].item()
                self.reframer.update_history(hard_idx)
                temp_act = CONFIG["act_temp_base"] * (0.9 ** t)
                p_halt = torch.sigmoid(halt_logit / temp_act)
                reframe_type = self.reframer.classify_reframe(loss=energy.item(), ponder=p_halt.mean().item())
                if reframe_type:
                    meta_stat = self.meta_tracker.stats[sym_item]
                    if not self.ethical_gate.allow(meta_stat) or not self.pacer.allow(global_step):
                         reframe_type = None
                if reframe_type == "meaning":
                    z_flat = torch.cat([z.real, z.imag], dim=-1).detach()
                    hard_idx = self.reframer.meaning_reframe(
                        z_flat, self.vq.codebook.detach(), self.vq.adjacency_energy.detach(), hard_idx
                    )
                elif reframe_type == "context":
                    self.reframer.context_reframe(self.vq.adjacency_energy.data, hard_idx)
                changed = (sym_item != current_hard_sym) if current_hard_sym is not None else False
                self.meta_tracker.update(sym_item, energy.item(), p_halt.mean().item(), changed)
                self.reframer.label_belief(hard_idx)
            
            eth_loss = self.ethics(current_hard_sym, hard_idx, self.vq.adjacency_energy)
            ethics_cost += eth_loss
            
            current_hard_sym = hard_idx
            current_soft_sym = soft_sym
            current_entropy = ent
            z = 0.7 * z + 0.3 * z_q
            
            current_act_thresh = CONFIG["act_threshold"] 
            
            temp_act = CONFIG["act_temp_base"] * (0.9 ** t)
            p_halt = torch.sigmoid(halt_logit / temp_act)
            p = torch.minimum(remain, p_halt)
            if t == CONFIG["max_depth"] - 1: p = remain
            
            z_weighted = z_weighted + (p * z)
            halting_probability = halting_probability + p
            remain = remain - p
            
            drift = torch.mean((z.real - z_initial.real)**2)
            
            halt_cost += p * (t + 1)
            energy_cost += energy
            kl_cost += kl_div
            ground_cost += ground
            meta_cost += meta
            drift_cost += drift
            stack_depth_log = torch.sum(stack_ptr * torch.arange(CONFIG["stack_size"], device=z.device), dim=1)

        quantum_logits = self.quantum_layer(z_weighted)
        logits = self.decoder(quantum_logits)
        
        features_flat = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        aux_class = self.aux_classifier(features_flat)
        aux_recon = self.aux_reconstruction(features_flat)
        
        if current_soft_sym is not None:
            with torch.no_grad():
                self.prev_sym_soft = self.prev_sym_soft * 0.9 + current_soft_sym.mean(0) * 0.1
        
        next_hidden = (z_weighted, h_vae, c_vae, stack_mem, stack_ptr, current_soft_sym, current_entropy, current_hard_sym)
        avg_mu_l1 = torch.stack(posterior_mus_l1).mean(0)
        avg_mu_l2 = torch.stack(posterior_mus_l2).mean(0)
        avg_mem_efficiency = sum(mem_efficiency_log) / len(mem_efficiency_log)
        
        return logits, next_hidden, halt_cost, energy_cost, kl_cost, ground_cost, meta_cost, drift_cost, probe_logit, (avg_mu_l1, avg_mu_l2), stack_depth_log, aux_class, aux_recon, avg_mem_efficiency, tom_cost, fatigue, ethics_cost

# ==========================================
# 7. Training (Optimized Loop)
# ==========================================
class AdaptiveOptimizer:
    def __init__(self, model, base_lr=1e-3):
        self.opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
        self.base_lr = base_lr
        self.kl_history = []
    def step(self, loss, kl_value):
        if math.isnan(kl_value) or math.isinf(kl_value): kl_value = 100.0
        self.kl_history.append(kl_value)
        if len(self.kl_history) > 5:
            recent = np.array(self.kl_history[-5:])
            volatility = np.std(recent)
            if volatility > 0.5: lr = self.base_lr * 0.1
            elif volatility > 0.1: lr = self.base_lr * 0.5
            else: lr = self.base_lr
        else: lr = self.base_lr
        for param_group in self.opt.param_groups: param_group['lr'] = lr
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.opt.param_groups[0]['params'], 1.0)
        self.opt.step()
        return lr

def eaft_soft_weight(entropy, p_true):
    norm_ent = torch.tanh(entropy)
    confident_error = (1.0 - norm_ent) * (1.0 - p_true)
    return torch.exp(-CONFIG["eaft_alpha"] * confident_error).detach()

def train():
    model = EnhancedUberCRSN(vocab_size, CONFIG["dim"]).to(DEVICE)
    optimizer = AdaptiveOptimizer(model, base_lr=CONFIG["lr"])
    
    print(f"--- Training SACRSN v54-CPU-Pocket ---")
    
    global_step = 0
    
    # Simple Batching logic for CPU
    def get_batch():
        ix = torch.randint(len(data_tensor) - CONFIG["seq_len"], (1,))
        x = data_tensor[ix : ix + CONFIG["seq_len"]].view(1, CONFIG["seq_len"])
        y = data_tensor[ix+1 : ix + CONFIG["seq_len"] + 1].view(1, CONFIG["seq_len"])
        return x, y

    for epoch in range(CONFIG["epochs"]):
        hidden = None
        total_loss = 0
        avg_kl = 0
        
        # Reduced steps for CPU feedback (10 per epoch)
        for step in range(10): 
            global_step += 1
            x_seq, y_seq = get_batch()
            hidden = None 
            loss_seq = 0
            
            for t in range(CONFIG["seq_len"]):
                x = x_seq[:, t].view(1, 1)
                y = y_seq[:, t].view(1)
                
                try:
                    logits, hidden, halt, energy, kl, ground, meta, drift, probe_logit, mus, _, aux_c, aux_r, _, tom, fatigue, ethics = model(x, hidden, global_step=global_step, training=True)
                except RuntimeError as e:
                    print(f"Error: {e}"); hidden = None; break

                h_z, h_h, h_c, h_mem, h_ptr, h_sym, h_ent, h_idx = hidden
                
                # Phase Consistency Loss
                phase_loss = 0.0
                if t > 0:
                    phase_change = torch.abs(torch.angle(h_z) - torch.angle(prev_z))
                    phase_loss = torch.mean(F.relu(phase_change - 1.57)) 
                prev_z = h_z.detach()
                
                h_h_det = (h_h[0].detach(), h_h[1].detach())
                h_c_det = (h_c[0].detach(), h_c[1].detach())
                hidden = (h_z.detach(), h_h_det, h_c_det, h_mem.detach(), h_ptr.detach(), 
                          h_sym.detach() if h_sym is not None else None,
                          h_ent.detach() if h_ent is not None else None, h_idx)
                
                model.hallucination_vae.update_meta_prior(mus)
                
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
                weight = eaft_soft_weight(entropy, probs[0, y])
                
                nll = (F.cross_entropy(logits, y, reduction='none') * weight).mean()
                buffer = model.prev_sym_soft + 1e-9
                L_prior = -CONFIG["lambda_diversity"] * (buffer * torch.log(buffer)).sum()
                adj_sig = torch.sigmoid(model.vq.adjacency_energy)
                loss_consistency = CONFIG["symbol_consistency_weight"] * -(adj_sig * torch.log(adj_sig + CONFIG["eps"])).sum(dim=-1).mean()
                
                dummy_class = torch.randint(0, 10, (1,)).to(DEVICE)
                L_aux_c = F.cross_entropy(aux_c, dummy_class)
                L_aux_r = torch.mean(aux_r**2)
                probe_target = torch.sigmoid(entropy).detach()
                L_probe = F.binary_cross_entropy_with_logits(probe_logit.mean(), probe_target.mean())
                
                # KL Annealing
                cycle_progress = (epoch % CONFIG["kl_cycle_len"]) / CONFIG["kl_cycle_len"]
                current_beta = CONFIG["kl_max_beta"] * (1.0 if cycle_progress > 0.5 else cycle_progress * 2)

                loss_step = CONFIG["lambda_nll"] * nll + \
                       CONFIG["lambda_halt"] * halt + \
                       CONFIG["lambda_energy"] * energy + \
                       current_beta * kl + \
                       CONFIG["lambda_ground"] * ground + \
                       CONFIG["lambda_meta_energy"] * meta + \
                       CONFIG["lambda_aux_class"] * L_aux_c + \
                       CONFIG["lambda_aux_recon"] * L_aux_r + \
                       CONFIG["lambda_drift"] * drift + \
                       CONFIG["lambda_probe"] * L_probe + \
                       CONFIG["lambda_tom"] * tom + \
                       CONFIG["ethical_weight"] * ethics + \
                       loss_consistency + L_prior + \
                       CONFIG["phase_consistency_weight"] * phase_loss
                
                loss_seq += loss_step
                avg_kl += kl.item()

            loss_final = loss_seq / CONFIG["seq_len"]
            lr = optimizer.step(loss_final, avg_kl/CONFIG["seq_len"])
            total_loss += loss_final.item()

        # Logging: Print EVERY epoch so you know it's alive
        print(f"Ep {epoch:04d} | Loss: {total_loss/10:.4f} | Avg KL: {avg_kl/(10*CONFIG['seq_len']):.4f} | LR: {lr:.6f}")
        
        # Live generation every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                start_str = "The "
                input_ids = [char_to_ix.get(c, 0) for c in start_str]
                input_tensor = torch.tensor(input_ids, dtype=torch.long).to(DEVICE)
                hidden_gen = None
                
                for k in range(len(input_tensor) - 1):
                    _, hidden_gen, *args = model(input_tensor[k].view(1,1), hidden_gen, training=False)
                
                curr_input = input_tensor[-1].view(1,1)
                out_str = start_str
                
                for _ in range(80):
                    logits, hidden_gen, *args = model(curr_input, hidden_gen, training=False)
                    probs = F.softmax(logits, dim=-1)
                    next_ix = torch.argmax(probs).item() 
                    out_str += ix_to_char[next_ix]
                    curr_input = torch.tensor([[next_ix]], dtype=torch.long).to(DEVICE)
                print(f"   >>> GEN: {out_str.replace('\n', ' ')}...")
            model.train()
            
    return model # <--- INDENTATION FIXED: Outside loop

# ==========================================
# 8. Complete Diagnostics Suite
# ==========================================
def visualize_all(model):
    print("\n--- Generating Core Diagnostics ---")
    model.eval()
    symbol_to_char = defaultdict(list)
    hidden, prev_sym = None, None
    stack_depths, kl_vals, energies, mem_eff = [], [], [], []
    phase_reals, phase_imags = [], []
    
    print("Running Inference Scan...")
    x = torch.tensor([[char_to_ix["T"] if "T" in char_to_ix else 0]], device=DEVICE)
    gen_text = "T"
    
    for _ in range(200):
        with torch.no_grad():
            logits, hidden, _, energy, kl, _, _, _, _, _, depth, _, _, eff, _, _, _ = model(x, hidden, training=False)
            stack_depths.append(depth.item() if isinstance(depth, torch.Tensor) else 0)
            kl_vals.append(kl.item())
            energies.append(energy.item())
            mem_eff.append(eff.item())
            
            z = hidden[0].cpu().squeeze()
            if z.dim() > 0: 
                phase_reals.append(z.real[0].item())
                phase_imags.append(z.imag[0].item())
            else: 
                phase_reals.append(z.real.item())
                phase_imags.append(z.imag.item())
            
            sym_idx = hidden[7]
            if prev_sym is not None and sym_idx is not None:
                probs = F.softmax(logits, dim=-1)
                next_ix = torch.multinomial(probs, 1)
                char = ix_to_char[next_ix.item()]
                symbol_to_char[prev_sym.item()].append(char)
                gen_text += char
                x = next_ix
            else:
                 probs = F.softmax(logits, dim=-1)
                 x = torch.multinomial(probs, 1)
            prev_sym = sym_idx

    print(f"Generated: {gen_text}\n")

    # Topology
    adj_probs = torch.sigmoid(model.vq.adjacency_energy).detach().cpu().numpy()
    G = nx.DiGraph()
    node_labels = {}
    for i in range(CONFIG["symbols"]): 
        G.add_node(i)
        char_list = symbol_to_char.get(i, [])
        if char_list: node_labels[i] = f"{max(set(char_list), key=char_list.count)}\n({len(char_list)})"
        else: node_labels[i] = str(i)
    for i in range(CONFIG["symbols"]):
        for j in range(CONFIG["symbols"]):
            w = adj_probs[i, j]
            if w < 0.5: G.add_edge(i, j, weight=1.0-w)
    plt.figure(figsize=(12, 12))
    try: pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    except: pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=600, alpha=0.9, node_color='#a0cbe2')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3, edge_color='gray')
    plt.title("1_semantic_topology")
    plt.savefig("1_semantic_topology.png")
    plt.close()

    # MRI
    plt.figure(figsize=(10, 10))
    plt.subplot(2,2,1); plt.plot(stack_depths); plt.title("Stack Depth")
    plt.subplot(2,2,2); plt.plot(energies); plt.title("Energy")
    plt.subplot(2,2,3); plt.plot(kl_vals); plt.title("KL")
    plt.subplot(2,2,4); plt.plot(mem_eff); plt.title("Mem Efficiency")
    plt.tight_layout()
    plt.savefig("2_cognitive_mri.png")
    plt.close()
    
    # Grounding
    grounding = model.vq.symbol_ground.detach().cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(grounding[:20], cmap='inferno')
    plt.title("Symbol Grounding")
    plt.savefig("3_grounding.png")
    plt.close()

    # Phase Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(phase_reals, phase_imags, c=range(len(phase_reals)), cmap='plasma', alpha=0.5)
    plt.title("4_phase_scatter")
    plt.axis('equal')
    plt.savefig("4_phase_scatter.png")
    plt.close()

    # Specific Phase Diff
    phases = torch.tensor(phase_reals) + 1j * torch.tensor(phase_imags)
    phases = torch.angle(phases)
    phase_diff = phases[1:] - phases[:-1]
    plt.figure(figsize=(10, 4))
    plt.plot(phase_diff.cpu().numpy(), color='blue')
    plt.title("5_phase_diff_sequence")
    plt.savefig("5_phase_diff_sequence.png")
    plt.close()

    print("Core Diagnostics saved.")

def advanced_phase_analysis(model):
    print("\n--- Phase Coherence (Dataset) ---")
    model.eval()
    hidden = None
    zs = []
    sample_size = min(len(data_tensor), 2000)
    with torch.no_grad():
        for i in range(sample_size):
            x = data_tensor[i].view(1, 1)
            _, hidden, *args = model(x, hidden, training=False)
            zs.append(hidden[0])
    z_stack = torch.cat(zs)
    phases = torch.angle(z_stack)
    phase_diff = phases[1:] - phases[:-1]
    coherence = 1.0 - torch.std(phase_diff, dim=1).cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.plot(coherence, color='blue')
    plt.savefig("5_phase_coherence.png")
    plt.close()

def plot_attention_dynamics(model):
    print("\n--- Attention EEG (Intensity) ---")
    model.eval()
    attn_intensities = []
    
    def hook_fn(module, input, output):
        # Capture MAGNITUDE (Intensity) of the attention vector
        z_out = output
        mag = torch.abs(z_out).detach().cpu().squeeze()
        attn_intensities.append(mag.numpy())
        
    handle = model.cell.attention.register_forward_hook(hook_fn)
    x = torch.tensor([[char_to_ix["T"] if "T" in char_to_ix else 0]], device=DEVICE)
    model(x, None, training=False)
    handle.remove()
    
    if len(attn_intensities) > 0:
        map_data = np.stack(attn_intensities)
        plt.figure(figsize=(10, 6))
        plt.imshow(map_data, aspect='auto', cmap='inferno', interpolation='nearest')
        plt.colorbar(label="Attention Output Magnitude")
        plt.title(f"Recursive Attention Intensity (Depth {CONFIG['max_depth']})")
        plt.savefig("6_attention_eeg.png")
        plt.close()

def plot_neural_state_eeg(model):
    print("\n--- Neural State EEG ---")
    model.eval()
    activations = []
    def hook_fn(module, input, output):
        z_proc = output[0].detach().cpu()
        mag = torch.abs(z_proc).squeeze()
        activations.append(mag.numpy())
    handle = model.cell.register_forward_hook(hook_fn)
    x = torch.tensor([[0]], device=DEVICE)
    model(x, None, training=False)
    handle.remove()
    if len(activations) > 0:
        map_data = np.stack(activations) 
        plt.figure(figsize=(10, 6))
        plt.imshow(map_data, aspect='auto', cmap='magma', interpolation='nearest')
        plt.savefig("8_neural_state_eeg.png")
        plt.close()

def plot_symbol_utilization(model):
    print("\n--- Symbol Spectrum (Dead Code) ---")
    model.eval()
    symbol_counts = np.zeros(CONFIG["symbols"])
    hidden = None
    scan_len = min(len(data_tensor), 1000)
    with torch.no_grad():
        for i in range(scan_len):
            x = data_tensor[i].view(1, 1)
            _, hidden, *args = model(x, hidden, training=False)
            curr_sym = hidden[7] 
            if curr_sym is not None:
                symbol_counts += curr_sym.cpu().numpy().flatten()
    
    active_symbols = np.count_nonzero(symbol_counts)
    print(f"Active Symbols: {active_symbols} / {CONFIG['symbols']}")
    print(f"Dead Codes: {CONFIG['symbols'] - active_symbols}")
    
    freqs = symbol_counts / (symbol_counts.sum() + 1e-9)
    plt.figure(figsize=(12, 4))
    plt.bar(range(CONFIG["symbols"]), freqs, color='teal')
    plt.savefig("7_symbol_spectrum.png")
    plt.close()

def plot_ponder_profile(model):
    print("\n--- Ponder Profile ---")
    model.eval()
    x = torch.tensor([[0]], device=DEVICE)
    z = model.embed(x).squeeze(1)
    probs_at_step = []
    for t in range(CONFIG["max_depth"]):
        _, halt_logit, _ = model.cell(z)
        p = torch.sigmoid(halt_logit)
        probs_at_step.append(p.item())
    plt.figure(figsize=(8, 4))
    plt.plot(probs_at_step, marker='o', color='orange')
    plt.savefig("9_ponder_profile.png")
    plt.close()

def plot_semantic_drift(model):
    print("\n--- Semantic Drift ---")
    model.eval()
    x = torch.tensor([[0]], device=DEVICE)
    z_initial = model.embed(x).squeeze(1)
    z = z_initial.clone()
    drifts = []
    for t in range(CONFIG["max_depth"]):
        z_proc, _, _ = model.cell(z)
        drift = torch.mean((z_proc.real - z_initial.real)**2).item()
        drifts.append(drift)
        z = z_proc
    plt.figure(figsize=(8, 4))
    plt.plot(drifts, color='crimson', linestyle='--')
    plt.savefig("10_semantic_drift.png")
    plt.close()

def plot_homeostasis(model):
    print("\n--- Homeostatic State ---")
    model.eval()
    energy_levels = []
    model.homeostasis.energy.fill_(CONFIG["metabolic_capacity"])
    hidden = None
    scan_len = min(len(data_tensor), 500)
    with torch.no_grad():
        for i in range(scan_len):
            x = data_tensor[i].view(1, 1)
            output = model(x, hidden, training=False)
            fatigue = output[-2]
            hidden = output[1]
            energy_levels.append(fatigue.item())
    plt.figure(figsize=(10, 4))
    plt.plot(energy_levels, color='brown')
    plt.savefig("11_homeostasis.png")
    plt.close()

def plot_theory_of_mind(model):
    print("\n--- Theory of Mind Divergence ---")
    adj_self = model.vq.adjacency_energy.detach().cpu().numpy()
    adj_mirror = model.mirror_system.other_graph.adjacency_energy.detach().cpu().numpy()
    diff = np.abs(adj_self - adj_mirror)
    plt.figure(figsize=(8, 8))
    plt.imshow(diff, cmap='Purples')
    plt.savefig("12_tom_divergence.png")
    plt.close()

def anomaly_detector(model):
    print("\n--- Anomaly Detection ---")
    corrupt_text = "True without falsehood certain and most banana"
    # Safety filter to ensure we don't pass empty tensors if chars are missing
    input_ids = [char_to_ix.get(c, 0) for c in corrupt_text if c in char_to_ix]
    if not input_ids: input_ids = [0]
    
    input_tensor = torch.tensor(input_ids, dtype=torch.long).to(DEVICE)
    hidden, prev_sym = None, None
    anomalies = []
    with torch.no_grad():
        for i in range(len(input_tensor) - 1):
            x = input_tensor[i].view(1, 1)
            out = model(x, hidden, training=False)
            hidden = out[1]
            ethics_cost = out[-1]
            anomalies.append(ethics_cost.item())
    plt.figure(figsize=(10, 4))
    plt.plot(anomalies, color='crimson', marker='o')
    plt.savefig("13_anomaly_detection.png")
    plt.close()

def extract_logic_rules_hybrid(model):
    print("\n--- Logic Rules ---")
    adj = model.vq.adjacency_energy.detach().cpu().numpy()
    probs = np.exp(-adj) / np.sum(np.exp(-adj), axis=1, keepdims=True)
    
    print(f"{'FROM':<6} | {'TO':<6} | {'PROBABILITY'}")
    print("-" * 35)
    
    count = 0
    for i in range(CONFIG["symbols"]):
        best_j = np.argmax(probs[i])
        prob = probs[i, best_j]
        if prob > 0.02: 
            print(f"S_{i:<4} -> S_{best_j:<4} | {prob:.4f}")
            count += 1
            if count > 10: break
            
    print(f"\n[Dynamic] Observed Transitions:")
    rule_book = defaultdict(list)
    hidden, prev_sym = None, None
    model.eval()
    scan_len = min(len(data_tensor), 1000)
    with torch.no_grad():
        for i in range(scan_len):
            x = data_tensor[i].view(1, 1)
            out = model(x, hidden, training=False)
            hidden = out[1]
            sym_idx = hidden[7]
            if prev_sym is not None and sym_idx is not None:
                rule_book[(prev_sym.item(), sym_idx.item())].append(1)
            prev_sym = sym_idx
    sorted_rules = sorted(rule_book.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"{'FROM':<6} | {'TO':<6} | {'COUNT'}")
    print("-" * 35)
    for (src, dst), hits in sorted_rules[:8]:
        print(f"S_{src:<4} -> S_{dst:<4} | {len(hits)}")

def dream_mode(model):
    print("\n--- 🌙 Dream Mode ---")
    model.eval()
    energy_matrix = model.vq.adjacency_energy.detach().cpu()
    curr = 0
    out = "T"
    for _ in range(50):
        logits = -energy_matrix[curr]
        probs = F.softmax(logits, dim=-1).numpy()
        next_idx = np.random.choice(len(probs), p=probs)
        z_flat = model.vq.codebook[next_idx].unsqueeze(0)
        half_dim = z_flat.shape[-1] // 2
        z_complex = torch.complex(z_flat[..., :half_dim], z_flat[..., half_dim:])
        q_logits = model.quantum_layer(z_complex.to(DEVICE))
        char_idx = torch.argmax(model.decoder(q_logits)).item()
        out += ix_to_char[char_idx]
        curr = next_idx
    print(f"Dream: {out}\n")

def injection_test(model):
    print("\n--- Introspection Injection ---")
    model.eval()
    x = torch.tensor([[0]], device=DEVICE)
    out_base = model(x, None, training=False)
    kl_base = out_base[4]
    inject = torch.complex(torch.randn(1, 64).to(DEVICE)*2.0, torch.randn(1, 64).to(DEVICE)*2.0)
    try:
        out_inj = model(x, None, injected_thought=inject, training=False)
        kl_inj = out_inj[4]
        print(f"Normal KL: {kl_base.item():.4f} | Injected: {kl_inj.item():.4f}")
    except RuntimeError:
        print("System Shock (Expected).")

def create_dashboard(model):
    if not PLOTLY_AVAILABLE: return
    print("\n--- Interactive Dashboard ---")
    model.eval()
    reals, imags = [], []
    hidden = None
    scan_len = min(len(data_tensor), 500)
    with torch.no_grad():
        for i in range(scan_len):
            x = data_tensor[i].view(1, 1)
            _, hidden, *args = model(x, hidden, training=False)
            z = hidden[0].cpu().squeeze()
            reals.append(z.real.mean().item())
            imags.append(z.imag.mean().item())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reals, y=imags, mode='lines+markers', name='Trajectory'))
    fig.write_html("dashboard.html")
    print("Saved dashboard.html")

def generate_text(model, start_str="The", length=300, temperature=0.7):
    print(f"\n--- 🧠 Generation (Prompt: '{start_str}') ---")
    model.eval()
    input_ids = [char_to_ix.get(c, 0) for c in start_str if c in char_to_ix]
    if not input_ids: input_ids = [0]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).to(DEVICE)
    hidden = None
    with torch.no_grad():
        for i in range(len(input_tensor) - 1):
            _, hidden, *args = model(input_tensor[i].view(1, 1), hidden, training=False)
        curr_input = input_tensor[-1].view(1, 1)
        generated = start_str
        for _ in range(length):
            logits, hidden, *args = model(curr_input, hidden, training=False)
            probs = F.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            char = ix_to_char[next_idx]
            generated += char
            curr_input = torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)
    print(f"Generated: {generated}\n")

# ==========================================
# 9. Main Execution
# ==========================================
if __name__ == "__main__":
    trained_model = train()
    
    # Core
    visualize_all(trained_model)
    
    # Advanced
    advanced_phase_analysis(trained_model)
    create_dashboard(trained_model)
    plot_attention_dynamics(trained_model)
    plot_neural_state_eeg(trained_model)
    plot_symbol_utilization(trained_model)
    plot_ponder_profile(trained_model)
    plot_semantic_drift(trained_model)
    
    # v54 Features
    plot_homeostasis(trained_model)
    plot_theory_of_mind(trained_model)
    
    # Logic & Dreams
    anomaly_detector(trained_model)
    extract_logic_rules_hybrid(trained_model)
    dream_mode(trained_model)
    injection_test(trained_model)
    
    # Gen
    generate_text(trained_model, start_str="The ", length=500)
    
    save_name = "sacrsn_v54_pocket.pth"
    torch.save(trained_model.state_dict(), save_name)
    print(f"\n--- Saved to {save_name} ---")
    
    # Auto-Download (Colab)
    try:
        from google.colab import files
        print("Downloading artifacts...")
        files.download(save_name)
        files.download("1_semantic_topology.png")
        files.download("2_cognitive_mri.png")
        files.download("3_grounding.png")
        files.download("4_phase_scatter.png")
        files.download("5_phase_coherence.png")
        files.download("5_phase_diff_sequence.png")
        files.download("6_attention_eeg.png")
        files.download("7_symbol_spectrum.png")
        files.download("8_neural_state_eeg.png")
        files.download("9_ponder_profile.png")
        files.download("10_semantic_drift.png")
        files.download("11_homeostasis.png")
        files.download("12_tom_divergence.png")
        files.download("13_anomaly_detection.png")
        if PLOTLY_AVAILABLE: files.download("dashboard.html")
    except ImportError:
        pass
    except Exception as e:
        print(f"Colab download failed: {e}")
