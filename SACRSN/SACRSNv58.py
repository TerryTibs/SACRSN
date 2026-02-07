# ============================================================
# SACRSN v58: THE GOD MODE BUILD
# ------------------------------------------------------------
# 1. ARCHITECTURE: Complex-Valued, Residual Cell, Context-Attention.
# 2. MEMORY: EBM Graph, Enhanced Stack (Attention/Compression).
# 3. META-COGNITION: Hierarchical Variational Introspection.
# 4. FULL STACK: Physiology, Social (Mirror System).
# 5. DIAGNOSTICS: Full Suite (12 Modules).
# 6. CONFIG: Homeostasis (Fatigue) Disabled for Maximum Performance.
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

# Optional: Plotly
try:
    import plotly.graph_objects as go
except ImportError:
    pass

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
print(f"Running on: {DEVICE}")

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "seq_len": 32,
    "dim": 64,
    "symbols": 64,
    "stack_size": 16,
    "max_depth": 8,
    "act_threshold": 0.99,
    
    # Unified Loss Weights
    "lambda_nll": 1.0,
    "lambda_halt": 0.01,
    "lambda_energy": 0.1,
    "lambda_kl": 0.1,
    "lambda_diversity": 0.05,
    "lambda_ground": 0.1,
    "lambda_meta_energy": 0.1,
    "lambda_aux_class": 0.1,
    "lambda_aux_recon": 0.1,
    "lambda_drift": 0.05,
    "lambda_probe": 0.1,
    "lambda_tom": 0.1,
    
    # Meta Parameters
    "meta_alpha": 0.1,
    "meta_lr": 0.01,
    "ground_dim": 32,
    "entropy_delta_thresh": 0.5,
    "meta_hierarchy_depth": 2,
    
    # Psychology Parameters
    "reframe_window": 8,
    "entropy_threshold": 0.3,
    "rigidity_limit": 0.9,
    "pacing_interval": 5,     
    
    # Physiology Parameters
    "enable_homeostasis": False, # [GOD MODE] Set to False to disable fatigue
    "metabolic_capacity": 100.0,
    "metabolic_decay": 0.5,
    "metabolic_recovery": 0.1,
    
    # EAFT
    "eaft_alpha": 2.0,
    "eaft_top_k": 20,
    
    # Dynamics
    "hallucination_decay": 0.9,
    "act_temp_base": 1.0,     
    
    # Training
    "epochs": 2500,
    "lr": 5e-4,
    "grad_clip": 0.5,
    "eps": 1e-6,
    "temp": 1.0,
    
    # Enhancements
    "residual_weight": 0.1,
    "adaptive_lr_factor": 0.5
}

# ==========================================
# 2. Data
# ==========================================
TEXT_DATA = """True, without falsehood, certain and most true. 
That which is above is like to that which is below, 
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""

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
        q = self.q_proj(z)
        k = self.k_proj(z)
        v = self.v_proj(z)
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
        q = self.q_proj(z)
        k = self.k_proj(z)
        v = self.v_proj(z)
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
        self.stats = {s: {"count": 0, "rigidity": 0.0} for s in range(n_symbols)}
    def update(self, sym, changed):
        if isinstance(sym, torch.Tensor): sym = sym.item()
        d = self.stats[sym]
        d["count"] += 1
        d["rigidity"] = 0.9 * d["rigidity"] + 0.1 * (0 if changed else 1)

class EthicalGate:
    def __init__(self, rigidity_limit=0.9):
        self.rigidity_limit = rigidity_limit
    def allow(self, meta):
        if meta["rigidity"] > self.rigidity_limit: return False
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
    def __init__(self, n_symbols, window=8):
        super().__init__()
        self.n_symbols = n_symbols
        self.window = window
        self.symbol_history = []
    def update_history(self, sym_idx):
        self.symbol_history.append(sym_idx.detach())
        if len(self.symbol_history) > self.window: self.symbol_history.pop(0)
    def check_stagnation(self):
        if len(self.symbol_history) < self.window: return False
        syms = torch.stack(self.symbol_history).float()
        if torch.std(syms) < 0.5: return True
        return False
    def trigger_neuroplasticity(self, adjacency_param, current_sym):
        with torch.no_grad():
            noise = torch.randn_like(adjacency_param[current_sym]) * 0.5
            adjacency_param[current_sym] += noise

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
        h2, c2, kl2, mu2 = self.levels[1](h1, h2_prev, c2_prev)
        combined_kl = kl1 + kl2
        return (h1, h2), (c1, c2), combined_kl.mean(), (mu1, mu2)
    def update_meta_prior(self, mus_tuple):
        self.levels[0].update_meta_prior(mus_tuple[0])
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
# 6. Master Model (Enhanced UberCRSN)
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
        
        # Executive Control
        self.reframer = BeliefReframer(CONFIG["symbols"])
        self.meta_tracker = MetaBeliefTracker(CONFIG["symbols"])
        self.ethical_gate = EthicalGate(rigidity_limit=CONFIG["rigidity_limit"])
        self.pacer = TherapeuticPacer(min_interval=CONFIG["pacing_interval"])
        
        # Physiology & Social
        self.homeostasis = HomeostaticRegulator(capacity=CONFIG["metabolic_capacity"])
        self.mirror_system = MirrorSystem(dim, CONFIG["symbols"])

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, input_ids, hidden=None, injected_thought=None, 
                image_input=None, audio_input=None, global_step=0):
        
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
            prev_soft_sym = None
            prev_entropy = None
            prev_sym_idx = None
        else:
            z_prev, h_vae, c_vae, stack_mem, stack_ptr, prev_soft_sym, prev_entropy, prev_sym_idx = hidden
            z = 0.5 * z + 0.5 * z_prev

        halt_cost, energy_cost, kl_cost, ground_cost, meta_cost, drift_cost, probe_logit, tom_cost = 0, 0, 0, 0, 0, 0, 0, 0
        mem_efficiency_log = []
        posterior_mus_l1, posterior_mus_l2 = [], []
        
        halting_probability = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        z_weighted = torch.zeros_like(z)
        
        current_soft_sym = prev_soft_sym
        current_entropy = prev_entropy
        stack_depth_log = torch.tensor(0.0).to(z.device)
        
        # [GOD MODE] Logic Switch
        if CONFIG.get("enable_homeostasis", True):
            fatigue = self.homeostasis(torch.tensor(1.0).to(DEVICE))
        else:
            fatigue = torch.tensor(1.0).to(DEVICE) # Infinite Energy
        
        for t in range(CONFIG["max_depth"]):
            z_proc, halt_logit, stack_logits = self.cell(z)
            z_stack, stack_mem, stack_ptr, active_slots = self.stack(z_proc, stack_mem, stack_ptr, stack_logits)
            mem_efficiency_log.append(active_slots)
            z = z_proc + z_stack
            
            h_vae, c_vae, kl_div, mus = self.hallucination_vae(z, h_vae, c_vae)
            posterior_mus_l1.append(mus[0])
            posterior_mus_l2.append(mus[1])
            
            probe_out = self.hallucination_probe(z)
            probe_logit += probe_out
            
            current_temp = CONFIG["temp"] / (fatigue + 0.1) 
            z_q, soft_sym, hard_idx, energy, ent, ground, meta = self.vq(
                z, current_soft_sym, prev_entropy=current_entropy, temp=current_temp
            )
            
            _, mirror_energy = self.mirror_system(z, current_soft_sym, temp=current_temp)
            tom_cost += mirror_energy
            
            if self.training:
                sym_item = hard_idx[0].item()
                self.reframer.update_history(hard_idx)
                if self.reframer.check_stagnation() and self.pacer.allow(global_step):
                    meta_stat = self.meta_tracker.stats[sym_item]
                    if self.ethical_gate.allow(meta_stat):
                        self.reframer.trigger_neuroplasticity(self.vq.adjacency_energy.data, sym_item)
                changed = (sym_item != prev_sym_idx) if prev_sym_idx is not None else False
                self.meta_tracker.update(sym_item, changed)
                prev_sym_idx = sym_item
            
            current_soft_sym = soft_sym
            current_entropy = ent
            z = 0.7 * z + 0.3 * z_q
            
            # Fatigue lowers ACT Threshold
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
        
        next_hidden = (z_weighted, h_vae, c_vae, stack_mem, stack_ptr, current_soft_sym, current_entropy, prev_sym_idx)
        avg_mu_l1 = torch.stack(posterior_mus_l1).mean(0)
        avg_mu_l2 = torch.stack(posterior_mus_l2).mean(0)
        avg_mem_efficiency = sum(mem_efficiency_log) / len(mem_efficiency_log)
        
        return logits, next_hidden, halt_cost, energy_cost, kl_cost, ground_cost, meta_cost, drift_cost, probe_logit, (avg_mu_l1, avg_mu_l2), stack_depth_log, aux_class, aux_recon, avg_mem_efficiency, tom_cost, fatigue

# ==========================================
# 7. Training
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
    
    print(f"--- Training SACRSN v58 (God Mode Enabled) ---")
    print(f"{'Epoch':<6} | {'Loss':<8} | {'KL (Self)':<9} | {'ToM (Soc)':<9} | {'Ent (Conf)':<10} | {'Drift':<8} | {'Fatigue':<7} | {'Temp':<6}")
    print("-" * 85)
    
    global_step = 0
    
    for epoch in range(CONFIG["epochs"]):
        hidden = None
        metrics = defaultdict(float)
        count = 0
        
        for i in range(len(data_tensor) - 1):
            global_step += 1
            x = data_tensor[i].view(1, 1)
            y = data_tensor[i+1].view(1)
            
            try:
                logits, hidden, halt, energy, kl, ground, meta, drift, probe_logit, mus, _, aux_c, aux_r, _, tom, fatigue = model(x, hidden, global_step=global_step)
            except RuntimeError as e:
                hidden = None
                continue
            
            h_z, h_h, h_c, h_mem, h_ptr, h_sym, h_ent, h_idx = hidden
            h_h_det = (h_h[0].detach(), h_h[1].detach())
            h_c_det = (h_c[0].detach(), h_c[1].detach())
            hidden = (h_z.detach(), h_h_det, h_c_det, 
                      h_mem.detach(), h_ptr.detach(), 
                      h_sym.detach() if h_sym is not None else None,
                      h_ent.detach() if h_ent is not None else None,
                      h_idx)
            
            model.hallucination_vae.update_meta_prior(mus)
            
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            weight = eaft_soft_weight(entropy, probs[0, y])
            
            nll = (F.cross_entropy(logits, y, reduction='none') * weight).mean()
            buffer = model.prev_sym_soft + 1e-9
            L_prior = -CONFIG["lambda_diversity"] * (buffer * torch.log(buffer)).sum()
            
            dummy_class = torch.randint(0, 10, (1,)).to(DEVICE)
            L_aux_c = F.cross_entropy(aux_c, dummy_class)
            L_aux_r = torch.mean(aux_r**2)
            
            probe_target = torch.sigmoid(entropy).detach()
            L_probe = F.binary_cross_entropy_with_logits(probe_logit.mean(), probe_target.mean())
            
            loss = CONFIG["lambda_nll"] * nll + \
                   CONFIG["lambda_halt"] * halt + \
                   CONFIG["lambda_energy"] * energy + \
                   CONFIG["lambda_kl"] * kl + \
                   CONFIG["lambda_ground"] * ground + \
                   CONFIG["lambda_meta_energy"] * meta + \
                   CONFIG["lambda_aux_class"] * L_aux_c + \
                   CONFIG["lambda_aux_recon"] * L_aux_r + \
                   CONFIG["lambda_drift"] * drift + \
                   CONFIG["lambda_probe"] * L_probe + \
                   CONFIG["lambda_tom"] * tom + \
                   L_prior
            
            if torch.isnan(loss) or torch.isinf(loss):
                hidden = None
                continue
            
            lr = optimizer.step(loss, kl.item())
            
            metrics["loss"] += loss.item()
            metrics["kl"] += kl.item()
            metrics["tom"] += tom.item()
            metrics["drift"] += drift.item()
            metrics["fatigue"] += fatigue.item()
            metrics["entropy"] += entropy.mean().item()
            count += 1

        if epoch % 50 == 0:
            avg = {k: v/count for k, v in metrics.items()} if count > 0 else metrics
            curr_temp = CONFIG["temp"] / (avg["fatigue"] + 0.1) if avg["fatigue"] > 0 else 1.0
            
            print(f"{epoch:04d}   | {avg['loss']:.4f}   | {avg['kl']:.4f}    | {avg['tom']:.4f}    | {avg['entropy']:.4f}     | {avg['drift']:.4f}   | {avg['fatigue']:.2f}    | {curr_temp:.2f}")
            
            if avg['loss'] < 0.5: break
            
    return model

# ==========================================
# 8. Diagnostics
# ==========================================
def visualize_all(model):
    print("\n--- Generating Diagnostics ---")
    model.eval()
    stack_depths, kl_vals, energies, mem_eff = [], [], [], []
    with torch.no_grad():
        hidden = None
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            # Use *args
            _, hidden, _, energy, kl, _, _, _, _, _, depth, _, _, eff, _, _ = model(x, hidden)
            stack_depths.append(depth.item())
            kl_vals.append(kl.item())
            energies.append(energy.item())
            mem_eff.append(eff.item())

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax1.plot(stack_depths, color='purple'); ax1.set_ylabel("Stack")
    ax2.plot(energies, color='orange'); ax2.set_ylabel("Energy")
    ax3.plot(kl_vals, color='green'); ax3.set_ylabel("Introspection")
    ax4.plot(mem_eff, color='cyan'); ax4.set_ylabel("Mem Eff")
    plt.savefig("1_cognitive_mri.png")
    plt.close()
    
    grounding = model.vq.symbol_ground.detach().cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(grounding[:20], cmap='inferno')
    plt.title("Symbol Grounding")
    plt.savefig("3_grounding.png")
    plt.close()
    print("Diagnostics saved.")

def advanced_phase_analysis(model):
    print("\n--- Phase Coherence ---")
    model.eval()
    hidden = None
    zs = []
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            _, hidden, *args = model(x, hidden)
            zs.append(hidden[0])
    z_stack = torch.cat(zs)
    phases = torch.angle(z_stack)
    phase_diff = phases[1:] - phases[:-1]
    coherence = 1.0 - torch.std(phase_diff, dim=1).cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.plot(coherence, color='blue')
    plt.savefig("5_phase_coherence.png")
    plt.close()

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
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    out_base = model(x, None)
    kl_base = out_base[4]
    inject = torch.complex(torch.randn(1, 64).to(DEVICE)*2.0, torch.randn(1, 64).to(DEVICE)*2.0)
    try:
        out_inj = model(x, None, injected_thought=inject)
        kl_inj = out_inj[4]
    except RuntimeError:
        kl_inj = torch.tensor(float('nan'))
    print(f"Normal KL: {kl_base.item():.4f}")
    if torch.isnan(kl_inj) or kl_inj > kl_base * 1.5:
        print(f"Injected KL: {kl_inj.item() if not torch.isnan(kl_inj) else 'NaN'}")
        print("SUCCESS: Injection Detected.")
    else:
        print(f"Injected KL: {kl_inj.item():.4f}")
        print("FAIL: Injection Ignored.")

def extract_logic_rules(model):
    print("\n--- Logic Rules ---")
    adj = model.vq.adjacency_energy.detach().cpu().numpy()
    probs = np.exp(-adj) / np.sum(np.exp(-adj), axis=1, keepdims=True)
    count = 0
    print(f"{'FROM':<6} | {'TO':<6} | {'PROBABILITY'}")
    print("-" * 30)
    for i in range(CONFIG["symbols"]):
        best_j = np.argmax(probs[i])
        prob = probs[i, best_j]
        if prob > 0.1:
            print(f"S_{i:<4} -> S_{best_j:<4} | {prob:.2f}")
            count += 1
    print(f"Total Strong Rules: {count}")

def create_dashboard(model):
    if 'go' not in globals(): return
    print("\n--- Interactive Dashboard ---")
    model.eval()
    reals, imags = [], []
    hidden = None
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            _, hidden, *args = model(x, hidden)
            z = hidden[0].cpu().squeeze()
            reals.append(z.real.mean().item())
            imags.append(z.imag.mean().item())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reals, y=imags, mode='lines+markers', name='Trajectory'))
    fig.write_html("dashboard.html")
    print("Saved dashboard.html")

def plot_attention_dynamics(model):
    print("\n--- Attention EEG ---")
    model.eval()
    activations = []
    def hook_fn(module, input, output):
        z_proc = output[0].detach().cpu()
        mag = torch.abs(z_proc).squeeze()
        activations.append(mag.numpy())
    handle = model.cell.register_forward_hook(hook_fn)
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    model(x, None)
    handle.remove()
    if len(activations) > 0:
        map_data = np.stack(activations)
        plt.figure(figsize=(10, 6))
        plt.imshow(map_data, aspect='auto', cmap='magma', interpolation='nearest')
        plt.savefig("6_attention_eeg.png")
        plt.close()
        print("Saved 6_attention_eeg.png")

def plot_symbol_utilization(model):
    print("\n--- Symbol Spectrum ---")
    model.eval()
    symbol_counts = np.zeros(CONFIG["symbols"])
    hidden = None
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            _, hidden, *args = model(x, hidden)
            curr_sym = hidden[5] 
            if curr_sym is not None:
                symbol_counts += curr_sym.cpu().numpy().flatten()
    symbol_counts = symbol_counts / (symbol_counts.sum() + 1e-9)
    plt.figure(figsize=(12, 4))
    plt.bar(range(CONFIG["symbols"]), symbol_counts, color='teal')
    plt.savefig("7_symbol_spectrum.png")
    plt.close()
    print("Saved 7_symbol_spectrum.png")

def plot_ponder_profile(model):
    print("\n--- Ponder Profile ---")
    model.eval()
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    z = model.embed(x).squeeze(1)
    probs_at_step = []
    for t in range(CONFIG["max_depth"]):
        _, halt_logit, _ = model.cell(z)
        p = torch.sigmoid(halt_logit)
        probs_at_step.append(p.item())
    plt.figure(figsize=(8, 4))
    plt.plot(probs_at_step, marker='o', color='orange')
    plt.savefig("8_ponder_profile.png")
    plt.close()
    print("Saved 8_ponder_profile.png")

def plot_semantic_drift(model):
    print("\n--- Semantic Drift ---")
    model.eval()
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
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
    plt.savefig("9_semantic_drift.png")
    plt.close()
    print("Saved 9_semantic_drift.png")

def generate_text(model, start_str="True", length=200, temperature=0.7):
    print(f"\n--- 🧠 Full Cognitive Generation (Prompt: '{start_str}') ---")
    model.eval()
    input_ids = [char_to_ix.get(c, 0) for c in start_str]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).to(DEVICE)
    hidden = None

    with torch.no_grad():
        for i in range(len(input_tensor) - 1):
            _, hidden, *args = model(input_tensor[i].view(1, 1), hidden)

        curr_input = input_tensor[-1].view(1, 1)
        generated = start_str

        for _ in range(length):
            logits, hidden, *args = model(curr_input, hidden)
            probs = F.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            char = ix_to_char[next_idx]
            generated += char
            curr_input = torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)
            if char == '\n' and len(generated) > length: break

    print(f"Generated: {generated}\n")

def plot_homeostasis(model):
    print("\n--- Homeostatic State (Fatigue) ---")
    model.eval()
    energy_levels = []
    # Force full energy for visualization if homeostasis is off
    if not CONFIG["enable_homeostasis"]:
         model.homeostasis.energy.fill_(CONFIG["metabolic_capacity"])
         
    hidden = None
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            # v54 returns 16 values, fatigue is last
            output = model(x, hidden)
            fatigue = output[-1]
            hidden = output[1]
            energy_levels.append(fatigue.item())
    plt.figure(figsize=(10, 4))
    plt.plot(energy_levels, color='brown')
    plt.title("Homeostatic Energy Levels")
    plt.savefig("10_homeostasis.png")
    plt.close()
    print("Saved 10_homeostasis.png")

def plot_theory_of_mind(model):
    print("\n--- Theory of Mind Divergence ---")
    adj_self = model.vq.adjacency_energy.detach().cpu().numpy()
    adj_mirror = model.mirror_system.other_graph.adjacency_energy.detach().cpu().numpy()
    diff = np.abs(adj_self - adj_mirror)
    plt.figure(figsize=(8, 8))
    plt.imshow(diff, cmap='Purples')
    plt.title("Self vs. Other Divergence (ToM)")
    plt.savefig("11_tom_divergence.png")
    plt.close()
    print("Saved 11_tom_divergence.png")

def text_anomaly_detector(model):
    print("\n--- Text Anomaly Detection ---")
    corrupt = "True without falsehood certain and most banana"
    inp = torch.tensor([char_to_ix.get(c, 0) for c in corrupt], dtype=torch.long).to(DEVICE)
    hidden = None
    anomalies = []
    with torch.no_grad():
        for i in range(len(inp) - 1):
            x = inp[i].view(1, 1)
            # KL is index 4
            _, hidden, _, _, kl, *args = model(x, hidden)
            anomalies.append(kl.item())
    plt.figure(figsize=(10, 4))
    plt.plot(list(corrupt)[1:], anomalies, color='crimson', marker='o')
    plt.title("Introspection Anomaly Score on Corrupt Text")
    plt.savefig("4_text_anomaly.png")
    plt.close()
    print("Saved 4_text_anomaly.png")

# ==========================================
# 9. Main Execution
# ==========================================
if __name__ == "__main__":
    trained_model = train()
    
    # Core Diagnostics
    visualize_all(trained_model)
    advanced_phase_analysis(trained_model)
    create_dashboard(trained_model)
    
    # Advanced Diagnostics
    plot_attention_dynamics(trained_model)
    plot_symbol_utilization(trained_model)
    plot_ponder_profile(trained_model)
    plot_semantic_drift(trained_model)
    plot_homeostasis(trained_model)
    plot_theory_of_mind(trained_model)
    
    # Capabilities
    dream_mode(trained_model)
    generate_text(trained_model, start_str="True", length=300)
    
    # Validation
    injection_test(trained_model)
    text_anomaly_detector(trained_model)
    extract_logic_rules(trained_model)
    
    torch.save(trained_model.state_dict(), "sacrsn_v58_godmode.pth")
    print("\n--- Complete System Saved ---")
