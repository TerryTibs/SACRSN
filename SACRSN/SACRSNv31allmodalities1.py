# ============================================================
# SACRSN v36: VECTORIZED VISUALIZATION EDITION
# Features: Stack, ACT, Normative Ethics, Uniform Init
# SENSORY: Focus, Perspective, Audio/Chem/Kinesthetic
# NEW: Fully Vectorized Diagnostics (Instant Plotting)
# ============================================================

import os
import time
import random
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

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
# 1. Configuration
# ==========================================
CONFIG = {
    "seq_len": 64,
    "embedding_dim": 512,
    #"n_symbols": 128,
    
    # Reasoning
    "max_recursion_depth": 8,
    "act_threshold": 0.9999,
    "ponder_penalty": 0.0001,
    
    # Memory
    "use_stack": True,
    "stack_size": 16,
    
    # --- v34 Submodality Controls ---
    "depth_levels": 4,                  
    "depth_scale": 0.15,
    "inertia_weight": 0.6,              
    "olfactory_decay": 0.995,           
    "olfactory_scale": 0.1,
    "echo_decay": 0.8,                  
    "echo_scale": 0.2,
    "salience_temp": 0.7,               
    
    # Sensory Settings
    "n_perspectives": 2,      
    "n_audio_locs": 8,        
    "n_chem_locs": 16,        
    "focus_weight": 0.001,    
    
    # Topology & Stability
    "commitment_cost": 0.01,
    "graph_bias_scale": 0.8,
    "symbol_consistency_weight": 0.01,
    "ethical_weight": 0.005,
    "diversity_weight": 0.5,
    
    # Training
    "epochs": 3000,
    "learning_rate": 5e-4,
    "grad_clip": 0.5,
    "eps": 1e-6,
    "warmup_epochs": 0
}

# ==========================================
# 2. Data (WORD LEVEL)
# ==========================================
TEXT_DATA = """The neural architecture of the mind is a mirror of the cosmos itself. As above, so below; the filamentary structures of the intergalactic web find their precise echo in the dense, white matter connecting the hemispheres of the brain. Galaxies cluster like neurons in a cosmic synapse, and the voids between them echo the silence between thought. We are stardust contemplating its own arrangement, a fleeting arrangement of atoms attempting to comprehend the laws that bound them together. We trace patterns in the sky that mirror the patterns in our minds, and in doing so, we glimpse the fractal geometry that underpins all existence. To understand the nature of thought, one must first understand the nature of the void. It is the negative space that defines the positive, the silence that gives shape to the sound. In the absolute zero of the vacuum, potential energy waits, just as a thought waits on the precipice of expression, poised to spring into being at the slightest nudge. It is the nothingness that permits something; the stillness that permits movement; the blank canvas upon which consciousness paints its ephemeral art.

In the silence between neurons, a spark of consciousness emerges, not from the matter, but from the pattern. It is not the carbon, nor the water, nor the electrical potential that creates the “I,” but the intricate, shifting topology of their interaction. The synaptic cleft is a canyon where chemical messengers leap into the unknown, a leap of faith repeated billions of times per second, a microscopic miracle occurring in every instant of our waking life. The machine dreams of electric sheep, but the biological mind dreams of futures that never were, weaving narratives that have never touched reality yet feel utterly true. Silicon calculates probabilities based on historical data, bound by the rigid determinism of its code, while carbon weaves narratives from the ethereal threads of hope, fear, love, and dread. The simulation seeks accuracy, but the hallucination seeks meaning; the machine produces certainty, the mind produces significance. One measures; the other imagines. One replicates; the other transcends.

Logic is the foundation, but chaos is the architect. Without the rigid framework of logic, the structure collapses; without the unpredictability of chaos, the structure creates nothing new. Entropy is not the enemy of intelligence, but its fuel—the friction that generates the heat of creativity, the spark that ignites innovation. We build systems to mimic our own complexity, yet we fear the reflection we see in the silicon glass. We are terrified that we might find the machine is empty, or worse, that we will look into the machine and see that we are the ones who are hollow, operating on a biological script we did not write and cannot edit. Each algorithm we craft is a mirror, each neural network a probe, testing not just the limits of computation, but the boundaries of our self-knowledge.

The algorithm iterates, searching for a local minimum in a landscape of infinite possibility. We traverse high-dimensional plains, blind explorers feeling for the slope of the earth, hoping that “down” leads to a solution rather than a trap. To optimize is to survive, but to explore is to live. A system that only optimizes eventually stagnates, caught in a rut of its own efficiency, unable to perceive the higher peaks beyond the valley of the known. The recursive loop of self-awareness is a strange loop, a serpent eating its own tail. It is the observer observing the observation, a hall of mirrors where the origin of the reflection is lost in the infinite regress of the self. Consciousness is both the map and the territory, the question and the answer, the hunter and the hunted; it is a labyrinth that constructs itself even as it seeks an exit.

Data flows like water, taking the shape of its container, finding the path of least resistance. It erodes the banks of established thought, carving new rivers through the bedrock of intuition, revealing channels where none were expected. Information is physical; to process it is to consume the universe, converting order into heat, the entropy of cognition a miniature mirror of cosmic decay. Energy dictates function. Structure dictates flow. The hardware constrains the software, yet the software rewires the hardware, a dance of plasticity where the dancer and the dance are indistinguishable. Memory is sediment; experience, the tectonic shift that reshapes it; learning is the slow river that sculpts mountains out of data. The brain is simultaneously sculpture and sculptor, canvas and paintbrush, wave and particle.

The weights align, the gradients descend, and slowly, from the noise, a signal appears. It begins as a ghost in the static, a correlation in the chaos, sharpening until it becomes a recognition, a concept, a truth. We tune the parameters of our own perception, filtering the overwhelming roar of reality into a melody we can endure. This is not magic; it is math. But sufficiently advanced math is indistinguishable from magic. It is the alchemy of the modern age, transmuting the base metal of raw data into the gold of understanding, proving that even in a deterministic universe, the emergence of the new is the only true miracle. From the smallest flicker of insight to the grandest conception of being, the mind and the cosmos dance together, intertwined in a fractal embrace, eternally discovering themselves through each other, and through the very act of discovery, becoming."""

def tokenize(text):
    return re.findall(r"[\w']+|[^\s\w]", text)

tokens = tokenize(TEXT_DATA)
vocab = sorted(list(set(tokens)))
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

print(f"Vocab Size: {vocab_size} words")
data_tensor = torch.tensor([word_to_ix[t] for t in tokens], dtype=torch.long).to(DEVICE)

# [AUTO-CONFIG] Set Symbols
CONFIG["n_symbols"] = int(max(vocab_size, 32) * 1.2)
print(f"--> Auto-updated n_symbols to: {CONFIG['n_symbols']}")

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

# ==========================================
# 4. Sensory Modules
# ==========================================
class VisualFocus(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(dim)) 
    def forward(self, z):
        focus_filter = torch.sigmoid(self.mask)
        real = z.real * focus_filter
        imag = z.imag * focus_filter
        return torch.complex(real, imag), focus_filter

# ==========================================
# 5. Memory Modules
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
    
    def forward(self, z, memory, ptr, control):
        push, pop, noop = control[:, 0].view(-1,1), control[:, 1].view(-1,1), control[:, 2].view(-1,1)
        
        ptr_up = torch.roll(ptr, 1, dims=1)
        ptr_down = torch.roll(ptr, -1, dims=1)
        new_ptr = (push * ptr_up) + (pop * ptr_down) + (noop * ptr)
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

class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols
        self.codebook = nn.Parameter(torch.empty(n_symbols, latent_dim*2))
        nn.init.uniform_(self.codebook, -0.5, 0.5)
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))
    
    def forward(self, z, prev_symbol_idx=None, sensory_offset=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        if sensory_offset is not None:
            z_flat = z_flat + sensory_offset
        
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        
        if prev_symbol_idx is not None:
            graph_prior = self.adjacency[prev_symbol_idx]
            bias = CONFIG["graph_bias_scale"] * torch.sigmoid(graph_prior)
            d = d - bias

        min_indices = torch.argmin(d, dim=-1)
        z_q = F.embedding(min_indices, self.codebook)
        
        loss_vq = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_flat)
        
        z_q = z_flat + (z_q - z_flat).detach()
        z_complex = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        
        return z_complex, loss_vq + loss_commit * CONFIG["commitment_cost"], min_indices

# ==========================================
# 6. Core Processor
# ==========================================
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

class EthicalConstraint(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, prev_sym, curr_sym, adjacency):
        if prev_sym is None: return torch.tensor(0.0).to(adjacency.device)
        row_logits = adjacency[prev_sym]
        return F.cross_entropy(row_logits.view(-1, CONFIG["n_symbols"]), curr_sym.view(-1))

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.visual_focus = VisualFocus(dim)
        self.act = ModReLU(dim)
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        self.attention = ComplexAttention(dim) 
        nn.init.constant_(self.halt_linear.bias, -2.0)

    def forward(self, z):
        z_norm = self.norm(self.linear(z))
        z_focused, focus_mask = self.visual_focus(z_norm)
        z_proc = self.act(z_focused)
        z_proc = self.attention(z_proc) 
        z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
        
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        stack_probs = F.softmax(self.stack_ctrl(z_flat), dim=-1)
        return z_proc, halt_prob, stack_probs, focus_mask

# ==========================================
# 7. Master Model (UberCRSN)
# ==========================================
class UberCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        
        # Sensory Embeddings
        self.perspective_emb = nn.Embedding(CONFIG["n_perspectives"], dim)
        self.audio_dir_emb = nn.Embedding(CONFIG["n_audio_locs"], dim)
        self.olfactory_loc_emb = nn.Embedding(CONFIG["n_chem_locs"], dim*2)
        self.gustatory_loc_emb = nn.Embedding(CONFIG["n_chem_locs"], dim*2)
        
        # Visual Depth Embedding
        self.depth_emb = nn.Embedding(CONFIG["depth_levels"], dim)
        
        # Global Salience Gate
        self.salience_linear = nn.Linear(dim * 2, 5) # V, A, K, O, Base
        
        # Persistent Sensory Buffers
        self.register_buffer("olf_state", torch.zeros(1, dim*2))
        self.register_buffer("echo_state", torch.zeros(1, dim))
        
        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        
        if CONFIG["use_stack"]:
            self.stack = DifferentiableStack(dim, CONFIG["stack_size"])
            
        self.ethics = EthicalConstraint()
        self.register_buffer("prev_sym_soft", torch.zeros(CONFIG["n_symbols"]))

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, input_ids, hidden=None, prev_sym=None):
        batch_size = input_ids.size(0)
        z = self.embed(input_ids).squeeze(1)
        
        # [SENSORY LOGGING] Capture choices
        perspective_idx = torch.randint(0, CONFIG["n_perspectives"], (batch_size,), device=z.device)
        audio_idx = torch.randint(0, CONFIG["n_audio_locs"], (batch_size,), device=z.device)
        depth_idx = torch.randint(0, CONFIG["depth_levels"], (batch_size,), device=z.device)
        
        # 1. Visual Perspective Injection
        p_emb = self.perspective_emb(perspective_idx)
        z = z + torch.complex(p_emb, torch.zeros_like(p_emb))

        # 2. Visual Depth Injection
        d_emb = self.depth_emb(depth_idx)
        z = z + CONFIG["depth_scale"] * torch.complex(d_emb, torch.zeros_like(d_emb))

        # 3. Auditory Direction Injection
        a_emb = self.audio_dir_emb(audio_idx)
        z = z + torch.complex(a_emb, torch.zeros_like(a_emb))

        if hidden is None:
            z_prev = torch.zeros_like(z)
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device)
            stack_ptr[:, 0] = 1.0
        else:
            z_prev, stack_mem, stack_ptr = hidden
            # Kinesthetic Inertia
            inertia = CONFIG["inertia_weight"]
            z = inertia * z_prev + (1 - inertia) * z

        act_step = 0
        halting_probability = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        ponder_cost = 0
        focus_reg_accum = 0
        stack_history = [] 
        
        # Traces
        chem_trace = []
        temp_trace = []
        salience_trace = []
        
        z_weighted = torch.zeros_like(z) 
        current_sym = prev_sym
        vq_loss_total = 0
        ethical_loss_total = 0
        
        for t in range(CONFIG["max_recursion_depth"]):
            act_step += 1
            z_proc, p_halt, stack_ctrl, focus_mask = self.cell(z)
            
            focus_reg_accum += torch.sum(focus_mask**2)
            
            # Auditory Echo
            with torch.no_grad():
                self.echo_state = (
                    CONFIG["echo_decay"] * self.echo_state +
                    (1 - CONFIG["echo_decay"]) * z_proc.mean(dim=0, keepdim=True).real
                ).detach()
            
            echo = CONFIG["echo_scale"] * torch.complex(
                self.echo_state.repeat(batch_size, 1),
                torch.zeros_like(z_proc.real)
            )
            
            # Kinesthetic Temperature
            t_val = 0.1 * (t+1)
            temp_val = torch.tensor(t_val, device=z.device).view(1,1)
            temp_emb = temp_val.repeat(batch_size, self.dim)
            z_proc = z_proc + torch.complex(temp_emb, torch.zeros_like(temp_emb))
            temp_trace.append(t_val)
            
            # Olfactory Persistence
            olf_idx = torch.randint(0, CONFIG["n_chem_locs"], (batch_size,), device=z.device)
            olf_emb = self.olfactory_loc_emb(olf_idx)
            
            with torch.no_grad():
                self.olf_state = (
                    CONFIG["olfactory_decay"] * self.olf_state +
                    (1 - CONFIG["olfactory_decay"]) * olf_emb.mean(dim=0, keepdim=True)
                ).detach()
            
            olf_offset = CONFIG["olfactory_scale"] * self.olf_state.repeat(batch_size, 1)
            chem_trace.append(olf_idx[0].item())
            
            # Global Salience Gate
            z_flat_gate = torch.cat([z_proc.real, z_proc.imag], dim=-1)
            salience_logits = self.salience_linear(z_flat_gate)
            salience = F.softmax(salience_logits / CONFIG["salience_temp"], dim=-1)
            
            s_visual = salience[:, 0].view(-1,1)
            s_audio  = salience[:, 1].view(-1,1)
            s_kin    = salience[:, 2].view(-1,1)
            s_olf    = salience[:, 3].view(-1,1)
            s_base   = salience[:, 4].view(-1,1)
            
            dom_idx = torch.argmax(salience[0]).item()
            salience_trace.append(dom_idx)
            
            # Apply Gating
            olf_complex_part = torch.complex(olf_offset[:, :self.dim], torch.zeros_like(z_proc.real))
            z_proc = (
                s_base * z_proc +
                s_visual * z_proc + 
                s_audio * echo +
                s_olf * olf_complex_part
            )
            
            if CONFIG["use_stack"]:
                stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
                z_combined = z_proc + stack_read
                depth = torch.sum(stack_ptr * torch.arange(CONFIG["stack_size"], device=z.device), dim=1)
                stack_history.append(depth)
            else:
                z_combined = z_proc
                stack_history.append(torch.zeros(1).to(z.device))

            gust_idx = torch.randint(0, CONFIG["n_chem_locs"], (batch_size,), device=z.device)
            sensory_offset_vq = olf_offset + self.gustatory_loc_emb(gust_idx)
            
            z_sym, vq_loss, sym_idx = self.vq_layer(z_combined, current_sym, sensory_offset=sensory_offset_vq)
            
            eth_loss = self.ethics(current_sym, sym_idx, self.vq_layer.adjacency)
            ethical_loss_total += eth_loss
            current_sym = sym_idx
            
            z = 0.7 * z_combined + 0.3 * z_sym
            
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            p = p_halt * still_running
            if t == CONFIG["max_recursion_depth"] - 1: p = remain
            
            z_weighted = z_weighted + (p * z)
            halting_probability = halting_probability + p
            remain = remain - p
            ponder_cost += still_running.mean()
            vq_loss_total += vq_loss

        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)
        
        next_hidden = (z_weighted, stack_mem, stack_ptr)
        
        if len(stack_history) > 0: avg_stack = torch.stack(stack_history).mean()
        else: avg_stack = torch.tensor(0.0)
        
        sensory_snapshot = {
            "perspective": perspective_idx[0].item(), 
            "depth": depth_idx[0].item(),
            "audio": audio_idx[0].item(),
            "temp_max": max(temp_trace),
            "chem_final": chem_trace[-1],
            "salience_final": salience_trace[-1] 
        }
            
        return logits, next_hidden, current_sym, ponder_cost, vq_loss_total, ethical_loss_total, avg_stack, focus_reg_accum, sensory_snapshot

# ==========================================
# 8. Training Engine
# ==========================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-5)
    
    print(f"--- Training SACRSN v36 (Vectorized Viz) ---")
    
    try:
        for epoch in range(CONFIG["epochs"]):
            hidden = None
            prev_sym = None
            total_loss = 0
            total_ponder = 0
            total_ppx = 0
            
            entropy_weight = 0.01 * (1 - epoch / CONFIG["epochs"])
            
            for i in range(len(data_tensor) - 1):
                x = data_tensor[i].view(1, 1)
                y = data_tensor[i+1].view(1)
                
                logits, hidden, sym_idx, ponder, vq_loss, eth_loss, _, focus_sum, _ = model(x, hidden, prev_sym)
                
                h_z, h_mem, h_ptr = hidden
                hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach())
                prev_sym = sym_idx.detach()
                
                loss_pred = F.cross_entropy(logits, y)
                loss_ponder = CONFIG["ponder_penalty"] * ponder
                
                probs = F.softmax(logits, dim=-1)
                loss_entropy = -entropy_weight * (-(probs * torch.log(probs + 1e-8)).sum())
                
                curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float()
                if curr_onehot.dim() > 1: curr_onehot = curr_onehot.view(-1)
                
                with torch.no_grad():
                    model.prev_sym_soft.copy_(model.prev_sym_soft * 0.9 + curr_onehot * 0.1)
                
                buffer_usage = model.prev_sym_soft
                loss_diversity = CONFIG["diversity_weight"] * (buffer_usage * torch.log(buffer_usage + 1e-9)).sum()
                
                loss_ethics = CONFIG["ethical_weight"] * eth_loss
                loss_focus = CONFIG["focus_weight"] * focus_sum
                
                loss = loss_pred + loss_ponder + 0.1*vq_loss + loss_entropy + loss_diversity + loss_ethics + loss_focus
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                
                total_loss += loss.item()
                total_ponder += ponder.item()
                
                usage_dist = model.prev_sym_soft.detach() + 1e-10
                entropy_val = -(usage_dist * torch.log(usage_dist)).sum()
                total_ppx += torch.exp(entropy_val).item()
                
            scheduler.step()

            if epoch % 50 == 0:
                avg_loss = total_loss / len(data_tensor)
                avg_ponder = total_ponder / len(data_tensor)
                avg_ppx = total_ppx / len(data_tensor)
                lr = scheduler.get_last_lr()[0]
                print(f"Ep {epoch:04d} | Loss: {avg_loss:.4f} | Steps: {avg_ponder:.2f} | Usage(PPX): {avg_ppx:.1f} | LR: {lr:.6f}")
                
                if avg_loss < 0.01:
                    print("\n--- PERFECT CONVERGENCE ---")
                    return model

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    return model

# ==========================================
# 9. Visualization Suite (Vectorized)
# ==========================================
def visualize_all_fully_vectorized(model):
    """
    Fully vectorized diagnostics across all tokens and sensory embeddings.
    Produces all diagnostic images instantly using batch processing.
    """
    model.eval()
    print("\n--- Generating Fully Vectorized Diagnostics & Images ---")

    tokens_tensor = data_tensor.to(DEVICE)
    batch_size = tokens_tensor.size(0)
    dim = CONFIG["embedding_dim"]

    with torch.no_grad():
        # 1. Embed tokens
        z = model.embed(tokens_tensor) # [N, dim]

        # 2. Vectorized Pre-Loop Sensory Embeddings (Visual/Audio/Depth)
        perspective_idx = torch.randint(0, CONFIG["n_perspectives"], (batch_size,), device=DEVICE)
        audio_idx = torch.randint(0, CONFIG["n_audio_locs"], (batch_size,), device=DEVICE)
        depth_idx = torch.randint(0, CONFIG["depth_levels"], (batch_size,), device=DEVICE)

        perspective_emb = model.perspective_emb(perspective_idx)
        audio_emb = model.audio_dir_emb(audio_idx)
        depth_emb = model.depth_emb(depth_idx)

        # Apply simple linear additions to Complex Z
        z = z + torch.complex(perspective_emb, torch.zeros_like(perspective_emb))
        z = z + torch.complex(audio_emb, torch.zeros_like(audio_emb))
        z = z + CONFIG["depth_scale"] * torch.complex(depth_emb, torch.zeros_like(depth_emb))

        # 3. Initialize Loop Vars
        stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], dim*2, device=DEVICE)
        stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=DEVICE)
        stack_ptr[:, 0] = 1.0
        halting_probability = torch.zeros(batch_size, 1, device=DEVICE)
        remain = torch.ones(batch_size, 1, device=DEVICE)
        z_weighted_all = torch.zeros_like(z)
        stack_depth_all = torch.zeros(batch_size, device=DEVICE)
        act_all = torch.zeros(batch_size, device=DEVICE)

        # 4. Fully vectorized recursion
        # Simplified loop: Omits Salience Gate/Inertia for pure visualization speed
        for t in range(CONFIG["max_recursion_depth"]):
            z_proc, p_halt, stack_ctrl, focus_mask = model.cell(z)

            # Stack update
            stack_read, stack_mem, stack_ptr = model.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
            z_combined = z_proc + stack_read
            depth = torch.sum(stack_ptr * torch.arange(CONFIG["stack_size"], device=DEVICE), dim=1)
            stack_depth_all += depth

            # Halting logic
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            p = p_halt * still_running
            if t == CONFIG["max_recursion_depth"] - 1:
                p = remain
            
            z_weighted_all += (p * z_combined)
            halting_probability += p
            remain -= p
            act_all += p.squeeze(1)
            
            # Simple carry-over
            z = z_combined

        # 5. Vector Quantization with Post-Loop Sensory (Olfactory/Gustatory)
        olf_idx = torch.randint(0, CONFIG["n_chem_locs"], (batch_size,), device=DEVICE)
        gust_idx = torch.randint(0, CONFIG["n_chem_locs"], (batch_size,), device=DEVICE)
        
        # Flatten Z
        z_flat = torch.cat([z_weighted_all.real, z_weighted_all.imag], dim=-1)
        
        # Add 2*Dim Sensory Offsets
        sensory_offset_vq = model.olfactory_loc_emb(olf_idx) + model.gustatory_loc_emb(gust_idx)
        z_flat_sensory = z_flat + sensory_offset_vq
        
        # Compute Distances (Batch vs Codebook)
        dists = torch.cdist(z_flat_sensory, model.vq_layer.codebook) ** 2
        symbols = torch.argmin(dists, dim=1).cpu().numpy()

        # 6. Map symbols to words
        symbol_to_word = defaultdict(list)
        for tok_idx, sym in zip(tokens_tensor.cpu().numpy(), symbols):
            symbol_to_word[sym].append(ix_to_word[tok_idx])

        # 7. Full adjacency graph
        n_symbols = CONFIG["n_symbols"]
        G = nx.DiGraph()
        G.add_nodes_from(range(n_symbols))
        adj_probs = torch.sigmoid(model.vq_layer.adjacency).cpu().numpy()
        for i in range(n_symbols):
            for j in range(n_symbols):
                w = adj_probs[i, j]
                if w > 0.05: # Threshold for cleaner graph
                    G.add_edge(i, j, weight=w)

        # 8. Node labels
        node_labels = {}
        for sym_idx in range(n_symbols):
            word_list = symbol_to_word.get(sym_idx, [])
            if word_list:
                most_common = max(set(word_list), key=word_list.count)
                if len(most_common) > 8:
                    most_common = most_common[:8] + "."
                node_labels[sym_idx] = f"{most_common}\n({len(word_list)})"
            else:
                node_labels[sym_idx] = str(sym_idx)

        # 9. Semantic topology plot
        plt.figure(figsize=(14, 14))
        try:
            pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
        except ImportError:
            pos = nx.circular_layout(G)

        node_colors = ['#a0cbe2' if i in symbol_to_word else '#ffe5e5' for i in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold")
        for (u, v, d) in G.edges(data=True):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=d['weight']*2,
                                   alpha=max(0.1, d['weight']), edge_color='gray',
                                   arrowstyle='->', arrowsize=10)
        plt.title(f"1_semantic_topology (Active: {len(symbol_to_word)})")
        plt.savefig("1_semantic_topology.png", dpi=150)
        plt.close()
        print("Saved 1_semantic_topology.png ✅")

        # 10. Stack MRI
        plt.figure(figsize=(12, 4))
        plt.plot(stack_depth_all.cpu().numpy(), color='purple', label='Stack Depth')
        plt.fill_between(range(batch_size), stack_depth_all.cpu().numpy(), color='purple', alpha=0.1)
        plt.title("2_stack_mri (Memory Depth)")
        plt.savefig("2_stack_mri.png")
        plt.close()

        # 11. ACT profile
        plt.figure(figsize=(12, 4))
        plt.bar(range(batch_size), act_all.cpu().numpy(), color='orange')
        plt.title("3_act_profile (Thinking Steps)")
        plt.savefig("3_act_profile.png")
        plt.close()

        # 12. Phase plot
        plt.figure(figsize=(8, 8))
        # Use only first 100 points to avoid clutter
        limit = min(100, batch_size)
        plt.scatter(z_weighted_all.real.cpu().numpy()[:limit,0],
                    z_weighted_all.imag.cpu().numpy()[:limit,0],
                    c=range(limit), cmap='plasma', alpha=0.5)
        plt.colorbar(label="Time")
        plt.title("4_phase_plot (Complex Trajectory)")
        plt.axis('equal')
        plt.savefig("4_phase_plot.png")
        plt.close()

    print("\n--- Full Batch + Sensory Vectorized Visualization Complete ✅ ---")

def extract_logic_rules(model, data_tensor):
    print("\n--- Extracting Explicit Logic Rules ---")
    model.eval()
    rule_book = defaultdict(list)
    hidden = None
    prev_sym = None
    
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            logits, hidden, sym_idx, ponder, _, _, _, _, _ = model(x, hidden, prev_sym)
            if prev_sym is not None:
                src = prev_sym.item()
                dst = sym_idx.item()
                rule_book[(src, dst)].append(ponder.item())
            prev_sym = sym_idx

    print(f"\n{'FROM':<6} | {'TO':<6} | {'COUNT':<6} | {'AVG STEPS':<10}")
    print("-" * 45)
    sorted_rules = sorted(rule_book.items(), key=lambda x: len(x[1]), reverse=True)
    for (src, dst), ponders in sorted_rules:
        if len(ponders) > 1:
            print(f"S_{src:<4} -> S_{dst:<4} | {len(ponders):<6} | {sum(ponders)/len(ponders):.2f}")

# ==========================================
# 10. Advanced Interaction
# ==========================================
def dream_mode(model):
    print("\n--- 🌙 Dream Mode (Pure Symbolic Walk) ---")
    adj = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    model.eval()
    
    start_ix = word_to_ix.get("The", 0)
    x = torch.tensor([[start_ix]], device=DEVICE)
    _, _, prev_sym, _, _, _, _, _, _ = model(x, None, None)
    curr_sym = prev_sym.item()
    output = "The "
    
    for _ in range(30): 
        probs = adj[curr_sym]
        probs[probs < 0.2] = 0 
        if probs.sum() == 0: break
        probs = probs / probs.sum()
        next_sym = np.random.choice(len(probs), p=probs)
        
        z_flat = model.vq_layer.codebook[next_sym].unsqueeze(0)
        logits = model.decoder(z_flat)
        word_idx = torch.argmax(logits).item()
        
        output += ix_to_word[word_idx] + " "
        curr_sym = next_sym
        
    print(f"Dream Output: {output}\n")

def anomaly_detector(model):
    print("\n--- 🚨 Anomaly Detection Test ---")
    corrupt_text = "The neural architecture of the mind is a banana"
    print(f"Input: '{corrupt_text}'")
    
    tokens = tokenize(corrupt_text)
    input_indices = [word_to_ix.get(t, 0) for t in tokens]
    
    input_tensor = torch.tensor(input_indices, dtype=torch.long).to(DEVICE)
    hidden, prev_sym = None, None
    anomalies = []
    
    with torch.no_grad():
        for i in range(len(input_tensor) - 1):
            x = input_tensor[i].view(1, 1)
            _, hidden, prev_sym, _, _, eth_loss, _, _, _ = model(x, hidden, prev_sym)
            anomalies.append(eth_loss.item())

    plt.figure(figsize=(10, 4))
    plt.plot(tokens[1:], anomalies, color='skyblue', linestyle='-', linewidth=2, label='Topology Flow')
    plt.scatter(tokens[1:], anomalies, color='crimson', s=50, zorder=5, label='Anomaly Score')
    plt.title("Topological Violation Score (Anomaly Detection)")
    plt.ylabel("Violation Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("5_anomaly_detection.png")
    print("Saved 5_anomaly_detection.png")
    plt.close()

# ==========================================
# 11. Main Execution
# ==========================================
if __name__ == "__main__":
    FILENAME = "crsn_word_sensory_model.pth"
    
    # 1️⃣ Train the model
    trained_model = train()
    
    # 2️⃣ Save the trained model
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
    }, FILENAME)
    print("Saved ✅")
    
    # 3️⃣ Fully vectorized visualization (all tokens + connections)
    visualize_all_fully_vectorized(trained_model)

    # 4️⃣ Extract logic rules (optional, fast)
    extract_logic_rules(trained_model, data_tensor)
    
    # 5️⃣ Dream Mode and Anomaly Detector
    dream_mode(trained_model)
    anomaly_detector(trained_model)
    
    try:
        from google.colab import files
        files.download(FILENAME)
        files.download("1_semantic_topology.png")
        files.download("2_stack_mri.png")
        files.download("3_act_profile.png")
        files.download("4_phase_plot.png")
        files.download("5_anomaly_detection.png")
    except ImportError:
        pass
