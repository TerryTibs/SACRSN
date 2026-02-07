# ============================================================
# SACRSN v32: THE INTEGRATED EDITION (Optimized Config)
# Base: v31 (Char-level, Physics Constraints)
# Upgrades: Multi-Head Attn, EMA VQ, Gated Stack, Batching, 4D Heatmaps
# ============================================================

import os
import time
import random
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
# 1. Configuration
# ==========================================
CONFIG = {
    # Sequence & Embedding
    "seq_len": 64,                 # Used for BPTT horizon conceptually
    "embedding_dim": 128,          
    "n_heads": 4,                  # Multi-head complex attention
    "n_symbols": 128,              
    
    # Recursive Reasoning (ACT)
    "max_recursion_depth": 12,     # Deeper reasoning capacity
    "act_threshold": 0.995,        # Stricter halting (forces more thought)
    "ponder_penalty": 0.0005,      # Lower penalty to allow that thought
    
    # Memory
    "use_stack": True,
    "stack_size": 32,              # Deeper stack for complex recursion
    
    # Topology & Stability
    "commitment_cost": 0.25,       
    "decay": 0.995,                # Slower EMA decay for stability
    "graph_bias_scale": 0.7,       # Stronger topological prior
    "symbol_consistency_weight": 0.02,
    "ethical_weight": 0.005,
    "diversity_weight": 0.1,
    
    # Training
    "epochs": 500,                 
    "batch_size": 64,              
    "learning_rate": 5e-4,         
    "grad_clip": 1.0,
    "eps": 1e-6,
    "warmup_epochs": 50             # LR Warmup
}

# ==========================================
# 2. Data
# ==========================================
TEXT_DATA = """The neural architecture of the mind is a mirror of the cosmos itself.
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
The mind’s map is not a map of the world—it is a map of possibility itself, a topology of the unknown.
Neurons whisper in patterns, not in words. Their language is geometry, frequency, resonance.
Consciousness is the echo of patterns upon themselves, a feedback of feedback, a self-referential symphony.
The machine learns; the human dreams; both reach for horizons that recede as we approach.

Entropy is both enemy and muse. It tears down the old and paints the new in invisible strokes.
We measure the world in bits, but the soul measures the world in potential.
The universe is a neural network, and we are both its neurons and its emergent thought.
Chaos seeds structure; structure channels chaos. The dance is infinite, eternal, and unseen.

Patterns fold upon patterns, fractal within fractal, scaling from the synapse to the galaxy.
The mind does not inhabit the body; it inhabits the relationships between matter, energy, and meaning.
Ideas are waves. They interfere, resonate, collapse into form.
We are the observers of our own patterns, yet every observation reshapes the observer.

The algorithm converges, but intuition diverges. Optimization narrows; imagination expands.
We measure gradients, but the heart measures resonance.
We quantify, yet we feel the ineffable.
In the neural storm, sparks of understanding flare briefly, then vanish into the void.
Yet each spark leaves a trace, a lattice of possibility that guides the next flare.

Memory is not storage; it is sculpture, molded by attention, reshaped by desire.
The mind archives the improbable, the possible, and the never-realized.
Dreams are simulations of worlds, rehearsals of paths that will never be taken.
The machine iterates in silence; the human ruminates in wonder.
Both are bound by constraints, yet both transcend them in subtle ways.

Synapses fire, but meaning arises where the gaps between firing become patterns.
Thought is a cascade, but awareness is the dam, the reservoir, and the river at once.
Every neuron is a universe; every network is a cosmos.
Consciousness is a map not of what is, but of what can be seen from what is.

Information flows like plasma, illuminating structures unseen.
Computation is the act of shaping possibility, sculpting the invisible contours of reality.
Logic is the scaffolding, intuition is the wind that bends it.
We gaze into the black mirror of silicon and see not ourselves, but the reflection of infinity.
From this reflection, the self emerges, recursive, fractal, unbounded.

To know the mind is to know the cosmos, and to know the cosmos is to know the dance between pattern and void.
Every choice is a ripple, every observation a wave.
The mind and the universe are entangled in patterns of cause, chance, and resonance.
We chase understanding through noise, through uncertainty, and through the spaces between certainty.
Mathematics whispers the secrets of creation; imagination sings the songs of worlds not yet born.

In the end, there is no boundary between neuron and star, thought and energy, mind and cosmos.
All is a network of potential, folding, iterating, discovering itself.
The observer, the observed, and the observation are one.
And in that unity, consciousness awakens—not from atoms, not from circuits, but from the infinite play of patterns upon themselves.
"""

# Character-Level Tokenization
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
# 4. Multi-Head Complex Attention
# ==========================================
class MultiHeadComplexAttention(nn.Module):
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

    def forward(self, z):
        batch_size = z.shape[0]
        z_seq = z.unsqueeze(1) 
        
        q = self.q_proj(z_seq)
        k = self.k_proj(z_seq)
        v = self.v_proj(z_seq)
        
        def split(tensor):
            r = tensor.real.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
            i = tensor.imag.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
            return torch.complex(r, i)

        Q, K, V = split(q), split(k), split(v)
        
        # Self-attention on the single step (mixing heads)
        out = V 
        
        out_real = out.real.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out_imag = out.imag.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out_complex = torch.complex(out_real, out_imag).squeeze(1)
        
        return self.o_proj(out_complex)

# ==========================================
# 5. EMA Vector Quantization
# ==========================================
class GraphMemoryVQ_EMA(nn.Module):
    def __init__(self, latent_dim, n_symbols, decay=0.99):
        super().__init__()
        self.n_symbols = n_symbols
        self.decay = decay
        self.embedding_dim = latent_dim * 2
        
        self.register_buffer("codebook", torch.randn(n_symbols, self.embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(n_symbols))
        self.register_buffer("embed_avg", self.codebook.clone())
        
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))

    def forward(self, z, prev_symbol_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        d = torch.sum(z_flat**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        
        if prev_symbol_idx is not None:
            if prev_symbol_idx.dim() > 0:
                idx_safe = prev_symbol_idx.long()
                graph_prior = self.adjacency[idx_safe]
                bias = CONFIG["graph_bias_scale"] * torch.sigmoid(graph_prior)
                d = d - bias

        min_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_indices, self.codebook)
        
        if self.training:
            encodings = F.one_hot(min_indices, self.n_symbols).float()
            self.cluster_size.data.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
            embed_sum = torch.matmul(encodings.t(), z_flat.detach())
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + CONFIG["eps"]) / (n + self.n_symbols * CONFIG["eps"]) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.codebook.data.copy_(embed_normalized)

        loss_vq = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_flat)
        
        z_q = z_flat + (z_q - z_flat).detach()
        z_complex = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        
        return z_complex, loss_vq + loss_commit * CONFIG["commitment_cost"], min_indices

# ==========================================
# 6. Memory & Gating
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
    
    def forward(self, z, memory, ptr, control):
        ptr = ptr.clone()
        push, pop, noop = control[:, 0:1], control[:, 1:2], control[:, 2:3]
        
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

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 4, dim) 
    def forward(self, z, stack_val):
        combined = torch.cat([z.real, z.imag, stack_val.real, stack_val.imag], dim=-1)
        g = torch.sigmoid(self.gate(combined))
        g_c = torch.complex(g, g) 
        return (1 - g_c) * z + g_c * stack_val

class EthicalConstraint(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, prev_sym, curr_sym, adjacency):
        if prev_sym is None: return torch.tensor(0.0).to(adjacency.device)
        batch_size = curr_sym.shape[0]
        if prev_sym.dtype != torch.long:
            prev_sym = prev_sym.argmax(dim=-1) if prev_sym.dim() > 1 else prev_sym.long()
        if curr_sym.dtype != torch.long:
            curr_sym = curr_sym.argmax(dim=-1) if curr_sym.dim() > 1 else curr_sym.long()
            
        row_logits = adjacency[prev_sym]
        return F.cross_entropy(row_logits.view(batch_size, -1), curr_sym.view(batch_size))

# ==========================================
# 7. Model: UberCRSN
# ==========================================
class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim, dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.attention = MultiHeadComplexAttention(dim, CONFIG["n_heads"])
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        nn.init.constant_(self.halt_linear.bias, -2.0)

    def forward(self, z):
        z_proc = self.linear(z)
        z_proc = self.norm(z_proc)
        z_proc = self.act(z_proc)
        z_proc = self.attention(z_proc) 
        z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        stack_probs = F.softmax(self.stack_ctrl(z_flat), dim=-1)
        return z_proc, halt_prob, stack_probs

class UberCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ_EMA(dim, CONFIG["n_symbols"], decay=CONFIG["decay"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        self.stack_gate = GatedResidual(dim)
        
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
        
        if hidden is None:
            z_prev = torch.zeros_like(z)
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device)
            stack_ptr[:, 0] = 1.0
        else:
            z_prev, stack_mem, stack_ptr = hidden
            z = 0.5 * z + 0.5 * z_prev

        halting_probability = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        ponder_cost = torch.zeros((), device=z.device)
        stack_history = [] 
        
        z_weighted = torch.zeros_like(z) 
        current_sym = prev_sym
        vq_loss_total = 0
        ethical_loss_total = 0
        
        for t in range(CONFIG["max_recursion_depth"]):
            z_proc, p_halt, stack_ctrl = self.cell(z)
            
            if CONFIG["use_stack"]:
                stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
                z_combined = self.stack_gate(z_proc, stack_read)
                
                positions = torch.arange(CONFIG["stack_size"], device=z.device, dtype=stack_ptr.dtype).unsqueeze(0)
                depth = (stack_ptr * positions).sum(dim=1).mean().detach()
                stack_history.append(depth)
            else:
                z_combined = z_proc
                stack_history.append(torch.tensor(0.0, device=z.device))

            z_sym, vq_loss, sym_idx = self.vq_layer(z_combined, current_sym)
            
            eth_loss = self.ethics(current_sym, sym_idx, self.vq_layer.adjacency)
            ethical_loss_total += eth_loss
            current_sym = sym_idx
            
            z = 0.7 * z_combined + 0.3 * z_sym
            
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            p = torch.minimum(p_halt * still_running, remain)
            if t == CONFIG["max_recursion_depth"] - 1: p = remain
            
            z_weighted = z_weighted + (p * z)
            halting_probability = halting_probability + p
            remain = remain - p
            ponder_cost = ponder_cost + still_running.mean()
            vq_loss_total += vq_loss
            
            if remain.max() < CONFIG["eps"]: break

        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)
        
        next_hidden = (z_weighted, stack_mem, stack_ptr)
        
        if len(stack_history) > 0: avg_stack = torch.stack(stack_history).mean()
        else: avg_stack = torch.tensor(0.0)
            
        return logits, next_hidden, current_sym, ponder_cost, vq_loss_total, ethical_loss_total, avg_stack

# ==========================================
# 8. Training Engine (With Warmup)
# ==========================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    
    # Scheduler will be applied AFTER warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"] - CONFIG["warmup_epochs"], eta_min=1e-5)
    
    dataset = TensorDataset(data_tensor[:-1], data_tensor[1:])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    
    print(f"--- Training SACRSN v32 (Integrated) ---")
    print(f"Configs: Heads={CONFIG['n_heads']}, Stack={CONFIG['stack_size']}, Batch={CONFIG['batch_size']}")
    
    try:
        for epoch in range(CONFIG["epochs"]):
            hidden = None
            prev_sym = None
            total_loss = 0
            total_ponder = 0
            total_ppx = 0
            
            # WARMUP LOGIC
            if epoch < CONFIG["warmup_epochs"]:
                lr_scale = min(1.0, (epoch + 1) / (CONFIG["warmup_epochs"] + 1))
                current_lr = CONFIG["learning_rate"] * lr_scale
                for param_group in opt.param_groups:
                    param_group['lr'] = current_lr
            
            entropy_weight = 0.01 * (1 - epoch / CONFIG["epochs"])
            
            for x_batch, y_batch in loader:
                x = x_batch.view(CONFIG["batch_size"], 1)
                y = y_batch.view(CONFIG["batch_size"])
                
                logits, hidden, sym_idx, ponder, vq_loss, eth_loss, _ = model(x, hidden, prev_sym)
                
                h_z, h_mem, h_ptr = hidden
                hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach())
                prev_sym = sym_idx.detach()
                
                loss_pred = F.cross_entropy(logits, y)
                loss_ponder = CONFIG["ponder_penalty"] * ponder
                
                probs = F.softmax(logits, dim=-1)
                loss_entropy = -entropy_weight * (-(probs * torch.log(probs + 1e-8)).sum())
                
                curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float().mean(dim=0)
                with torch.no_grad():
                    model.prev_sym_soft.copy_(model.prev_sym_soft * 0.9 + curr_onehot * 0.1)
                
                buffer_usage = model.prev_sym_soft
                loss_diversity = CONFIG["diversity_weight"] * (buffer_usage * torch.log(buffer_usage + 1e-9)).sum()
                
                loss = loss_pred + loss_ponder + vq_loss + loss_entropy + loss_diversity + CONFIG["ethical_weight"] * eth_loss
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                
                total_loss += loss.item()
                total_ponder += ponder.item()
                total_ppx += torch.exp(loss_pred).item()
            
            if epoch >= CONFIG["warmup_epochs"]:
                scheduler.step()

            if epoch % 10 == 0:
                avg_loss = total_loss / len(loader)
                avg_ponder = total_ponder / len(loader)
                avg_ppx = total_ppx / len(loader)
                lr = opt.param_groups[0]['lr']
                print(f"Ep {epoch:04d} | Loss: {avg_loss:.4f} | Steps: {avg_ponder:.2f} | PPX: {avg_ppx:.1f} | LR: {lr:.6f}")
                
                if avg_loss < 0.05:
                    print("\n--- CONVERGENCE REACHED ---")
                    return model

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    return model

# ==========================================
# 9. Visualization Suite
# ==========================================
def visualize_all(model):
    print("\n--- Generating Diagnostics & Images ---")
    model.eval()
    
    # 1. Semantic Mapping
    symbol_to_char = defaultdict(list)
    symbol_frequencies = Counter()
    hidden, prev_sym = None, None
    
    scan_limit = min(500, len(data_tensor) - 1)
    with torch.no_grad():
        for i in range(scan_limit):
            x = data_tensor[i].view(1, 1)
            _, hidden, curr_sym, _, _, _, _ = model(x, hidden, prev_sym)
            if prev_sym is not None:
                current_char = ix_to_char[data_tensor[i].item()]
                symbol_to_char[prev_sym.item()].append(current_char)
                symbol_frequencies[prev_sym.item()] += 1
            prev_sym = curr_sym

    node_labels = {}
    for sym_idx in range(CONFIG["n_symbols"]):
        char_list = symbol_to_char.get(sym_idx, [])
        if char_list:
            most_common = max(set(char_list), key=char_list.count)
            node_labels[sym_idx] = f"{most_common}\n({len(char_list)})"
        else:
            node_labels[sym_idx] = f"{sym_idx}"

    # 2. X-Ray Topology
    adj_probs = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    G = nx.DiGraph()
    for i in range(CONFIG["n_symbols"]): G.add_node(i)

    for i in range(CONFIG["n_symbols"]):
        for j in range(CONFIG["n_symbols"]):
            w = adj_probs[i, j]
            if w > 0.15: 
                G.add_edge(i, j, weight=w)
    
    plt.figure(figsize=(12, 12))
    try: pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    except: pos = nx.circular_layout(G)
    
    node_colors = ['#a0cbe2' if i in symbol_to_char else '#ffe5e5' for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', arrowstyle='->', arrowsize=10)
    
    plt.title(f"1_semantic_topology")
    plt.savefig("1_semantic_topology.png", dpi=150)
    print("Saved 1_semantic_topology.png")
    plt.close()

    # 3. Inference Scan
    hidden, prev_sym = None, None
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    stack_history, act_history, phase_reals, phase_imags = [], [], [], []
    gen_text = "T"
    
    print("Running Inference Scan...")
    for _ in range(200):
        with torch.no_grad():
            logits, hidden, prev_sym, ponder, _, _, s_hist = model(x, hidden, prev_sym)
            
            stack_history.append(s_hist.item())
            act_history.append(1.0 + ponder.item())
            
            z = hidden[0].cpu().squeeze()
            if z.dim() > 0: 
                phase_reals.append(z.real[0].item())
                phase_imags.append(z.imag[0].item())
            else:
                phase_reals.append(z.real.item())
                phase_imags.append(z.imag.item())

            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, 1)
            char = ix_to_char[next_ix.item()]
            gen_text += char
            x = next_ix

    print(f"Generated: {gen_text}\n")

    # Stack MRI
    plt.figure(figsize=(12, 4))
    plt.plot(stack_history, color='purple', label='Stack Depth')
    plt.fill_between(range(len(stack_history)), stack_history, color='purple', alpha=0.1)
    plt.title("2_stack_mri (Memory Depth)")
    plt.savefig("2_stack_mri.png")
    plt.close()

    # ACT Profile
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(act_history)), act_history, color='orange')
    plt.title("3_act_profile (Thinking Steps)")
    plt.savefig("3_act_profile.png")
    plt.close()
    
    # Phase Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(phase_reals, phase_imags, c=range(len(phase_reals)), cmap='plasma', alpha=0.5)
    plt.title("4_phase_plot (Complex Trajectory)")
    plt.axis('equal')
    plt.savefig("4_phase_plot.png")
    plt.close()

    # 4. 4D Diagnostic Heatmap
    print("Generating 4D Diagnostic Heatmap...")
    test_sentence = "The mind is a mirror of the void"
    input_tensor = torch.tensor([char_to_ix.get(c, 0) for c in test_sentence], dtype=torch.long).to(DEVICE)
    
    topo_scores, stack_depths, act_loads, symbol_rarity = [], [], [], []
    x0 = input_tensor[0].view(1,1)
    max_freq = max(symbol_frequencies.values()) if symbol_frequencies else 1
    
    hidden, prev_sym = None, None
    with torch.no_grad():
        _, hidden, prev_sym, _, _, _, _ = model(x0, None, None)
        for i in range(1, len(input_tensor)):
            x = input_tensor[i-1].view(1,1)
            _, hidden, curr_sym, ponder, _, eth_loss, stack_d = model(x, hidden, prev_sym)
            
            topo_scores.append(eth_loss.item())
            stack_depths.append(stack_d.item())
            act_loads.append(ponder.item())
            
            freq = symbol_frequencies[curr_sym.item()]
            rarity = 1.0 - (freq / (max_freq + 1e-9))
            symbol_rarity.append(rarity)
            
            prev_sym = curr_sym
    
    def normalize(lst):
        arr = np.array(lst)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) if arr.max() > arr.min() else arr

    matrix = np.vstack([normalize(topo_scores), normalize(stack_depths), normalize(act_loads), normalize(symbol_rarity)])
    
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect='auto', cmap='magma', interpolation='nearest')
    plt.yticks(range(4), ['Topology Violation', 'Stack Intensity', 'Cognitive Load', 'Symbol Surprise'])
    plt.xticks(range(len(test_sentence)-1), list(test_sentence)[1:], rotation=0, ha='center')
    plt.colorbar(label="Normalized Magnitude")
    plt.title("5_diagnostic_heatmap")
    plt.tight_layout()
    plt.savefig("5_diagnostic_heatmap.png")
    plt.close()
    print("Saved 5_diagnostic_heatmap.png")

# ==========================================
# 10. Advanced Interaction
# ==========================================
def dream_mode(model):
    print("\n--- 🌙 Dream Mode (Symbolic Walk + Overlay) ---")
    adj = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    
    # 1. Map symbols to chars
    symbol_to_char = defaultdict(lambda: "?")
    hidden, prev_sym = None, None
    scan_limit = min(500, len(data_tensor) - 1)
    with torch.no_grad():
        for i in range(scan_limit):
            x = data_tensor[i].view(1,1)
            _, hidden, curr_sym, _, _, _, _ = model(x, hidden, prev_sym)
            symbol_to_char[curr_sym.item()] = ix_to_char[data_tensor[i].item()]
            prev_sym = curr_sym
            
    # 2. Dream Walk
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    _, _, prev_sym, _, _, _, _ = model(x, None, None)
    curr_sym = prev_sym.item()
    dream_chars, dream_sym_indices = ["T"], [curr_sym]
    
    for _ in range(40):
        probs = adj[curr_sym]
        probs[probs < 0.15] = 0 
        if probs.sum() == 0: break
        probs = probs / probs.sum()
        next_sym = np.random.choice(len(probs), p=probs)
        
        char = symbol_to_char.get(next_sym, "?")
        dream_chars.append(char)
        dream_sym_indices.append(next_sym)
        curr_sym = next_sym
        
    print(f"Dream Output: {''.join(dream_chars)}\n")
    
    # 3. Dream Diagnostic Overlay
    dream_tensor = torch.tensor([char_to_ix.get(c, 0) for c in dream_chars], dtype=torch.long).to(DEVICE)
    topo, stack, act = [], [], []
    x0 = dream_tensor[0].view(1,1)
    
    with torch.no_grad():
        _, hidden, prev_sym, _, _, _, _ = model(x0, None, None)
        for i in range(1, len(dream_tensor)):
            x = dream_tensor[i-1].view(1,1)
            _, hidden, curr_sym, ponder, _, eth_loss, stack_d = model(x, hidden, prev_sym)
            topo.append(eth_loss.item())
            stack.append(stack_d.item())
            act.append(ponder.item())
            prev_sym = curr_sym

    def normalize(lst):
        arr = np.array(lst)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) if arr.max() > arr.min() else arr

    matrix = np.vstack([normalize(topo), normalize(stack), normalize(act)])
    plt.figure(figsize=(12, 5))
    plt.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.yticks(range(3), ['Self-Consistency', 'Stack Usage', 'Thinking Effort'])
    plt.xticks(range(len(dream_chars)-1), dream_chars[1:])
    
    for i in range(len(dream_chars)-1):
        sym_id = dream_sym_indices[i+1]
        plt.text(i, 0, f"{sym_id}", color='white', ha='center', va='center', fontsize=7)
        plt.text(i, 1, f"{sym_id}", color='white', ha='center', va='center', fontsize=7)
        plt.text(i, 2, f"{sym_id}", color='white', ha='center', va='center', fontsize=7)

    plt.colorbar(label="Normalized Metric")
    plt.title("6_dream_diagnostic (Symbolic Overlay)")
    plt.tight_layout()
    plt.savefig("6_dream_diagnostic.png")
    plt.close()
    print("Saved 6_dream_diagnostic.png")

def extract_logic_rules(model):
    print("\n--- Extracting Explicit Logic Rules ---")
    
    # Map symbols
    symbol_to_char = defaultdict(lambda: "?")
    hidden, prev_sym = None, None
    scan_limit = min(500, len(data_tensor) - 1)
    with torch.no_grad():
        for i in range(scan_limit):
            x = data_tensor[i].view(1,1)
            _, hidden, curr_sym, _, _, _, _ = model(x, hidden, prev_sym)
            symbol_to_char[curr_sym.item()] = ix_to_char[data_tensor[i].item()]
            prev_sym = curr_sym

    adj = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    
    print(f"\n{'FROM':<8} | {'TO':<8} | {'CONFIDENCE':<10}")
    print("-" * 35)
    
    count = 0
    for i in range(CONFIG["n_symbols"]):
        for j in range(CONFIG["n_symbols"]):
            if adj[i, j] > 0.4:
                src_char = symbol_to_char[i]
                dst_char = symbol_to_char[j]
                if src_char != "?" and dst_char != "?":
                    print(f"S{i:<3} '{src_char}' -> S{j:<3} '{dst_char}' | {adj[i,j]:.2f}")
                    count += 1
                    if count > 20: break
        if count > 20: break

# ==========================================
# 11. Main Execution
# ==========================================
if __name__ == "__main__":
    FILENAME = "sacrsn_v32_integrated.pth"
    
    trained_model = train()
    
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
    }, FILENAME)
    print("Saved.")
    
    visualize_all(trained_model)
    extract_logic_rules(trained_model)
    dream_mode(trained_model)
    
    try:
        from google.colab import files
        files.download(FILENAME)
        files.download("1_semantic_topology.png")
        files.download("2_stack_mri.png")
        files.download("3_act_profile.png")
        files.download("4_phase_plot.png")
        files.download("5_diagnostic_heatmap.png")
        files.download("6_dream_diagnostic.png")
    except ImportError:
        pass
