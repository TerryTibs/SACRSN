# ============================================================
# SACRSN v32: THE UNIFIED OBSERVABLE EDITION
# Base: SACRSN v31 (Original)
# Fixes: Batch-safe, CPU-safe, No In-place Ops, Dtypes, Shapes
# Preserved: Stack, ACT, Ethics, VQ, Dream, Anomaly, All Viz
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
from collections import defaultdict
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
    "seq_len": 64,
    "embedding_dim": 128,
    "n_symbols": 128,
    
    # Reasoning
    "max_recursion_depth": 8,
    "act_threshold": 0.9999,
    "ponder_penalty": 0.0001,
    
    # Memory
    "use_stack": True,
    "stack_size": 16,
    
    # Topology & Stability
    "commitment_cost": 0.01,
    "graph_bias_scale": 0.8,
    "symbol_consistency_weight": 0.01,
    "ethical_weight": 0.005,
    "diversity_weight": 0.5,
    
    # Training
    "epochs": 1000,          # Adjusted for CPU feasibility, scale up if needed
    "learning_rate": 1e-3,
    "grad_clip": 0.5,
    "eps": 1e-6,
    "warmup_epochs": 0,
    "batch_size": 16         # Added for batch support
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

words_list = TEXT_DATA.split()                   # Split text into words
unique_words = sorted(list(set(words_list)))    # Unique words for vocabulary
vocab_size = len(unique_words)

word_to_ix = {w: i+1 for i, w in enumerate(unique_words)}  # Shift indices by 1
word_to_ix["<UNK>"] = 0                                  # Reserve 0 for unknown words
ix_to_word = {i: w for w, i in word_to_ix.items()}

# Convert the text into a tensor of word indices
data_tensor = torch.tensor([word_to_ix[w] for w in words_list], dtype=torch.long).to(DEVICE)

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
# 4. Memory Modules
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
    
    def forward(self, z, memory, ptr, control):
        # [FIX] Clone ptr to avoid inplace error
        ptr = ptr.clone()
        
        push, pop, noop = control[:, 0].view(-1,1), control[:, 1].view(-1,1), control[:, 2].view(-1,1)
        
        # [FIX] Positional arguments for torch.roll (compat with all versions)
        ptr_up = torch.roll(ptr, 1, 1)
        ptr_down = torch.roll(ptr, -1, 1)
        
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
        
        # Uniform Init
        self.codebook = nn.Parameter(torch.empty(n_symbols, latent_dim*2))
        nn.init.uniform_(self.codebook, -0.5, 0.5)
        
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))
    
    def forward(self, z, prev_symbol_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        
        if prev_symbol_idx is not None:
            # [FIX] Ensure prev_symbol_idx is strictly Long for indexing
            if prev_symbol_idx.dtype != torch.long:
                if prev_symbol_idx.dim() > 1:
                    idx_safe = prev_symbol_idx.argmax(dim=-1)
                else:
                    # If it's a batch of scalars but float, cast it
                    idx_safe = prev_symbol_idx.long()
            else:
                idx_safe = prev_symbol_idx
                
            graph_prior = self.adjacency[idx_safe]
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
# 5. Core Processor
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
        
        # [FIX] Soft/Hard index conversion + Batch size handling
        batch_size = curr_sym.shape[0]
        
        if prev_sym.dtype != torch.long:
            if prev_sym.dim() > 1: prev_sym = prev_sym.argmax(dim=-1)
            else: prev_sym = prev_sym.long()
            
        if curr_sym.dtype != torch.long:
            if curr_sym.dim() > 1: curr_sym = curr_sym.argmax(dim=-1)
            else: curr_sym = curr_sym.long()
            
        row_logits = adjacency[prev_sym]
        
        # [FIX] Ensure shapes match for CrossEntropy
        return F.cross_entropy(row_logits.view(batch_size, -1), curr_sym.view(batch_size))

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        self.attention = ComplexAttention(dim) 
        nn.init.constant_(self.halt_linear.bias, -2.0)

    def forward(self, z):
        z_proc = self.act(self.norm(self.linear(z)))
        z_proc = self.attention(z_proc) 
        z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        stack_probs = F.softmax(self.stack_ctrl(z_flat), dim=-1)
        return z_proc, halt_prob, stack_probs

# ==========================================
# 6. Master Model (UberCRSN)
# ==========================================
class UberCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
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
        
        if hidden is None:
            z_prev = torch.zeros_like(z)
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device)
            stack_ptr[:, 0] = 1.0
        else:
            z_prev, stack_mem, stack_ptr = hidden
            z = 0.5 * z + 0.5 * z_prev

        act_step = 0
        halting_probability = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        ponder_cost = 0
        stack_history = [] 
        
        z_weighted = torch.zeros_like(z) 
        current_sym = prev_sym
        vq_loss_total = 0
        ethical_loss_total = 0
        
        for t in range(CONFIG["max_recursion_depth"]):
            act_step += 1
            z_proc, p_halt, stack_ctrl = self.cell(z)
            
            if CONFIG["use_stack"]:
                stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
                z_combined = z_proc + stack_read
                # Record average depth for visualization
                depth = torch.sum(stack_ptr * torch.arange(CONFIG["stack_size"], device=z.device), dim=1).mean()
                stack_history.append(depth)
            else:
                z_combined = z_proc
                stack_history.append(torch.zeros(1).to(z.device))

            z_sym, vq_loss, sym_idx = self.vq_layer(z_combined, current_sym)
            
            eth_loss = self.ethics(current_sym, sym_idx, self.vq_layer.adjacency)
            ethical_loss_total += eth_loss
            current_sym = sym_idx
            
            z = 0.7 * z_combined + 0.3 * z_sym
            
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            p = p_halt * still_running
            if t == CONFIG["max_recursion_depth"] - 1: p = remain
            
            # [FIX] Out-of-place updates to prevent autograd errors
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
            
        return logits, next_hidden, current_sym, ponder_cost, vq_loss_total, ethical_loss_total, avg_stack

# ==========================================
# 7. Training Engine
# ==========================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-5)
    
    # [FIX] Batch training setup
    dataset = TensorDataset(data_tensor[:-1], data_tensor[1:])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    
    print(f"--- Training SACRSN v32 (Unified) ---")
    
    try:
        for epoch in range(CONFIG["epochs"]):
            hidden = None
            prev_sym = None
            total_loss = 0
            total_ponder = 0
            total_ppx = 0
            
            entropy_weight = 0.01 * (1 - epoch / CONFIG["epochs"])
            
            for x_batch, y_batch in loader:
                x = x_batch.view(CONFIG["batch_size"], 1)
                y = y_batch.view(CONFIG["batch_size"])
                
                logits, hidden, sym_idx, ponder, vq_loss, eth_loss, _ = model(x, hidden, prev_sym)
                
                # TBPTT Detach
                h_z, h_mem, h_ptr = hidden
                hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach())
                prev_sym = sym_idx.detach()
                
                # [FIX] CrossEntropy handles batches naturally now
                loss_pred = F.cross_entropy(logits, y)
                loss_ponder = CONFIG["ponder_penalty"] * ponder
                
                probs = F.softmax(logits, dim=-1)
                loss_entropy = -entropy_weight * (-(probs * torch.log(probs + 1e-8)).sum())
                
                # [FIX] prev_sym_soft batch update
                curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float()
                # Average one-hot across batch to update global buffer
                curr_onehot_avg = curr_onehot.mean(dim=0)
                
                # Safe out-of-place update
                model.prev_sym_soft = (model.prev_sym_soft * 0.9 + curr_onehot_avg * 0.1).detach()
                
                buffer_usage = model.prev_sym_soft
                loss_diversity = CONFIG["diversity_weight"] * (buffer_usage * torch.log(buffer_usage + 1e-9)).sum()
                loss_ethics = CONFIG["ethical_weight"] * eth_loss
                
                loss = loss_pred + loss_ponder + 0.1*vq_loss + loss_entropy + loss_diversity + loss_ethics
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                
                total_loss += loss.item()
                total_ponder += ponder.item()
                
                # True PPX
                usage_dist = model.prev_sym_soft.detach() + 1e-10
                entropy_val = -(usage_dist * torch.log(usage_dist)).sum()
                total_ppx += torch.exp(entropy_val).item()
                
            scheduler.step()

            if epoch % 50 == 0:
                avg_loss = total_loss / len(loader)
                avg_ponder = total_ponder / len(loader)
                avg_ppx = total_ppx / len(loader)
                lr = scheduler.get_last_lr()[0]
                print(f"Ep {epoch:04d} | Loss: {avg_loss:.4f} | Steps: {avg_ponder:.2f} | Usage(PPX): {avg_ppx:.1f} | LR: {lr:.6f}")
                
                if avg_loss < 0.01:
                    print("\n--- PERFECT CONVERGENCE ---")
                    return model

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    return model

# ==========================================
# 8. Visualization Suite (Preserved & Fixed)
# ==========================================
def visualize_all(model):
    print("\n--- Generating Diagnostics & Images ---")
    model.eval()
    
    # 1. Semantic Mapping
    symbol_to_char = defaultdict(list)
    hidden, prev_sym = None, None
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            _, hidden, prev_sym, _, _, _, _ = model(x, hidden, prev_sym)
            current_word = ix_to_word[data_tensor[i].item()]
            symbol_to_char[prev_sym.item()].append(current_word)

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

    edges, weights = [], []
    for i in range(CONFIG["n_symbols"]):
        for j in range(CONFIG["n_symbols"]):
            w = adj_probs[i, j]
            if w > 0.05: 
                G.add_edge(i, j, weight=w)
                edges.append((i, j))
                weights.append(w)
    
    plt.figure(figsize=(14, 14))
    try: pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    except: pos = nx.circular_layout(G)
    
    node_colors = ['#a0cbe2' if i in symbol_to_char else '#ffe5e5' for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold")
    
    for (u, v), w in zip(edges, weights):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=w * 2.0, alpha=max(0.1, w), edge_color='gray', arrowstyle='->', arrowsize=10)
    
    plt.title(f"1_semantic_topology (Active: {len(symbol_to_char)})")
    plt.savefig("1_semantic_topology.png", dpi=150)
    print("Saved 1_semantic_topology.png")
    plt.close()

    # 3. Inference Scan
    hidden, prev_sym = None, None
    x = torch.tensor([[word_to_ix["The"]]], device=DEVICE)
    stack_history, act_history, phase_reals, phase_imags = [], [], [], []
    gen_text = "The"
    
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
            word = ix_to_word[next_ix.item()]
            gen_text += " " + word
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
    plt.colorbar(label="Time")
    plt.title("4_phase_plot (Complex Trajectory)")
    plt.axis('equal')
    plt.savefig("4_phase_plot.png")
    plt.close()

    # ==========================================
    # 8.1. Semantic Anomaly Detection
    # ==========================================
    print("\nRunning Semantic Anomaly Detection...")
    corrupt_text = "The neural architecture of the mind is a banana"
    words = corrupt_text.split()
    input_tensor = torch.tensor([word_to_ix.get(w, word_to_ix.get("<UNK>", 0)) for w in words],
                                dtype=torch.long).to(DEVICE)

    anomalies = []
    symbol_labels = []

    # Build symbol → most frequent word mapping (reuse from above)
    symbol_to_word = defaultdict(lambda: "<UNK>")
    for sym_idx, word_list in symbol_to_char.items():
        if word_list:
            symbol_to_word[sym_idx] = max(set(word_list), key=word_list.count)

    # Seed prev_sym
    x0 = input_tensor[0].view(1, 1)
    with torch.no_grad():
        _, hidden, prev_sym, _, _, _, _ = model(x0, None, None)

    # Compute anomalies
    with torch.no_grad():
        for i in range(1, len(input_tensor)):
            x = input_tensor[i-1].view(1, 1)
            _, hidden, curr_sym, _, _, eth_loss, _ = model(x, hidden, prev_sym)
            anomalies.append(eth_loss.item())
            symbol_labels.append(symbol_to_word.get(curr_sym.item(), "<UNK>"))
            prev_sym = curr_sym

    anomalies = np.array(anomalies)
    if len(anomalies) > 2:
        anomalies = np.convolve(anomalies, np.ones(3)/3, mode='same')

    plt.figure(figsize=(10, 4))
    plt.plot(words[1:], anomalies, color='skyblue', linestyle='-', linewidth=2, label='Topology Flow')
    for w, val, sym in zip(words[1:], anomalies, symbol_labels):
        plt.scatter(w, val, color='crimson', s=50, zorder=5)
        plt.text(w, val + 0.02, sym, fontsize=8, ha='center', va='bottom', rotation=45)
    
    plt.title("5_anomaly_detection_semantic")
    plt.ylabel("Violation Magnitude")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("5_anomaly_detection_semantic.png")
    plt.close()
    print("Saved 5_anomaly_detection_semantic.png")

# ==========================================
# 10. Advanced Interaction
# ==========================================
def dream_mode(model):
    print("\n--- 🌙 Dream Mode (Symbolic & Semantic Consistent) ---")
    adj = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    model.eval()
    
    # --- Build symbol → most frequent word mapping ---
    symbol_to_word = defaultdict(lambda: "<UNK>")
    hidden, prev_sym = None, None
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            _, hidden, sym_idx, _, _, _, _ = model(x, hidden, prev_sym)
            prev_sym = sym_idx
            word = ix_to_word[data_tensor[i].item()]
            if symbol_to_word[sym_idx.item()] == "<UNK>":
                symbol_to_word[sym_idx.item()] = word  # first occurrence

    # Start with "The"
    x = torch.tensor([[word_to_ix["The"]]], device=DEVICE)
    _, _, prev_sym, _, _, _, _ = model(x, None, None)
    curr_sym = prev_sym.item()
    output = "The"
    
    for _ in range(100):
        probs = adj[curr_sym]
        probs[probs < 0.2] = 0  # prune low-prob symbols
        if probs.sum() == 0: break
        probs = probs / probs.sum()
        next_sym = np.random.choice(len(probs), p=probs)
        
        # Use most frequent word for this symbol
        word = symbol_to_word.get(next_sym, "<UNK>")
        output += " " + word
        curr_sym = next_sym
        
    print(f"Dream Output: {output}\n")

def anomaly_detector(model, smooth=True):
    print("\n--- 🚨 Anomaly Detection Test (Semantic Labels) ---")
    corrupt_text = "The neural architecture of the mind is a banana"
    words = corrupt_text.split()
    print(f"Input words: {words}")

    # --- 0. Handle unknown words ---
    input_tensor = torch.tensor([word_to_ix.get(w, word_to_ix.get("<UNK>", 0)) for w in words],
                                dtype=torch.long).to(DEVICE)

    model.eval()
    anomalies = []
    symbol_labels = []

    # --- 1. Build symbol → most frequent word mapping ---
    symbol_to_word = defaultdict(lambda: "<UNK>")
    hidden, prev_sym = None, None
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            _, hidden, sym_idx, _, _, _, _ = model(x, hidden, prev_sym)
            prev_sym = sym_idx
            word = ix_to_word[data_tensor[i].item()]
            if symbol_to_word[sym_idx.item()] == "<UNK>":
                symbol_to_word[sym_idx.item()] = word  # first occurrence

    # --- 2. Seed prev_sym with first symbol ---
    x0 = input_tensor[0].view(1, 1)
    with torch.no_grad():
        _, hidden, prev_sym, _, _, _, _ = model(x0, None, None)

    # --- 3. Iterate over remaining words ---
    with torch.no_grad():
        for i in range(1, len(input_tensor)):
            x = input_tensor[i-1].view(1, 1)
            _, hidden, curr_sym, _, _, eth_loss, _ = model(x, hidden, prev_sym)
            
            anomalies.append(eth_loss.item())
            symbol_labels.append(symbol_to_word.get(curr_sym.item(), "<UNK>"))
            prev_sym = curr_sym

    anomalies = np.array(anomalies)

    # --- 4. Optional smoothing ---
    if smooth and len(anomalies) > 2:
        anomalies = np.convolve(anomalies, np.ones(3)/3, mode='same')

    # --- 5. Plot word-aligned anomalies with semantic labels ---
    plt.figure(figsize=(10, 4))
    plt.plot(words[1:], anomalies, color='skyblue', linestyle='-', linewidth=2, label='Topology Flow')
    for w, val, sym in zip(words[1:], anomalies, symbol_labels):
        plt.scatter(w, val, color='crimson', s=50, zorder=5)
        plt.text(w, val + 0.02, sym, fontsize=8, ha='center', va='bottom', rotation=45)

    plt.title("Topological Violation Score (Semantic)")
    plt.ylabel("Violation Magnitude")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("5_anomaly_detection_semantic.png")
    print("Saved 5_anomaly_detection_semantic.png")
    plt.close()

# ==========================================
# 11. Main Execution
# ==========================================
if __name__ == "__main__":
    FILENAME = "crsn_omni_model.pth"
    
    # [FIX] Added main batch size handling
    trained_model = train()
    
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
    }, FILENAME)
    print("Saved.")
    
    visualize_all(trained_model)
    extract_logic_rules(trained_model, data_tensor)
    
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
