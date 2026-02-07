# ============================================================
# SACRSN v33: THE COMPLETE DYNAMIC EDITION
# Architecture: Neuro-Symbolic Recursive Network (Complex-Valued)
# Upgrades: Dynamic Graph Topology, Word-Level Tokenization (BPE-Lite)
# Restored: Full Diagnostic Suite (Logic, Phase, MRI, 4D Heatmap)
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
    "seq_len": 32,                 # Word-level: 32 words is a full paragraph context
    "embedding_dim": 128,
    "n_heads": 4,                  # Multi-Head Complex Attention
    "n_symbols": 128,              # The Concept Bottleneck
    
    # Recursive Reasoning (ACT)
    "max_recursion_depth": 10,
    "act_threshold": 0.99,
    "ponder_penalty": 0.001,
    
    # Memory
    "use_stack": True,
    "stack_size": 24,
    
    # Dynamic Topology (New)
    "commitment_cost": 0.25,
    "decay": 0.99,
    "graph_bias_scale": 0.8,       # Strong prior
    "context_influence": 0.5,      # How much the Hidden State warps the Graph
    
    # Training
    "epochs": 600,
    "batch_size": 32,
    "learning_rate": 8e-4,
    "grad_clip": 1.0,
    "eps": 1e-6,
    "warmup_epochs": 10
}

# ==========================================
# 2. Data & Tokenization (Word-Level)
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
Entropy is both enemy and muse. It tears down the old and paints the new in invisible strokes.
We measure the world in bits, but the soul measures the world in potential.
The universe is a neural network, and we are both its neurons and its emergent thought.
The pattern repeats, scaling from the atom to the galaxy, a fractal of meaning in a sea of noise.
"""

# Simple Regex Tokenizer (Splits words and punctuation)
def tokenizer(text):
    return re.findall(r"\w+|[^\w\s]", text)

tokens = tokenizer(TEXT_DATA)
vocab = sorted(list(set(tokens)))
vocab_size = len(vocab)
token_to_ix = {t: i for i, t in enumerate(vocab)}
ix_to_token = {i: t for i, t in enumerate(vocab)}

print(f"Vocab Size: {vocab_size} | Total Tokens: {len(tokens)}")

data_tensor = torch.tensor([token_to_ix[t] for t in tokens], dtype=torch.long).to(DEVICE)

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
        out = V 
        
        out_real = out.real.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out_imag = out.imag.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out_complex = torch.complex(out_real, out_imag).squeeze(1)
        return self.o_proj(out_complex)

# ==========================================
# 5. DYNAMIC GRAPH MEMORY VQ (The Upgrade)
# ==========================================
class GraphMemoryVQ_Dynamic(nn.Module):
    def __init__(self, latent_dim, n_symbols, decay=0.99):
        super().__init__()
        self.n_symbols = n_symbols
        self.decay = decay
        self.embedding_dim = latent_dim * 2
        
        # VQ Buffers
        self.register_buffer("codebook", torch.randn(n_symbols, self.embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(n_symbols))
        self.register_buffer("embed_avg", self.codebook.clone())
        
        # 1. Static Global Rules (The "Grammar")
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))
        
        # 2. Dynamic Context Gate (The "Meaning")
        # Projects the complex hidden state to a bias vector for all symbols
        self.context_gate = nn.Linear(latent_dim * 2, n_symbols)

    def forward(self, z, hidden_state_context, prev_symbol_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        # Euclidean Distance
        d = torch.sum(z_flat**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        
        # --- DYNAMIC TOPOLOGY LOGIC ---
        if prev_symbol_idx is not None:
            if prev_symbol_idx.dim() > 0:
                idx_safe = prev_symbol_idx.long()
                
                # A. Static Transition
                static_prior = self.adjacency[idx_safe]
                
                # B. Dynamic Modulation
                # Use the raw hidden state (before VQ) to determine context
                ctx_flat = torch.cat([hidden_state_context.real, hidden_state_context.imag], dim=-1)
                context_bias = self.context_gate(ctx_flat)
                
                # C. Fusion: Context warps the static probabilities
                combined_prior = torch.sigmoid(static_prior) * (1.0 + CONFIG["context_influence"] * torch.tanh(context_bias))
                
                d = d - (CONFIG["graph_bias_scale"] * combined_prior)
        # -----------------------------

        min_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_indices, self.codebook)
        
        # EMA Update
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
        if prev_sym.dtype != torch.long: prev_sym = prev_sym.long()
        if curr_sym.dtype != torch.long: curr_sym = curr_sym.long()
        row_logits = adjacency[prev_sym]
        return F.cross_entropy(row_logits.view(batch_size, -1), curr_sym.view(batch_size))

# ==========================================
# 7. Model: UberCRSN v33
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
        # Using Dynamic VQ
        self.vq_layer = GraphMemoryVQ_Dynamic(dim, CONFIG["n_symbols"], decay=CONFIG["decay"])
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

            # Pass z_combined (Context) to VQ for Dynamic Gating
            z_sym, vq_loss, sym_idx = self.vq_layer(z_combined, z_combined, current_sym)
            
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
# 8. Training Engine
# ==========================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"] - CONFIG["warmup_epochs"], eta_min=1e-5)
    
    dataset = TensorDataset(data_tensor[:-1], data_tensor[1:])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    
    print(f"--- Training SACRSN v33 (Dynamic Word-Level) ---")
    
    try:
        for epoch in range(CONFIG["epochs"]):
            hidden = None
            prev_sym = None
            total_loss = 0
            total_ponder = 0
            total_ppx = 0
            
            # Warmup
            if epoch < CONFIG["warmup_epochs"]:
                lr_scale = min(1.0, (epoch + 1) / (CONFIG["warmup_epochs"] + 1))
                current_lr = CONFIG["learning_rate"] * lr_scale
                for param_group in opt.param_groups: param_group['lr'] = current_lr
            
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
                
                # Update Buffer
                curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float().mean(dim=0)
                with torch.no_grad():
                    model.prev_sym_soft.copy_(model.prev_sym_soft * 0.9 + curr_onehot * 0.1)
                loss_diversity = 0.1 * (model.prev_sym_soft * torch.log(model.prev_sym_soft + 1e-9)).sum()
                
                loss = loss_pred + loss_ponder + vq_loss + loss_entropy + loss_diversity + 0.01 * eth_loss
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                
                total_loss += loss.item()
                total_ponder += ponder.item()
                total_ppx += torch.exp(loss_pred).item()
            
            if epoch >= CONFIG["warmup_epochs"]: scheduler.step()

            if epoch % 10 == 0:
                avg_loss = total_loss / len(loader)
                avg_ponder = total_ponder / len(loader)
                avg_ppx = total_ppx / len(loader)
                lr = opt.param_groups[0]['lr']
                print(f"Ep {epoch:04d} | Loss: {avg_loss:.4f} | Steps: {avg_ponder:.2f} | PPX: {avg_ppx:.1f} | LR: {lr:.6f}")
                
                if avg_loss < 0.1: 
                    print("\n--- CONVERGENCE REACHED ---")
                    return model

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    return model

# ==========================================
# 9. Full Visualization Suite
# ==========================================
def visualize_all(model):
    print("\n--- Generating Diagnostics ---")
    model.eval()
    
    # 1. Semantic Mapping
    symbol_to_word = defaultdict(list)
    symbol_freq = Counter()
    hidden, prev_sym = None, None
    scan_limit = min(500, len(data_tensor) - 1)
    
    with torch.no_grad():
        for i in range(scan_limit):
            x = data_tensor[i].view(1, 1)
            _, hidden, curr_sym, _, _, _, _ = model(x, hidden, prev_sym)
            if prev_sym is not None:
                word = ix_to_token[data_tensor[i].item()]
                symbol_to_word[prev_sym.item()].append(word)
                symbol_freq[prev_sym.item()] += 1
            prev_sym = curr_sym

    node_labels = {}
    for sym_idx in range(CONFIG["n_symbols"]):
        w_list = symbol_to_word.get(sym_idx, [])
        if w_list:
            most_common = max(set(w_list), key=w_list.count)
            node_labels[sym_idx] = f"{most_common}\n({len(w_list)})"
        else:
            node_labels[sym_idx] = f"{sym_idx}"

    # 2. X-Ray Topology
    adj_probs = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    G = nx.DiGraph()
    for i in range(CONFIG["n_symbols"]): G.add_node(i)

    for i in range(CONFIG["n_symbols"]):
        for j in range(CONFIG["n_symbols"]):
            w = adj_probs[i, j]
            if w > 0.15: G.add_edge(i, j, weight=w)
    
    plt.figure(figsize=(12, 12))
    try: pos = nx.spring_layout(G, k=0.15, seed=42)
    except: pos = nx.circular_layout(G)
    
    node_colors = ['#a0cbe2' if i in symbol_to_word else '#ffe5e5' for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, font_weight="bold")
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', arrowstyle='->', arrowsize=10)
    plt.title(f"1_semantic_topology (Word Level)")
    plt.savefig("1_semantic_topology.png", dpi=150)
    plt.close()

    # 3. Inference Scan & Phase Plot
    hidden, prev_sym = None, None
    start_token = tokens[0]
    x = torch.tensor([[token_to_ix[start_token]]], device=DEVICE)
    stack_hist, act_hist = [], []
    phase_reals, phase_imags = [], []
    gen_text = [start_token]
    
    print("Running Inference Scan...")
    for _ in range(50): 
        with torch.no_grad():
            logits, hidden, prev_sym, ponder, _, _, s_hist = model(x, hidden, prev_sym)
            stack_hist.append(s_hist.item())
            act_hist.append(1.0 + ponder.item())
            
            # Phase Capture
            z = hidden[0].cpu().squeeze()
            if z.dim() > 0: 
                phase_reals.append(z.real[0].item())
                phase_imags.append(z.imag[0].item())
            else:
                phase_reals.append(z.real.item())
                phase_imags.append(z.imag.item())
            
            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, 1)
            word = ix_to_token[next_ix.item()]
            gen_text.append(word)
            x = next_ix

    print(f"Generated: {' '.join(gen_text)}\n")

    # Stack MRI
    plt.figure(figsize=(12, 4))
    plt.plot(stack_hist, color='purple', label='Stack Depth')
    plt.fill_between(range(len(stack_hist)), stack_hist, color='purple', alpha=0.1)
    plt.title("2_stack_mri")
    plt.savefig("2_stack_mri.png")
    plt.close()

    # ACT Profile (Restored)
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(act_hist)), act_hist, color='orange')
    plt.title("3_act_profile")
    plt.savefig("3_act_profile.png")
    plt.close()
    
    # Phase Plot (Restored)
    plt.figure(figsize=(8, 8))
    plt.scatter(phase_reals, phase_imags, c=range(len(phase_reals)), cmap='plasma', alpha=0.5)
    plt.title("4_phase_plot")
    plt.axis('equal')
    plt.savefig("4_phase_plot.png")
    plt.close()

    # 4. 4D Diagnostic Heatmap (Word Level)
    print("Generating Heatmap...")
    test_str = "The neural architecture of the mind is a mirror"
    test_tokens = tokenizer(test_str)
    input_tensor = torch.tensor([token_to_ix.get(t, 0) for t in test_tokens], dtype=torch.long).to(DEVICE)
    
    topo, stack, act, rarity = [], [], [], []
    x0 = input_tensor[0].view(1,1)
    max_f = max(symbol_freq.values()) if symbol_freq else 1
    
    hidden, prev_sym = None, None
    with torch.no_grad():
        _, hidden, prev_sym, _, _, _, _ = model(x0, None, None)
        for i in range(1, len(input_tensor)):
            x = input_tensor[i-1].view(1,1)
            _, hidden, curr_sym, ponder, _, eth_loss, stack_d = model(x, hidden, prev_sym)
            
            topo.append(eth_loss.item())
            stack.append(stack_d.item())
            act.append(ponder.item())
            rarity.append(1.0 - (symbol_freq[curr_sym.item()] / (max_f + 1e-9)))
            prev_sym = curr_sym
    
    def normalize(lst):
        arr = np.array(lst)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) if arr.max() > arr.min() else arr

    matrix = np.vstack([normalize(topo), normalize(stack), normalize(act), normalize(rarity)])
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect='auto', cmap='magma', interpolation='nearest')
    plt.yticks(range(4), ['Topology Violation', 'Stack Intensity', 'Cognitive Load', 'Symbol Surprise'])
    plt.xticks(range(len(test_tokens)-1), test_tokens[1:], rotation=45, ha='right')
    plt.colorbar()
    plt.title("5_diagnostic_heatmap")
    plt.tight_layout()
    plt.savefig("5_diagnostic_heatmap.png")
    plt.close()

# ==========================================
# 10. Advanced Dream Mode (Restored Overlay)
# ==========================================
def dream_mode(model):
    print("\n--- 🌙 Dream Mode ---")
    adj = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    
    # Map symbols to words
    sym_to_word = defaultdict(lambda: "?")
    hidden, prev_sym = None, None
    scan_limit = min(500, len(data_tensor) - 1)
    with torch.no_grad():
        for i in range(scan_limit):
            x = data_tensor[i].view(1,1)
            _, hidden, curr_sym, _, _, _, _ = model(x, hidden, prev_sym)
            if prev_sym is not None:
                sym_to_word[prev_sym.item()] = ix_to_token[data_tensor[i].item()]
            prev_sym = curr_sym
            
    # Dream Walk
    x = torch.tensor([[token_to_ix[tokens[0]]]], device=DEVICE)
    _, _, prev_sym, _, _, _, _ = model(x, None, None)
    curr_sym = prev_sym.item()
    dream_tokens, dream_indices = [tokens[0]], [curr_sym]
    
    for _ in range(25):
        probs = adj[curr_sym]
        probs[probs < 0.15] = 0 
        if probs.sum() == 0: break
        probs = probs / probs.sum()
        next_sym = np.random.choice(len(probs), p=probs)
        
        w = sym_to_word.get(next_sym, "?")
        dream_tokens.append(w)
        dream_indices.append(next_sym)
        curr_sym = next_sym
        
    print(f"Dream: {' '.join(dream_tokens)}\n")
    
    # Dream Diagnostic Overlay (Restored)
    dream_tensor = torch.tensor([token_to_ix.get(t, 0) for t in dream_tokens], dtype=torch.long).to(DEVICE)
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
    plt.xticks(range(len(dream_tokens)-1), dream_tokens[1:], rotation=45, ha='right')
    
    # OVERLAY SYMBOL IDs
    for i in range(len(dream_tokens)-1):
        sym_id = dream_indices[i+1]
        plt.text(i, 0, f"{sym_id}", color='white', ha='center', va='center', fontsize=7)
        plt.text(i, 1, f"{sym_id}", color='white', ha='center', va='center', fontsize=7)
        plt.text(i, 2, f"{sym_id}", color='white', ha='center', va='center', fontsize=7)

    plt.colorbar(label="Normalized Metric")
    plt.title("6_dream_diagnostic (Symbolic Overlay)")
    plt.tight_layout()
    plt.savefig("6_dream_diagnostic.png")
    plt.close()

# ==========================================
# 11. Logic Extractor (Restored)
# ==========================================
def extract_logic_rules(model):
    print("\n--- Extracting Explicit Logic Rules ---")
    symbol_to_word = defaultdict(lambda: "?")
    hidden, prev_sym = None, None
    scan_limit = min(500, len(data_tensor) - 1)
    with torch.no_grad():
        for i in range(scan_limit):
            x = data_tensor[i].view(1,1)
            _, hidden, curr_sym, _, _, _, _ = model(x, hidden, prev_sym)
            symbol_to_word[curr_sym.item()] = ix_to_token[data_tensor[i].item()]
            prev_sym = curr_sym

    adj = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    
    print(f"\n{'FROM':<12} | {'TO':<12} | {'CONFIDENCE':<10}")
    print("-" * 40)
    
    count = 0
    for i in range(CONFIG["n_symbols"]):
        for j in range(CONFIG["n_symbols"]):
            if adj[i, j] > 0.4:
                src_word = symbol_to_word[i]
                dst_word = symbol_to_word[j]
                if src_word != "?" and dst_word != "?":
                    print(f"S{i:<3} '{src_word[:8]:<8}' -> S{j:<3} '{dst_word[:8]:<8}' | {adj[i,j]:.2f}")
                    count += 1
                    if count > 20: break
        if count > 20: break

# ==========================================
# 12. Main Execution
# ==========================================
if __name__ == "__main__":
    FILENAME = "sacrsn_v33_dynamic_complete.pth"
    
    trained_model = train()
    
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({'model': trained_model.state_dict(), 'config': CONFIG}, FILENAME)
    
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
