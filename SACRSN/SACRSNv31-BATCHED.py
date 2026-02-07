# ============================================================
# SACRSN v31: THE BATCHED EDITION
# Features: Stack, ACT, Normative Ethics, Uniform Init, Entropy Force
# Upgrades: Parallel Training (Batching), Scalar Reductions, 1D/2D Split
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

# ==========================================
# 0. Determinism & Speed
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
# [BATCHING UPDATE] Turn Benchmark ON for speed (15-30% gain)
torch.backends.cudnn.deterministic = False 
torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "seq_len": 32,
    "embedding_dim": 128,
    "n_symbols": 128,
    
    # [BATCHING UPDATE] Batch Size added
    "batch_size": 32,
    
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
    "epochs": 3000,
    # [BATCHING UPDATE] Higher LR for batched gradients
    "learning_rate": 3e-3, 
    "grad_clip": 0.5,
    "eps": 1e-6,
    "warmup_epochs": 5
}

# ==========================================
# 2. Data Preparation
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
"""

# [BATCHING UPDATE] Repeat data to ensure columns are long enough
required_len = CONFIG["batch_size"] * 256
if len(TEXT_DATA) < required_len:
    repeat_factor = (required_len // len(TEXT_DATA)) + 2
    TEXT_DATA = TEXT_DATA * repeat_factor
    print(f"Data repeated {repeat_factor} times for batch depth.")

chars = sorted(list(set(TEXT_DATA)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# 1. Create 1D Tensor for Analysis (Visualization)
eval_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(DEVICE)

# 2. Create 2D Tensor for Training (High Speed)
n_batches = len(eval_tensor) // CONFIG["batch_size"]
train_tensor = eval_tensor[:n_batches * CONFIG["batch_size"]]
train_tensor = train_tensor.view(CONFIG["batch_size"], -1) # Shape: [32, Seq_Len]

print(f"Train Tensor Shape: {train_tensor.shape}")

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
    
    def forward(self, z, prev_symbol_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        
        if prev_symbol_idx is not None:
            # Handle Batched Adjacency Lookup
            # prev_symbol_idx is [Batch]
            graph_prior = self.adjacency[prev_symbol_idx] # [Batch, N_Symbols]
            bias = CONFIG["graph_bias_scale"] * torch.sigmoid(graph_prior)
            d = d - bias

        min_indices = torch.argmin(d, dim=-1)
        z_q = F.embedding(min_indices, self.codebook)
        
        loss_vq = F.mse_loss(z_q, z_flat.detach(), reduction='none').mean(dim=-1) # Return vector [Batch]
        loss_commit = F.mse_loss(z_q.detach(), z_flat, reduction='none').mean(dim=-1)
        
        z_q = z_flat + (z_q - z_flat).detach()
        z_complex = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        
        # Returns vector loss for safety in batch training
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
        if prev_sym is None: return torch.zeros(curr_sym.size(0)).to(adjacency.device)
        row_logits = adjacency[prev_sym] # [Batch, N_Symbols]
        # Calculate CE per element in batch
        return F.cross_entropy(row_logits, curr_sym, reduction='none')

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
                depth = torch.sum(stack_ptr * torch.arange(CONFIG["stack_size"], device=z.device), dim=1)
                stack_history.append(depth)
            else:
                z_combined = z_proc
                stack_history.append(torch.zeros(batch_size).to(z.device))

            z_sym, vq_loss, sym_idx = self.vq_layer(z_combined, current_sym)
            
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
            ponder_cost += still_running
            vq_loss_total += vq_loss

        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)
        
        next_hidden = (z_weighted, stack_mem, stack_ptr)
        
        if len(stack_history) > 0: avg_stack = torch.stack(stack_history).mean()
        else: avg_stack = torch.tensor(0.0)
            
        return logits, next_hidden, current_sym, ponder_cost, vq_loss_total, ethical_loss_total, avg_stack

# ==========================================
# 7. Training Engine (Optimized)
# ==========================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    
    # [BATCHING UPDATE] Warmup Logic
    initial_lr = CONFIG["learning_rate"] * 0.1
    opt = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-5)
    
    print(f"--- Training SACRSN v32 (Batched) | BS: {CONFIG['batch_size']} ---")
    
    # [BATCHING UPDATE] Iterate Columns
    seq_len = train_tensor.size(1)
    print(f"Batched Steps per Epoch: {seq_len}")
    
    try:
        for epoch in range(CONFIG["epochs"]):
            hidden = None
            prev_sym = None
            total_loss = 0
            total_ponder = 0
            total_ppx = 0
            
            # [BATCHING UPDATE] Linear Warmup
            if epoch < CONFIG["warmup_epochs"]:
                target_lr = CONFIG["learning_rate"] * ((epoch + 1) / CONFIG["warmup_epochs"])
                for g in opt.param_groups: g['lr'] = target_lr
            
            entropy_weight = 0.01 * (1 - epoch / CONFIG["epochs"])
            
            # Inner Loop: Sequence
            for i in range(seq_len - 1):
                x = train_tensor[:, i].unsqueeze(1) # [Batch, 1]
                y = train_tensor[:, i+1]            # [Batch]
                
                logits, hidden, sym_idx, ponder, vq_loss, eth_loss, _ = model(x, hidden, prev_sym)
                
                # Detach for TBPTT
                h_z, h_mem, h_ptr = hidden
                hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach())
                prev_sym = sym_idx.detach()
                
                # [BATCHING UPDATE] Scalar Loss Reductions
                loss_pred = F.cross_entropy(logits, y, reduction='mean')
                loss_ponder = CONFIG["ponder_penalty"] * ponder.mean()
                loss_vq_scaled = 0.1 * vq_loss.mean()
                loss_ethics = CONFIG["ethical_weight"] * eth_loss.mean()
                
                # Entropy: Sum(Vocab) -> Mean(Batch)
                probs = F.softmax(logits, dim=-1)
                loss_entropy = -entropy_weight * (-(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean())
                
                # Buffer Update: Average Batch first
                curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float()
                with torch.no_grad():
                    avg_onehot = curr_onehot.mean(dim=0)
                    model.prev_sym_soft.copy_(model.prev_sym_soft * 0.9 + avg_onehot * 0.1)
                
                buffer_usage = model.prev_sym_soft
                loss_diversity = CONFIG["diversity_weight"] * (buffer_usage * torch.log(buffer_usage + 1e-9)).sum()
                
                loss = loss_pred + loss_ponder + loss_vq_scaled + loss_entropy + loss_diversity + loss_ethics
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                
                total_loss += loss.item()
                total_ponder += ponder.mean().item()
                
                # PPX Metric
                usage_dist = model.prev_sym_soft.detach() + 1e-10
                entropy_val = -(usage_dist * torch.log(usage_dist)).sum()
                total_ppx += torch.exp(entropy_val).item()
                
            scheduler.step()

            if epoch % 50 == 0:
                # [BATCHING UPDATE] Divide by Sequence Length
                avg_loss = total_loss / seq_len
                avg_ponder = total_ponder / seq_len
                avg_ppx = total_ppx / seq_len
                lr = opt.param_groups[0]['lr']
                print(f"Ep {epoch:04d} | Loss: {avg_loss:.4f} | Steps: {avg_ponder:.2f} | Usage(PPX): {avg_ppx:.1f} | LR: {lr:.6f}")
                
                if avg_loss < 0.05:
                    print("\n--- PERFECT CONVERGENCE ---")
                    return model

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    return model

# ==========================================
# 8. Visualization Suite
# ==========================================
def visualize_all(model):
    print("\n--- Generating Diagnostics & Images ---")
    model.eval()
    
    # 1. Semantic Mapping
    symbol_to_char = defaultdict(list)
    hidden, prev_sym = None, None
    with torch.no_grad():
        # [BATCHING UPDATE] Use eval_tensor (1D)
        for i in range(len(eval_tensor) - 1):
            x = eval_tensor[i].view(1, 1) # Batch=1
            _, hidden, prev_sym, _, _, _, _ = model(x, hidden, prev_sym)
            current_char = ix_to_char[eval_tensor[i].item()]
            symbol_to_char[prev_sym.item()].append(current_char)

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
    plt.colorbar(label="Time")
    plt.title("4_phase_plot (Complex Trajectory)")
    plt.axis('equal')
    plt.savefig("4_phase_plot.png")
    plt.close()

def extract_logic_rules(model, dataset):
    print("\n--- Extracting Explicit Logic Rules ---")
    model.eval()
    rule_book = defaultdict(list)
    hidden = None
    prev_sym = None
    
    with torch.no_grad():
        # [BATCHING UPDATE] Iterate the 1D dataset passed as argument
        for i in range(len(dataset) - 1):
            x = dataset[i].view(1, 1)
            logits, hidden, sym_idx, ponder, _, _, _ = model(x, hidden, prev_sym)
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
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    _, _, prev_sym, _, _, _, _ = model(x, None, None)
    curr_sym = prev_sym.item()
    output = "T"
    
    for _ in range(100):
        probs = adj[curr_sym]
        probs[probs < 0.2] = 0 
        if probs.sum() == 0: break
        probs = probs / probs.sum()
        next_sym = np.random.choice(len(probs), p=probs)
        
        z_flat = model.vq_layer.codebook[next_sym].unsqueeze(0)
        logits = model.decoder(z_flat)
        char_idx = torch.argmax(logits).item()
        
        output += ix_to_char[char_idx]
        curr_sym = next_sym
        
    print(f"Dream Output: {output}\n")

def anomaly_detector(model):
    print("\n--- 🚨 Anomaly Detection Test ---")
    corrupt_text = "True without falsehood certain and most banana"
    print(f"Input: '{corrupt_text}'")
    
    input_tensor = torch.tensor([char_to_ix.get(c, 0) for c in corrupt_text], dtype=torch.long).to(DEVICE)
    hidden, prev_sym = None, None
    anomalies = []
    
    with torch.no_grad():
        for i in range(len(input_tensor) - 1):
            x = input_tensor[i].view(1, 1)
            _, hidden, prev_sym, _, _, eth_loss, _ = model(x, hidden, prev_sym)
            anomalies.append(eth_loss.item())

    plt.figure(figsize=(10, 4))
    plt.plot(list(corrupt_text)[1:], anomalies, color='skyblue', linestyle='-', linewidth=2, label='Topology Flow')
    plt.scatter(list(corrupt_text)[1:], anomalies, color='crimson', s=50, zorder=5, label='Anomaly Score')
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
    FILENAME = "crsn_batched_model.pth"
    
    # 1. Train using 2D Batched Tensor
    trained_model = train()
    
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
    }, FILENAME)
    print("Saved.")
    
    # 2. Analysis using 1D Eval Tensor (Crucial for avoiding crash)
    visualize_all(trained_model)
    extract_logic_rules(trained_model, eval_tensor)
    
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
