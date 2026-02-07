# ============================================================
# SACRSN v34: THE ULTIMATE EDITION
# Engine: v32 (Self-Aware, Metacognition, Belief Reframing)
# Analytics: v31 (Topology Scan, Logic Extraction, Anomaly Detection)
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
    "seq_len": 32,
    "embedding_dim": 64,
    "n_symbols": 64,
    
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
    
    # Meta-Cognition (v32 Features)
    "reframe_window": 8,
    "entropy_threshold": 0.3,
    "pacing_interval": 5,
    
    # Training
    "epochs": 3000,
    "learning_rate": 1e-3,
    "grad_clip": 0.5,
    "eps": 1e-6,
    "warmup_epochs": 0
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
            graph_prior = self.adjacency[prev_symbol_idx]
            bias = CONFIG["graph_bias_scale"] * torch.sigmoid(graph_prior)
            d = d - bias

        min_indices = torch.argmin(d, dim=-1)
        z_q = F.embedding(min_indices, self.codebook)
        loss_vq = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_flat)
        
        encodings = F.one_hot(min_indices, self.n_symbols).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        z_q = z_flat + (z_q - z_flat).detach()
        z_complex = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        
        return z_complex, loss_vq + loss_commit * CONFIG["commitment_cost"], min_indices, perplexity

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
        row_logits = adjacency[prev_sym]
        return F.cross_entropy(row_logits.view(-1, CONFIG["n_symbols"]), curr_sym.view(-1))

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
# 6. Meta-Cognition Modules (v32)
# ==========================================
class MetaBeliefTracker:
    def __init__(self, n_symbols):
        self.stats = {
            s: {
                "count": 0,
                "avg_loss": 0.0,
                "avg_ponder": 0.0,
                "rigidity": 0.0,
                "ethical_flag": False
            } for s in range(n_symbols)
        }

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
        best = current_sym
        best_score = dists[current_sym]
        for s in candidates:
            score = dists[s] - 0.1 * adjacency[current_sym, s]
            if score < best_score:
                best_score = score
                best = s
        return best

    def context_reframe(self, adjacency, current_sym):
        with torch.no_grad():
            adjacency[current_sym] *= 0.9
            adjacency[current_sym] += 0.1 * torch.rand_like(adjacency[current_sym])

    def label_belief(self, sym_idx):
        key = sym_idx.item()
        self.belief_labels[key] += 1
        return key

# ==========================================
# 7. Master Model (UberCRSN)
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
        
        if CONFIG["use_stack"]: self.stack = DifferentiableStack(dim, CONFIG["stack_size"])
            
        self.ethics = EthicalConstraint()
        self.register_buffer("prev_sym_soft", torch.zeros(CONFIG["n_symbols"]))
        
        # Meta-Cognition Stack
        self.reframer = BeliefReframer(CONFIG["n_symbols"])
        self.meta_tracker = MetaBeliefTracker(CONFIG["n_symbols"])
        self.ethical_gate = EthicalGate()
        self.pacer = TherapeuticPacer()

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, input_ids, hidden=None, prev_sym=None, global_step=0, training=True):
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
        perplexity_total = 0
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
                stack_history.append(torch.zeros(1).to(z.device))

            z_sym, vq_loss, sym_idx, perplexity = self.vq_layer(z_combined, current_sym)
            
            # --- BELIEF REFRAMING LOGIC (Meta-Cognition) ---
            if training:
                self.reframer.update_history(sym_idx)
                reframe_type = self.reframer.classify_reframe(loss=vq_loss.item(), ponder=p_halt.mean().item())
                
                # Check Ethical Gate & Pacer
                if reframe_type:
                    meta = self.meta_tracker.stats[sym_idx.item()]
                    if not self.ethical_gate.allow(meta) or not self.pacer.allow(global_step):
                        reframe_type = None
                
                if reframe_type == "meaning":
                    z_flat = torch.cat([z_combined.real, z_combined.imag], dim=-1).detach()
                    sym_idx = self.reframer.meaning_reframe(
                        z_flat, self.vq_layer.codebook.detach(), self.vq_layer.adjacency.detach(), sym_idx
                    )
                elif reframe_type == "context":
                    self.reframer.context_reframe(self.vq_layer.adjacency.data, sym_idx)
                
                # Update Meta-Tracker
                changed = (sym_idx != prev_sym) if prev_sym is not None else False
                self.meta_tracker.update(sym_idx, vq_loss.item(), p_halt.mean().item(), changed)
                
                # Label Belief
                self.reframer.label_belief(sym_idx)
            # -----------------------------------------------

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
            perplexity_total += perplexity

        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)
        
        next_hidden = (z_weighted, stack_mem, stack_ptr)
        avg_stack = torch.stack(stack_history).mean() if stack_history else torch.tensor(0.0)
            
        # Return 8 items (v32 signature)
        return logits, next_hidden, current_sym, ponder_cost, vq_loss_total, perplexity_total/act_step, ethical_loss_total, avg_stack

# ==========================================
# 8. Training Engine
# ==========================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-5)
    
    print(f"--- Training SACRSN v34 (Self-Aware + Observable) ---")
    
    global_step = 0
    
    try:
        for epoch in range(CONFIG["epochs"]):
            hidden = None
            prev_sym = None
            total_loss = 0
            total_ponder = 0
            total_ppx = 0
            entropy_weight = 0.01 * (1 - epoch / CONFIG["epochs"])
            
            for i in range(len(data_tensor) - 1):
                global_step += 1
                x = data_tensor[i].view(1, 1)
                y = data_tensor[i+1].view(1)
                
                # Training = True triggers self-reframing
                logits, hidden, sym_idx, ponder, vq_loss, ppx, eth_loss, _ = model(
                    x, hidden, prev_sym, global_step=global_step, training=True
                )
                
                h_z, h_mem, h_ptr = hidden
                hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach())
                prev_sym = sym_idx.detach()
                
                loss_pred = F.cross_entropy(logits, y)
                loss_ponder = CONFIG["ponder_penalty"] * ponder
                
                probs = F.softmax(logits, dim=-1)
                loss_entropy = -entropy_weight * (-(probs * torch.log(probs + 1e-8)).sum())
                
                curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float().view(-1)
                loss_temporal = CONFIG["symbol_consistency_weight"] * F.mse_loss(curr_onehot, model.prev_sym_soft.detach())
                
                with torch.no_grad():
                    model.prev_sym_soft.copy_(model.prev_sym_soft * 0.9 + curr_onehot * 0.1)
                
                buffer_usage = model.prev_sym_soft
                loss_diversity = CONFIG["diversity_weight"] * (buffer_usage * torch.log(buffer_usage + 1e-9)).sum()
                
                loss_ethics = CONFIG["ethical_weight"] * eth_loss
                
                loss = loss_pred + loss_ponder + 0.1*vq_loss + loss_entropy + loss_static_consistency(model) + loss_temporal + loss_ethics + loss_diversity
                
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

def loss_static_consistency(model):
    adj_sig = torch.sigmoid(model.vq_layer.adjacency)
    return CONFIG["symbol_consistency_weight"] * -(adj_sig * torch.log(adj_sig + CONFIG["eps"])).sum(dim=-1).mean()

# ==========================================
# 9. Visualization Suite (Restored from v31)
# ==========================================
def visualize_all(model):
    print("\n--- Generating Diagnostics & Images ---")
    model.eval()
    
    # 1. Semantic Mapping & Topology
    symbol_to_char = defaultdict(list)
    hidden, prev_sym = None, None
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            # Use training=False to freeze metacognition
            _, hidden, prev_sym, _, _, _, _, _ = model(x, hidden, prev_sym, training=False)
            current_char = ix_to_char[data_tensor[i].item()]
            symbol_to_char[prev_sym.item()].append(current_char)

    node_labels = {}
    for sym_idx in range(CONFIG["n_symbols"]):
        char_list = symbol_to_char.get(sym_idx, [])
        if char_list:
            most_common = max(set(char_list), key=char_list.count)
            node_labels[sym_idx] = f"{most_common}\n({len(char_list)})"
        else:
            node_labels[sym_idx] = f"{sym_idx}"

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
    
    # Plot 1: Semantic Topology
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

    # Inference Scan for Plots 2, 3, 4
    hidden, prev_sym = None, None
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    stack_history, act_history, phase_reals, phase_imags = [], [], [], []
    gen_text = "T"
    
    print("Running Inference Scan...")
    for _ in range(200):
        with torch.no_grad():
            logits, hidden, prev_sym, ponder, _, _, _, s_hist = model(x, hidden, prev_sym, training=False)
            
            stack_history.append(s_hist.item() if isinstance(s_hist, torch.Tensor) else 0)
            act_history.append(1.0 + ponder.item())
            
            # Unpack hidden complex state for phase plot
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

    # Plot 2: Stack MRI
    plt.figure(figsize=(12, 4))
    plt.plot(stack_history, color='purple', label='Stack Depth')
    plt.fill_between(range(len(stack_history)), stack_history, color='purple', alpha=0.1)
    plt.title("2_stack_mri (Memory Depth)")
    plt.savefig("2_stack_mri.png")
    plt.close()

    # Plot 3: ACT Profile
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(act_history)), act_history, color='orange')
    plt.title("3_act_profile (Thinking Steps)")
    plt.savefig("3_act_profile.png")
    plt.close()
    
    # Plot 4: Phase Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(phase_reals, phase_imags, c=range(len(phase_reals)), cmap='plasma', alpha=0.5)
    plt.colorbar(label="Time")
    plt.title("4_phase_plot (Complex Trajectory)")
    plt.axis('equal')
    plt.savefig("4_phase_plot.png")
    plt.close()

def extract_logic_rules(model, data_tensor):
    print("\n--- Extracting Explicit Logic Rules ---")
    model.eval()
    rule_book = defaultdict(list)
    hidden = None
    prev_sym = None
    
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            # v32 returns 8 items; ponder is index 3, sym_idx is index 2
            _, hidden, sym_idx, ponder, _, _, _, _ = model(x, hidden, prev_sym, training=False)
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

def dream_mode(model):
    print("\n--- 🌙 Dream Mode (Pure Symbolic Walk) ---")
    adj = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    model.eval()
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    _, _, prev_sym, _, _, _, _, _ = model(x, None, None, training=False)
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
    print("\n--- 🚨 Anomaly Detection Test (Plot 5) ---")
    corrupt_text = "True without falsehood certain and most banana"
    print(f"Input: '{corrupt_text}'")
    
    input_tensor = torch.tensor([char_to_ix.get(c, 0) for c in corrupt_text], dtype=torch.long).to(DEVICE)
    hidden, prev_sym = None, None
    anomalies = []
    
    with torch.no_grad():
        for i in range(len(input_tensor) - 1):
            x = input_tensor[i].view(1, 1)
            # v32 return unpacking: eth_loss is index 6
            _, hidden, prev_sym, _, _, _, eth_loss, _ = model(x, hidden, prev_sym, training=False)
            anomalies.append(eth_loss.item())

    # Plot 5: Anomaly Score
    plt.figure(figsize=(10, 4))
    plt.plot(list(corrupt_text)[1:], anomalies, color='skyblue', linestyle='-', linewidth=2, label='Topology Flow')
    plt.scatter(list(corrupt_text)[1:], anomalies, color='crimson', s=50, zorder=5, label='Anomaly Score')
    
    plt.title("5_anomaly_detection (Topological Violation)")
    plt.ylabel("Violation Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("5_anomaly_detection.png")
    print("Saved 5_anomaly_detection.png")
    plt.close()

# ==========================================
# 10. Main Execution
# ==========================================
if __name__ == "__main__":
    FILENAME = "crsn_ultimate_model.pth"
    
    trained_model = train()
    
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
    }, FILENAME)
    print("Saved.")
    
    # Run all restored analytics
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
