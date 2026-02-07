import os
import time
import random
import math
import numpy as np
from collections import defaultdict, deque

# Conditional imports for PyTorch/Plotting (v31 and v45)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import networkx as nx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/Matplotlib not found. Modes 2 and 3 will not work.")

try:
    import plotly.graph_objects as go
except ImportError:
    pass

# ==============================================================================
# PART 1: SACRSN v46 combo — FULL INTEGRATED SCRIPT (Mathematically Grounded)
# ==============================================================================

def softmax_np(x, temp=1.0):
    x = np.array(x) / max(temp, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def entropy_np(p):
    p = np.clip(p, 1e-9, 1.0)
    return -np.sum(p * np.log(p))

class SymbolMemory_v41:
    def __init__(self, dim=32, capacity=256):
        self.dim = dim
        self.capacity = capacity
        self.codes = np.random.randn(capacity, dim)
        self.usage = np.zeros(capacity)

    def energy(self, x):
        d = np.linalg.norm(self.codes - x[None, :], axis=1)
        return d

    def sample(self, x, temp=1.0):
        E = self.energy(x)
        p = softmax_np(-E, temp)
        idx = np.random.choice(len(p), p=p)
        self.usage[idx] += 1
        return idx, self.codes[idx], E[idx]

class BeliefGraph_v41:
    def __init__(self):
        self.edges = defaultdict(lambda: defaultdict(float))
        self.confidence = defaultdict(float)

    def update(self, a, b, strength):
        self.edges[a][b] += strength
        self.confidence[(a, b)] = min(1.0, self.confidence[(a, b)] + abs(strength))

class AdaptiveHalt_v41:
    def __init__(self, cost=0.01):
        self.cost = cost
    def halt_prob(self, signal):
        return 1 / (1 + math.exp(-signal))

class IntrospectionModule_v41:
    def __init__(self):
        self.uncertainty = 0.0
        self.hallucination = 0.0
    def update(self, belief_entropy, contradiction):
        self.uncertainty = 0.9 * self.uncertainty + 0.1 * belief_entropy
        self.hallucination = 0.9 * self.hallucination + 0.1 * contradiction
    def penalty(self):
        return self.uncertainty + 2.0 * self.hallucination

class EthicalField_v41:
    def __init__(self, blacklist=None, threshold=0.7):
        self.blacklist = blacklist or set()
    def violation_energy(self, symbol):
        return 5.0 if symbol in self.blacklist else 0.0

class MetaBeliefs_v41:
    def __init__(self):
        self.trust = defaultdict(lambda: 0.5)
    def update(self, belief, success):
        lr = 0.05
        self.trust[belief] += lr * (success - self.trust[belief])

class PriorField_v41:
    def __init__(self, dim):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
    def update(self, x):
        self.mean = 0.99 * self.mean + 0.01 * x
        self.var = 0.99 * self.var + 0.01 * (x - self.mean) ** 2
    def energy(self, x):
        return np.sum((x - self.mean) ** 2 / (self.var + 1e-6))

class SACRSN_v41:
    def __init__(self, dim=32):
        self.dim = dim
        self.memory = SymbolMemory_v41(dim)
        self.graph = BeliefGraph_v41()
        self.halt = AdaptiveHalt_v41()
        self.introspect = IntrospectionModule_v41()
        self.ethics = EthicalField_v41()
        self.meta = MetaBeliefs_v41()
        self.prior = PriorField_v41(dim)
        self.loss_log = []

    def step(self, x, context=None):
        prior_E = self.prior.energy(x)
        idx, sym, sym_E = self.memory.sample(x)
        eth_E = self.ethics.violation_energy(idx)

        if context is not None:
            self.graph.update(context, idx, strength=1.0)

        belief_weights = list(self.graph.edges[idx].values())
        belief_entropy = entropy_np(softmax_np(belief_weights)) if belief_weights else 0.0
        contradiction = max(0.0, sym_E - 1.0)

        self.introspect.update(belief_entropy, contradiction)
        halt_p = self.halt.halt_prob(-sym_E)

        loss = sym_E + prior_E + eth_E + self.introspect.penalty() + self.halt.cost * halt_p
        self.loss_log.append(loss)
        self.prior.update(x)

        return {
            "symbol": idx, "energy": sym_E, "halt_p": halt_p, "loss": loss,
            "uncertainty": self.introspect.uncertainty,
            "hallucination": self.introspect.hallucination,
        }

def run_v41_logic():
    print("\n--- Running SACRSN v41 (NumPy/Math Core) ---")
    engine = SACRSN_v41(dim=16)
    ctx = None
    print(f"{'STEP':<5} | {'SYMBOL':<7} | {'ENERGY':<8} | {'LOSS':<8} | {'UNCERTAINTY'}")
    print("-" * 55)
    for t in range(20):
        x = np.random.randn(16)
        out = engine.step(x, context=ctx)
        ctx = out["symbol"]
        print(f"{t:<5} | {out['symbol']:<7} | {out['energy']:.4f}   | {out['loss']:.4f}   | {out['uncertainty']:.4f}")
    print("Done.\n")


# ==============================================================================
# SHARED PYTORCH PRIMITIVES (For v31 and v45)
# ==============================================================================
if TORCH_AVAILABLE:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    class ComplexLayerNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(dim))
            self.shift = nn.Parameter(torch.zeros(dim))
            self.eps = eps
        def forward(self, z):
            mag = torch.abs(z) + self.eps
            mean = mag.mean(dim=-1, keepdim=True)
            var = mag.var(dim=-1, keepdim=True)
            norm_mag = (mag - mean) / torch.sqrt(var + self.eps)
            norm_mag = norm_mag * self.scale + self.shift
            phase = torch.angle(z)
            return torch.complex(norm_mag * torch.cos(phase), norm_mag * torch.sin(phase))

    class ModReLU(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(dim))
            self.eps = eps
        def forward(self, z):
            norm = torch.abs(z) + self.eps
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


# ==============================================================================
# PART 2: SACRSN v31 — THE OBSERVABLE EDITION
# ==============================================================================
if TORCH_AVAILABLE:
    CONFIG_31 = {
        "seq_len": 32, "embedding_dim": 64, "n_symbols": 64,
        "max_recursion_depth": 8, "act_threshold": 0.9999, "ponder_penalty": 0.0001,
        "use_stack": True, "stack_size": 16,
        "commitment_cost": 0.01, "graph_bias_scale": 0.8,
        "symbol_consistency_weight": 0.01, "ethical_weight": 0.005, "diversity_weight": 0.5,
        "epochs": 200, "learning_rate": 1e-3, "grad_clip": 0.5, "eps": 1e-6
    }
    
    TEXT_DATA_31 = """True, without falsehood, certain and most true. 
    That which is above is like to that which is below, 
    and that which is below is like to that which is above.
    The father of all perfection in the whole world is here.
    Its force or power is entire if it be converted into earth."""

    class DifferentiableStack_v31(nn.Module):
        def __init__(self, dim, size):
            super().__init__()
            self.dim = dim
            self.size = size
        
        def forward(self, z, memory, ptr, control):
            push, pop, noop = control[:, 0].view(-1,1), control[:, 1].view(-1,1), control[:, 2].view(-1,1)
            ptr_up = torch.roll(ptr, 1, dims=1)
            ptr_down = torch.roll(ptr, -1, dims=1)
            new_ptr = (push * ptr_up) + (pop * ptr_down) + (noop * ptr)
            new_ptr = new_ptr / (new_ptr.sum(dim=1, keepdim=True) + CONFIG_31["eps"])
            
            z_flat = torch.cat([z.real, z.imag], dim=-1)
            write_mask = push * ptr_up
            write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)
            retain_mask = 1.0 - write_mask.unsqueeze(2)
            new_memory = write_val + (memory * retain_mask)
            
            read_mask = new_ptr.unsqueeze(2)
            read_flat = torch.sum(new_memory * read_mask, dim=1)
            read_z = torch.complex(read_flat[:, :self.dim], read_flat[:, self.dim:])
            return read_z, new_memory, new_ptr

    class GraphMemoryVQ_v31(nn.Module):
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
                bias = CONFIG_31["graph_bias_scale"] * torch.sigmoid(graph_prior)
                d = d - bias

            min_indices = torch.argmin(d, dim=-1)
            z_q = F.embedding(min_indices, self.codebook)
            loss_vq = F.mse_loss(z_q, z_flat.detach())
            loss_commit = F.mse_loss(z_q.detach(), z_flat)
            z_q = z_flat + (z_q - z_flat).detach()
            z_complex = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
            return z_complex, loss_vq + loss_commit * CONFIG_31["commitment_cost"], min_indices

    class AdaptiveRecursiveCell_v31(nn.Module):
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

    class UberCRSN_v31(nn.Module):
        def __init__(self, vocab_size, dim):
            super().__init__()
            self.dim = dim
            self.emb_mag = nn.Embedding(vocab_size, dim)
            self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
            self.cell = AdaptiveRecursiveCell_v31(dim)
            self.vq_layer = GraphMemoryVQ_v31(dim, CONFIG_31["n_symbols"])
            self.decoder = nn.Linear(dim*2, vocab_size)
            self.stack = DifferentiableStack_v31(dim, CONFIG_31["stack_size"])
            self.register_buffer("prev_sym_soft", torch.zeros(CONFIG_31["n_symbols"]))

        def embed(self, idx):
            r = self.emb_mag(idx)
            t = self.emb_phase[idx]
            return torch.complex(r*torch.cos(t), r*torch.sin(t))

        def forward(self, input_ids, hidden=None, prev_sym=None):
            batch_size = input_ids.size(0)
            z = self.embed(input_ids).squeeze(1)
            
            if hidden is None:
                z_prev = torch.zeros_like(z)
                stack_mem = torch.zeros(batch_size, CONFIG_31["stack_size"], self.dim*2, device=z.device)
                stack_ptr = torch.zeros(batch_size, CONFIG_31["stack_size"], device=z.device)
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
            
            for t in range(CONFIG_31["max_recursion_depth"]):
                act_step += 1
                z_proc, p_halt, stack_ctrl = self.cell(z)
                stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
                z_combined = z_proc + stack_read
                depth = torch.sum(stack_ptr * torch.arange(CONFIG_31["stack_size"], device=z.device), dim=1)
                stack_history.append(depth)

                z_sym, vq_loss, sym_idx = self.vq_layer(z_combined, current_sym)
                
                # Simple ethics check (placeholder)
                if current_sym is not None:
                    row_logits = self.vq_layer.adjacency[current_sym]
                    eth_loss = F.cross_entropy(row_logits.view(-1, CONFIG_31["n_symbols"]), sym_idx.view(-1))
                    ethical_loss_total += eth_loss

                current_sym = sym_idx
                z = 0.7 * z_combined + 0.3 * z_sym
                
                still_running = (halting_probability < CONFIG_31["act_threshold"]).float()
                p = p_halt * still_running
                if t == CONFIG_31["max_recursion_depth"] - 1: p = remain
                
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

    def run_v31_logic():
        print("\n--- Training SACRSN v31 (Observable Edition) ---")
        chars = sorted(list(set(TEXT_DATA_31)))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA_31], dtype=torch.long).to(DEVICE)
        
        model = UberCRSN_v31(len(chars), CONFIG_31["embedding_dim"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG_31["learning_rate"])
        
        try:
            for epoch in range(CONFIG_31["epochs"]):
                hidden = None
                prev_sym = None
                total_loss = 0
                
                for i in range(len(data_tensor) - 1):
                    x = data_tensor[i].view(1, 1)
                    y = data_tensor[i+1].view(1)
                    logits, hidden, sym_idx, ponder, vq_loss, eth_loss, _ = model(x, hidden, prev_sym)
                    
                    # Detach hidden for TBPTT
                    h_z, h_mem, h_ptr = hidden
                    hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach())
                    prev_sym = sym_idx.detach()
                    
                    loss_pred = F.cross_entropy(logits, y)
                    loss = loss_pred + 0.001*ponder + 0.1*vq_loss
                    
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG_31["grad_clip"])
                    opt.step()
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    print(f"Ep {epoch:04d} | Loss: {total_loss/len(data_tensor):.4f}")
                    
        except KeyboardInterrupt:
            print("\nStopped.")
        
        print("Saving visualizations...")
        visualize_v31(model, data_tensor, char_to_ix)
    
    def visualize_v31(model, data_tensor, char_to_ix):
        ix_to_char = {v: k for k, v in char_to_ix.items()}
        model.eval()
        # Simple topology gen
        adj_probs = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
        G = nx.DiGraph()
        for i in range(CONFIG_31["n_symbols"]): 
            for j in range(CONFIG_31["n_symbols"]):
                if adj_probs[i,j] > 0.1: G.add_edge(i, j, weight=adj_probs[i,j])
        
        plt.figure(figsize=(8, 8))
        try: pos = nx.spring_layout(G, k=0.15)
        except: pos = nx.circular_layout(G)
        nx.draw(G, pos, node_size=300, alpha=0.6, node_color='lightblue', with_labels=True, font_size=8)
        plt.title("v31 Semantic Topology")
        plt.savefig("v31_topology.png")
        print("Saved v31_topology.png")
        plt.close()


# ==============================================================================
# PART 3: SACRSN v45 — STABLE HYPER-INTEGRATION
# ==============================================================================
if TORCH_AVAILABLE:
    CONFIG_45 = {
        "seq_len": 32, "dim": 64, "symbols": 64, "stack_size": 16, "max_depth": 8,
        "epochs": 150, "lr": 1e-3, "temp": 1.0,
        "lambda_nll": 1.0, "lambda_halt": 0.01, "lambda_energy": 0.1,
        "meta_alpha": 0.1, "meta_lr": 0.01, "ground_dim": 32, "meta_hierarchy_depth": 2
    }

    class ContextAwareAttention_v45(nn.Module):
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

    class MetaIntrospectionTracker_v45(nn.Module):
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
            h_prior_mixed = CONFIG_45["meta_alpha"] * h_step_prior + (1 - CONFIG_45["meta_alpha"]) * self.meta_h_prior
            
            params = self.posterior_fc(z_flat)
            mu, logvar = torch.chunk(params, 2, dim=-1)
            std = torch.exp(0.5 * logvar)
            h_posterior = mu + torch.randn_like(std) * std
            
            kl_div = 0.5 * ((mu - h_prior_mixed).pow(2) + logvar.exp() - 1 - logvar).sum(dim=1)
            return h_posterior, c_next, kl_div.mean(), mu

        def update_meta_prior(self, batch_mu):
            with torch.no_grad():
                target = batch_mu.mean(0)
                self.meta_h_prior = (1 - CONFIG_45["meta_lr"]) * self.meta_h_prior + CONFIG_45["meta_lr"] * target

    class GroundedMetaGraphVQ_v45(nn.Module):
        def __init__(self, dim, n_symbols):
            super().__init__()
            self.codebook = nn.Parameter(torch.randn(n_symbols, dim*2) * 0.2)
            self.adjacency_energy = nn.Parameter(torch.randn(n_symbols, n_symbols)) 
            self.symbol_ground = nn.Parameter(torch.randn(n_symbols, CONFIG_45["ground_dim"]))
            self.grounding_proj = nn.Linear(dim*2, CONFIG_45["ground_dim"])
            
        def forward(self, z, prev_symbol_dist=None, temp=1.0):
            z_flat = torch.cat([z.real, z.imag], dim=-1)
            d_content = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
                        torch.sum(self.codebook**2, dim=-1) - \
                        2 * torch.matmul(z_flat, self.codebook.t())
            d_content = d_content / z_flat.shape[-1]
            
            if prev_symbol_dist is not None:
                graph_bias = torch.matmul(prev_symbol_dist, self.adjacency_energy)
                d_total = d_content - CONFIG_45["lambda_energy"] * torch.sigmoid(graph_bias)
            else: d_total = d_content

            probs = F.softmax(-d_total / temp, dim=-1)
            if self.training: soft_one_hot = F.gumbel_softmax(-d_total, tau=temp, hard=False)
            else: soft_one_hot = probs
            
            z_q_flat = torch.matmul(soft_one_hot, self.codebook)
            z_q = torch.complex(z_q_flat[..., :z.shape[-1]], z_q_flat[..., z.shape[-1]:])
            
            z_proj = self.grounding_proj(z_flat) 
            expected_anchor = torch.matmul(soft_one_hot, self.symbol_ground)
            grounding_loss = F.mse_loss(z_proj, expected_anchor)
            
            return z_q, soft_one_hot, d_total.mean(), grounding_loss

    class ResidualRecursiveCell_v45(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.linear = ComplexLinear(dim)
            self.norm = ComplexLayerNorm(dim)
            self.act = ModReLU(dim)
            self.attention = ContextAwareAttention_v45(dim)
            self.halt_linear = nn.Linear(dim * 2, 1)
            self.stack_ctrl = nn.Linear(dim * 2, 3)
            nn.init.constant_(self.halt_linear.bias, 2.0)

        def forward(self, z):
            z_proc = self.act(self.norm(self.linear(z)))
            z_proc = self.attention(z_proc, context=torch.cat([z.real, z.imag], dim=-1))
            z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
            halt_logit = self.halt_linear(z_flat)
            stack_logits = self.stack_ctrl(z_flat)
            return z_proc, halt_logit, stack_logits

    class EnhancedUberCRSN_v45(nn.Module):
        def __init__(self, vocab_size, dim):
            super().__init__()
            self.dim = dim
            self.emb_mag = nn.Embedding(vocab_size, dim)
            self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
            self.cell = ResidualRecursiveCell_v45(dim)
            self.vq = GroundedMetaGraphVQ_v45(dim, CONFIG_45["symbols"])
            self.meta_vae = MetaIntrospectionTracker_v45(dim)
            self.decoder = nn.Linear(dim, vocab_size)
            self.quantum_proj = nn.Linear(dim*2, dim) # Simulating quantum probability layer
            
            self.register_buffer("prev_sym_soft", torch.zeros(CONFIG_45["symbols"]))

        def embed(self, idx):
            r = self.emb_mag(idx)
            t = self.emb_phase[idx]
            return torch.complex(r*torch.cos(t), r*torch.sin(t))

        def forward(self, input_ids, hidden=None):
            batch_size = input_ids.size(0)
            z = self.embed(input_ids).squeeze(1)
            
            if hidden is None:
                z_prev = torch.zeros_like(z)
                h_vae = torch.zeros(batch_size, self.dim).to(z.device)
                c_vae = torch.zeros(batch_size, self.dim).to(z.device)
                prev_soft_sym = None
            else:
                z_prev, h_vae, c_vae, prev_soft_sym = hidden
                z = 0.5 * z + 0.5 * z_prev

            halt_cost, energy_cost, kl_cost, ground_cost = 0, 0, 0, 0
            mus_list = []
            
            halting_probability = torch.zeros(batch_size, 1).to(z.device)
            remain = torch.ones(batch_size, 1).to(z.device)
            z_weighted = torch.zeros_like(z)
            current_soft_sym = prev_soft_sym
            
            for t in range(CONFIG_45["max_depth"]):
                z_proc, halt_logit, _ = self.cell(z)
                z = z_proc 
                
                h_vae, c_vae, kl_div, mu = self.meta_vae(z, h_vae, c_vae)
                mus_list.append(mu)
                
                z_q, soft_sym, energy, ground = self.vq(z, current_soft_sym, temp=CONFIG_45["temp"])
                current_soft_sym = soft_sym
                z = 0.7 * z + 0.3 * z_q
                
                p_halt = torch.sigmoid(halt_logit)
                p = torch.minimum(remain, p_halt)
                if t == CONFIG_45["max_depth"] - 1: p = remain
                
                z_weighted = z_weighted + (p * z)
                halting_probability = halting_probability + p
                remain = remain - p
                
                halt_cost += p * (t + 1)
                energy_cost += energy
                kl_cost += kl_div
                ground_cost += ground

            quantum_logits = self.quantum_proj(torch.cat([z_weighted.real, z_weighted.imag], dim=-1))
            logits = self.decoder(quantum_logits)
            
            next_hidden = (z_weighted, h_vae, c_vae, current_soft_sym)
            avg_mu = torch.stack(mus_list).mean(0)
            
            return logits, next_hidden, halt_cost, energy_cost, kl_cost, ground_cost, avg_mu

    def run_v45_logic():
        print("\n--- Training SACRSN v45 (Hyper-Integrated) ---")
        TEXT_DATA_45 = """True, without falsehood, certain and most true. 
        That which is above is like to that which is below, 
        and that which is below is like to that which is above."""
        chars = sorted(list(set(TEXT_DATA_45)))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA_45], dtype=torch.long).to(DEVICE)
        
        model = EnhancedUberCRSN_v45(len(chars), CONFIG_45["dim"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG_45["lr"])
        
        try:
            for epoch in range(CONFIG_45["epochs"]):
                hidden = None
                total_loss = 0
                avg_kl = 0
                
                for i in range(len(data_tensor) - 1):
                    x = data_tensor[i].view(1, 1)
                    y = data_tensor[i+1].view(1)
                    
                    logits, hidden, halt, energy, kl, ground, mu = model(x, hidden)
                    
                    # Detach hidden for TBPTT
                    h_z, h_h, h_c, h_sym = hidden
                    hidden = (h_z.detach(), h_h.detach(), h_c.detach(), 
                              h_sym.detach() if h_sym is not None else None)
                    
                    model.meta_vae.update_meta_prior(mu)
                    
                    nll = F.cross_entropy(logits, y)
                    loss = nll + 0.01*halt + 0.1*energy + 0.1*kl + 0.1*ground
                    
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    total_loss += loss.item()
                    avg_kl += kl.item()

                if epoch % 20 == 0:
                    print(f"Ep {epoch:04d} | Loss: {total_loss/len(data_tensor):.4f} | KL: {avg_kl/len(data_tensor):.4f}")
        except KeyboardInterrupt:
            print("\nStopped.")
            
        print("Dream Mode (v45)...")
        dream_v45(model, char_to_ix)
    
    def dream_v45(model, char_to_ix):
        ix_to_char = {v: k for k, v in char_to_ix.items()}
        model.eval()
        energy_matrix = model.vq.adjacency_energy.detach().cpu()
        curr = 0
        out = "T"
        for _ in range(30):
            logits = -energy_matrix[curr]
            probs = F.softmax(logits, dim=-1).numpy()
            next_idx = np.random.choice(len(probs), p=probs)
            z_flat = model.vq.codebook[next_idx].unsqueeze(0)
            half_dim = z_flat.shape[-1] // 2
            # Simulate projection
            q_logits = model.quantum_proj(z_flat.to(DEVICE))
            char_idx = torch.argmax(model.decoder(q_logits)).item()
            out += ix_to_char.get(char_idx, "?")
            curr = next_idx
        print(f"Dream: {out}\n")


# ==============================================================================
# MAIN MENU
# ==============================================================================

if __name__ == "__main__":
    while True:
        print("======================================================")
        print("SACRSN INTEGRATED ENGINE LOADER")
        print("======================================================")
        print("1. SACRSN v41 (Math/NumPy Core)")
        print("2. SACRSN v31 (Observable/PyTorch)")
        print("3. SACRSN v45 (Hyper-Integrated/PyTorch)")
        print("q. Quit")
        print("======================================================")
        choice = input("Select Version > ").strip().lower()

        if choice == '1':
            run_v41_logic()
        elif choice == '2':
            if TORCH_AVAILABLE:
                run_v31_logic()
            else:
                print("Error: PyTorch not available.")
        elif choice == '3':
            if TORCH_AVAILABLE:
                run_v45_logic()
            else:
                print("Error: PyTorch not available.")
        elif choice == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid selection.")
