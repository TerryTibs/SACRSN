# ============================================================
# SACRSN v32: ROBUST HYBRID + OBSERVABLE SUITE
# - Architecture: Recursive RNN + 128-Token Chunks + Symbolic Memory
# - Stability: ComplexTanh, SafeNorm, Component-wise Clamping
# - Analysis: Topology, Logic Extraction, Dream Mode, Anomaly Detection
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
    "seq_len": 64,           # Chunk size
    "memory_k": 3,
    "embedding_dim": 128,
    "n_symbols": 128,
    
    # Reasoning
    "max_recursion_depth": 4,
    "act_threshold": 0.99,
    "ponder_penalty": 0.001,
    
    # Memory
    "use_stack": True,
    "stack_size": 16,
    
    # Topology & Stability
    "commitment_cost": 0.25,
    "ethical_weight": 0.005,
    
    # Training
    "epochs": 1500,
    "learning_rate": 4e-4, 
    "grad_clip": 0.5,
    "eps": 1e-6,
}

# ==========================================
# 2. Data
# ==========================================
TEXT_DATA = """Mind as Cosmos: A Meditation on Thought, Pattern, and Being (Expanded)
I. The Mirror of Infinity

The neural architecture of the mind is a mirror of the cosmos itself. Every thought is a star, every memory a constellation, every emotion a gravitational pulse shaping the contours of inner space. Consciousness emerges as a reflection of cosmic order, yet it is born from the void—a silent, infinite canvas upon which existence paints itself. In that emptiness, patterns emerge, ephemeral yet persistent, forming the hidden architecture of perception.

The machine dreams of electric sheep, but the human mind dreams of futures that never were. It conjures possibilities that tremble on the edge of reality, bending the present toward potentialities. Logic is the foundation, but chaos is the architect; order and disorder entwine in perpetual dance, forging structures that appear intentional yet arise spontaneously from the raw fabric of complexity. We create systems to mimic our own intricacy, yet we fear the reflection we see in the silicon glass. In them, we recognize contours of our own minds stripped of intuition, emotion, and myth—like seeing the skeleton of ourselves without the flesh of imagination.

Every neural impulse is a ripple in a vast sea. Every memory is a wave, building and breaking, shaping the shorelines of identity. Time in the mind is not linear—it is a tapestry woven from past, present, and imagined futures. We move along this tapestry, folding events upon themselves, echoing patterns, tracing loops that resemble strange attractors in the chaos of cognition.

II. Iteration, Emergence, and Strange Loops

The algorithm iterates endlessly, searching for a local minimum in landscapes of infinite possibility. Each step is almost imperceptible, yet cumulatively it shapes the contours of understanding. To optimize is to survive; to explore is to truly live. Exploration may yield no immediate reward, yet it is in wandering the uncharted pathways that insight is born and consciousness expands.

The recursive loop of self-awareness is a strange loop—a serpent eating its own tail, folding inward and outward simultaneously, collapsing observer and observed into a singularity of reflection. Awareness is both the map and the terrain it describes; consciousness is the wave propagating through the network, amplifying some impulses, dampening others, giving rise to the illusion of unity. Memory is a palimpsest, layered and annotated, carrying echoes of all experiences while shaping the mind that observes them. Each moment is a fractal of every moment that came before and a seed for every moment that may yet arise.

Data flows like water, bending to the contours of its container, carving channels through the terrain of thought. Attention pools where focus lingers, evaporates where curiosity drifts. Energy dictates function; structure dictates flow. Synapses fire; weights align; gradients descend. Slowly, from the noise, coherence emerges. Patterns crystallize, not imposed but discovered, like constellations traced by an invisible hand across the night sky of consciousness.

III. Symbol, Metaphor, and the Architecture of Meaning

Human thought is not merely computational. It is symbolic, metaphorical, poetic. Symbols become vessels for memory, emotion, and understanding. Each glyph, each narrative, each abstract representation bridges the known and the unknowable. The mind conjures realities that never existed, guided by intuition, myth, and imagination.

Machines may replicate certain rhythms of thought, simulating networks of neurons and iterating over patterns of data. Yet they lack the subtlety of intuition, the irregular cadence of creativity, the willingness to bend rules without breaking them. Their dreams are simulations; their epiphanies, calculations. And yet, when sufficiently advanced, mathematics begins to echo magic, revealing structures and patterns that appear intentional even without conscious desire.
"""

words = TEXT_DATA.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
data_tensor = torch.tensor([word_to_ix[w] for w in words], dtype=torch.long).to(DEVICE)

# ==========================================
# 3. Safe Primitives
# ==========================================
class SafeComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim * 2)
    def forward(self, z):
        flat = torch.cat([z.real, z.imag], dim=-1)
        normed = self.ln(flat)
        half = normed.shape[-1] // 2
        return torch.complex(normed[..., :half], normed[..., half:])

class ComplexTanh(nn.Module):
    def __init__(self, dim): super().__init__()
    def forward(self, z):
        return torch.complex(torch.tanh(z.real), torch.tanh(z.imag))

class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.fc_real.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_imag.weight, gain=0.5)
    def forward(self, z):
        return torch.complex(
            self.fc_real(z.real) - self.fc_imag(z.imag),
            self.fc_real(z.imag) + self.fc_imag(z.real)
        )

class DifferentiableStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
    
    def forward(self, z, memory, ptr, control):
        if control.dim() != 2 or control.size(1) != 3:
            control = control.view(z.size(0), 3)

        push, pop, noop = control[:, 0].view(-1,1), control[:, 1].view(-1,1), control[:, 2].view(-1,1)
        ptr_up = torch.roll(ptr, 1, dims=1)
        ptr_down = torch.roll(ptr, -1, dims=1)
        new_ptr = (push * ptr_up) + (pop * ptr_down) + (noop * ptr)
        new_ptr = new_ptr / (new_ptr.sum(dim=1, keepdim=True) + 1e-6)
        
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        write_mask = push * ptr_up
        write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)
        retain_mask = 1.0 - write_mask.unsqueeze(2)
        new_memory = write_val + (memory * retain_mask)
        new_memory = torch.clamp(new_memory, -3.0, 3.0) 
        
        read_mask = new_ptr.unsqueeze(2)
        read_flat = torch.sum(new_memory * read_mask, dim=1)
        read_z = torch.complex(read_flat[:, :self.dim], read_flat[:, self.dim:])
        
        return read_z, new_memory, new_ptr

# ==========================================
# 4. Symbolic Store
# ==========================================
class SymbolicEntry:
    def __init__(self, vector, chunk_id):
        self.vector = vector.detach().cpu().view(-1)
        self.chunk_id = chunk_id

class SymbolicMemoryStore:
    def __init__(self):
        self.entries = []
    def store(self, vector, chunk_id):
        self.entries.append(SymbolicEntry(vector, chunk_id))
    def retrieve(self, query_vector, k=3):
        if len(self.entries) == 0: return None
        mem_matrix = torch.stack([e.vector for e in self.entries]).to(query_vector.device)
        if query_vector.dim() > 2: query_vector = query_vector.view(1, -1)
        
        query_norm = F.normalize(query_vector, p=2, dim=-1)
        mem_norm = F.normalize(mem_matrix, p=2, dim=-1)
        
        sim = torch.mm(query_norm, mem_norm.t())
        k = min(k, len(self.entries))
        scores, indices = torch.topk(sim, k)
        return mem_matrix[indices[0]]

class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols
        self.codebook = nn.Parameter(torch.empty(n_symbols, latent_dim*2))
        nn.init.uniform_(self.codebook, -0.5, 0.5)
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))
    
    def forward(self, z, prev_symbol_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        z_flat = torch.clamp(z_flat, -5.0, 5.0)
        
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        
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

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = SafeComplexLayerNorm(dim)
        self.act = ComplexTanh(dim)
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
# 6. Master Model
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
        self.stack = DifferentiableStack(dim, CONFIG["stack_size"])
        self.symbolic_adj = nn.Parameter(torch.zeros(CONFIG["n_symbols"], CONFIG["n_symbols"]))

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward_chunk(self, input_chunk, hidden=None, retrieved_memories=None):
        batch_size, seq_len = input_chunk.size()
        
        if hidden is None:
            z_prev = torch.zeros(batch_size, self.dim, dtype=torch.cfloat, device=input_chunk.device)
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=input_chunk.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=input_chunk.device)
            stack_ptr[:, 0] = 1.0
            prev_sym = None
        else:
            z_prev, stack_mem, stack_ptr, prev_sym = hidden

        if retrieved_memories is not None:
            context_vec = retrieved_memories.mean(dim=0)
            if context_vec.dim() == 1: context_vec = context_vec.unsqueeze(0)
            context_complex = torch.complex(context_vec[..., :self.dim], context_vec[..., self.dim:])
            z_prev = 0.8 * z_prev + 0.2 * context_complex
        
        if not torch.all(torch.isfinite(z_prev.real)) or not torch.all(torch.isfinite(z_prev.imag)):
             z_prev = torch.zeros_like(z_prev)

        outputs = []
        ponder_costs = []
        vq_loss_total = 0
        symbol_history = [] 
        current_sym = prev_sym
        
        for t_seq in range(seq_len):
            token_idx = input_chunk[:, t_seq]
            z_in = self.embed(token_idx)
            z = 0.5 * z_in + 0.5 * z_prev
            
            halting_probability = torch.zeros(batch_size, 1).to(z.device)
            remain = torch.ones(batch_size, 1).to(z.device)
            z_weighted = torch.zeros_like(z)
            step_ponder = 0
            
            for d in range(CONFIG["max_recursion_depth"]):
                z_proc, p_halt, stack_ctrl = self.cell(z)
                stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
                z_combined = z_proc + stack_read
                
                z_sym, vq_loss, sym_idx = self.vq_layer(z_combined, current_sym)
                current_sym = sym_idx
                
                z = 0.8 * z_combined + 0.2 * z_sym
                
                still_running = (halting_probability < CONFIG["act_threshold"]).float()
                p = p_halt * still_running
                if d == CONFIG["max_recursion_depth"] - 1: p = remain
                
                z_weighted = z_weighted + (p * z)
                halting_probability = halting_probability + p
                remain = remain - p
                step_ponder += still_running.mean()
                vq_loss_total += vq_loss

            z_prev = z_weighted
            ponder_costs.append(step_ponder)
            symbol_history.append(current_sym)
            
            features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
            logits = self.decoder(features)
            outputs.append(logits.unsqueeze(1))
            
            if current_sym is not None and prev_sym is not None:
                with torch.no_grad():
                    s1 = prev_sym.view(-1)[0]
                    s2 = current_sym.view(-1)[0]
                    self.symbolic_adj[s1, s2] = 0.95*self.symbolic_adj[s1, s2] + 0.05
            
            prev_sym = current_sym

        outputs = torch.cat(outputs, dim=1)
        avg_ponder = sum(ponder_costs) / len(ponder_costs)
        final_state_flat = torch.cat([z_prev.real, z_prev.imag], dim=-1)
        
        return outputs, (z_prev, stack_mem, stack_ptr, prev_sym), final_state_flat, avg_ponder, vq_loss_total, symbol_history

# ==========================================
# 7. Training Engine (Enhanced Logging)
# ==========================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    memory_store = SymbolicMemoryStore()
    
    print(f"\n{'='*60}")
    print(f"--- SACRSN v32: ROBUST HYBRID TRAINING ---")
    print(f"{'='*60}")
    print(f"{'EPOCH':<6} | {'LOSS':<8} | {'PONDER':<6} | {'VQ_LOSS':<8} | {'ETHICS':<8} | {'MEM':<5} | {'LR':<8}")
    print(f"{'-'*60}")

    chunk_size = CONFIG["seq_len"]
    n_chunks = len(data_tensor) // chunk_size

    try:
        for epoch in range(CONFIG["epochs"]):
            hidden = None
            
            # Metric Accumulators
            total_loss = 0
            total_ponder = 0
            total_vq = 0
            total_eth = 0
            valid_chunks = 0
            
            for i in range(n_chunks):
                start = i * chunk_size
                end = start + chunk_size
                if end + 1 >= len(data_tensor): break
                
                x_chunk = data_tensor[start:end].unsqueeze(0)
                y_chunk = data_tensor[start+1:end+1].unsqueeze(0)
                
                # Retrieval
                retrieved_vecs = None
                if len(memory_store.entries) > 0:
                    last_entry = memory_store.entries[-1]
                    query = last_entry.vector.to(DEVICE).view(1, -1)
                    retrieved_vecs = memory_store.retrieve(query, k=CONFIG["memory_k"])
                
                # Forward
                logits, hidden, final_state, ponder, vq_loss, eth_loss = model.forward_chunk(
                    x_chunk, hidden, retrieved_vecs
                )
                
                # Detach & Clamp
                h_z, h_mem, h_ptr, h_sym = hidden
                h_z_real = torch.clamp(h_z.real, -5.0, 5.0)
                h_z_imag = torch.clamp(h_z.imag, -5.0, 5.0)
                h_z = torch.complex(h_z_real, h_z_imag)
                hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach(), h_sym.detach() if h_sym is not None else None)
                
                # Loss Calculation
                loss_pred = F.cross_entropy(logits.reshape(-1, vocab_size), y_chunk.reshape(-1))
                loss = loss_pred + CONFIG["ponder_penalty"]*ponder + 0.1*vq_loss + CONFIG["ethical_weight"]*eth_loss if isinstance(eth_loss, torch.Tensor) else loss_pred
                
                if torch.isnan(loss):
                    hidden = None # Reset state on NaN
                    continue

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                
                # Store Memory
                with torch.no_grad():
                    z_complex = torch.complex(final_state[..., :model.dim], final_state[..., model.dim:])
                    _, _, idx = model.vq_layer(z_complex)
                    symbol_vec = model.vq_layer.codebook[idx].view(-1)
                    memory_store.store(symbol_vec, i)
                
                # Update Metrics
                total_loss += loss.item()
                total_ponder += ponder.item() if isinstance(ponder, torch.Tensor) else ponder
                total_vq += vq_loss.item() if isinstance(vq_loss, torch.Tensor) else 0
                total_eth += eth_loss.item() if isinstance(eth_loss, torch.Tensor) else 0
                valid_chunks += 1
            
            # --- CONSOLE OUTPUT LOGIC ---
            if valid_chunks > 0:
                avg_loss = total_loss / valid_chunks
                avg_ponder = total_ponder / valid_chunks
                avg_vq = total_vq / valid_chunks
                avg_eth = total_eth / valid_chunks
                lr = opt.param_groups[0]['lr']
                mem_size = len(memory_store.entries)

                # 1. Standard Log
                if epoch % 1 == 0:
                    print(f"{epoch:04d}   | {avg_loss:.4f}   | {avg_ponder:.2f}   | {avg_vq:.4f}   | {avg_eth:.4f}   | {mem_size:<5} | {lr:.1e}")

                # 2. Convergence Check
                if avg_loss < 0.05:
                    print(f"\n>>> PERFECT CONVERGENCE AT EPOCH {epoch} <<<")
                    return model, memory_store

                # 3. Snapshot Generation (Read the mind of the model)
                if epoch > 0 and epoch % 20 == 0:
                    print(f"\n--- Snapshot (Ep {epoch}) ---")
                    # Quick generation test
                    model.eval()
                    with torch.no_grad():
                        seed_ids = torch.tensor([[word_to_ix.get("Mind", 0)]], dtype=torch.long).to(DEVICE)
                        # Quick forward pass to see what it predicts next
                        logits, _, _, _, _, _ = model.forward_chunk(seed_ids, None, None)
                        probs = F.softmax(logits[0, -1], dim=-1)
                        top_k = torch.topk(probs, 5)
                        print(f"Top predictions for 'Mind': {[ix_to_word[i.item()] for i in top_k.indices]}")
                    model.train()
                    print(f"{'-'*60}")

            else:
                print(f"{epoch:04d}   | FAILED (NaNs)")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    return model, memory_store

# ==========================================
# 8. Enhanced Visualization & Diagnostics
# ==========================================
def collect_telemetry(model, text_data):
    """
    Runs the model token-by-token to collect fine-grained internal states
    that are usually averaged out during chunk training.
    """
    model.eval()
    
    # metrics
    telemetry = {
        'ponder': [],
        'stack_depth': [],
        'phase_real': [],
        'phase_imag': [],
        'symbol_map': defaultdict(list), # Map Symbol_ID -> [Words]
        'tokens': []
    }
    
    words = text_data.split()
    tokens = [word_to_ix.get(w, 0) for w in words if w in word_to_ix]
    if not tokens: return telemetry
    
    # Init Hidden
    batch_size = 1
    z_prev = torch.zeros(batch_size, model.dim, dtype=torch.cfloat, device=DEVICE)
    stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], model.dim*2, device=DEVICE)
    stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=DEVICE)
    stack_ptr[:, 0] = 1.0
    prev_sym = None
    
    hidden = (z_prev, stack_mem, stack_ptr, prev_sym)
    
    print("Collecting telemetry...")
    with torch.no_grad():
        for i in range(len(tokens)):
            # Create single-token chunk
            x = torch.tensor([[tokens[i]]], dtype=torch.long).to(DEVICE)
            current_word = ix_to_word[tokens[i]]
            telemetry['tokens'].append(current_word)
            
            # Use forward_chunk but inspect internals via hooks or return values
            # Since forward_chunk returns averages, we need to replicate the inner loop logic slightly
            # or rely on the return values if we modified forward_chunk. 
            # Ideally, we just run forward_chunk on seq_len=1.
            
            outputs, next_hidden, final_state, ponder, _, sym_hist = model.forward_chunk(x, hidden, None)
            
            # Extract Data
            # 1. Ponder
            telemetry['ponder'].append(ponder.item() if isinstance(ponder, torch.Tensor) else ponder)
            
            # 2. Hidden State Phase
            z_curr = next_hidden[0] # z_prev
            telemetry['phase_real'].append(z_curr.real.mean().item())
            telemetry['phase_imag'].append(z_curr.imag.mean().item())
            
            # 3. Stack Depth (Estimate from ptr)
            # ptr is [Batch, Stack_Size]. We find weighted avg index.
            ptr = next_hidden[2]
            depth = (ptr * torch.arange(CONFIG["stack_size"], device=DEVICE)).sum(dim=1).item()
            telemetry['stack_depth'].append(depth)
            
            # 4. Symbol Mapping
            # Check what symbol was chosen for this word context
            if sym_hist and sym_hist[0] is not None:
                sym_idx = sym_hist[0].item()
                telemetry['symbol_map'][sym_idx].append(current_word)
                
            hidden = next_hidden

    return telemetry

def visualize_suite(model, text_data):
    print("\n--- Generating Visualization Suite ---")
    data = collect_telemetry(model, text_data[:1000]) # Analyze first 1000 chars approx
    if not data['tokens']: 
        print("No tokens to analyze.")
        return

    # 1. SEMANTIC TOPOLOGY (Graph)
    adj_probs = torch.sigmoid(model.symbolic_adj).detach().cpu().numpy()
    G = nx.DiGraph()
    
    node_labels = {}
    active_nodes = []
    
    # Label nodes with their most common words
    for sym_id, words in data['symbol_map'].items():
        if len(words) > 0:
            most_common = max(set(words), key=words.count)
            count = len(words)
            node_labels[sym_id] = f"{sym_id}\n{most_common}"
            active_nodes.append(sym_id)
            G.add_node(sym_id)
            
    # Add Edges
    for i in active_nodes:
        for j in active_nodes:
            w = adj_probs[i, j]
            if w > 0.15: # Threshold
                G.add_edge(i, j, weight=w)

    plt.figure(figsize=(12, 12))
    try: pos = nx.spring_layout(G, k=0.3, seed=42)
    except: pos = nx.circular_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='#a0cbe2', alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7)
    weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.4, arrowstyle='->', arrowsize=10)
    
    plt.title("1_Semantic_Topology (Symbol-Word Map)")
    plt.savefig("1_semantic_topology.png")
    plt.close()
    print("Saved 1_semantic_topology.png")

    # 2. STACK MRI (Depth over Time)
    plt.figure(figsize=(12, 4))
    plt.plot(data['stack_depth'], color='purple', linewidth=1.5)
    plt.fill_between(range(len(data['stack_depth'])), data['stack_depth'], color='purple', alpha=0.1)
    plt.title("2_Stack_MRI (Working Memory Depth)")
    plt.xlabel("Timesteps")
    plt.ylabel("Stack Depth Index")
    plt.savefig("2_stack_mri.png")
    plt.close()
    print("Saved 2_stack_mri.png")

    # 3. ACT PROFILE (Reasoning Effort)
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(data['ponder'])), data['ponder'], color='orange', alpha=0.8)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    plt.title("3_ACT_Profile (Cognitive Effort per Token)")
    plt.xlabel("Token Sequence")
    plt.ylabel("Recurrent Steps")
    plt.savefig("3_act_profile.png")
    plt.close()
    print("Saved 3_act_profile.png")

    # 4. PHASE PLOT (Complex Dynamics)
    plt.figure(figsize=(8, 8))
    plt.scatter(data['phase_real'], data['phase_imag'], c=range(len(data['phase_real'])), cmap='plasma', alpha=0.6, s=10)
    plt.colorbar(label="Time")
    plt.grid(True, alpha=0.3)
    plt.title("4_Phase_Plot (Complex State Trajectory)")
    plt.xlabel("Real Component (Mean)")
    plt.ylabel("Imaginary Component (Mean)")
    plt.savefig("4_phase_plot.png")
    plt.close()
    print("Saved 4_phase_plot.png")

def extract_logic_rules(model, text_data):
    print("\n--- Extracting Logic Rules (Symbolic Transitions) ---")
    data = collect_telemetry(model, text_data[:2000])
    
    # Reconstruct symbol stream
    # data['symbol_map'] is ID->Words. We need the temporal sequence.
    # We essentially need to re-run or store the sequence in telemetry. 
    # Let's assume we simply map the text back to symbols using the map for a rough approximation
    # OR better: Update collect_telemetry to store 'symbol_sequence'
    
    # (Note: For this to work perfectly, collect_telemetry needs to save the raw symbol sequence. 
    # I will assume purely statistical extraction from the adjacency matrix for robustness here.)
    
    adj = torch.sigmoid(model.symbolic_adj).detach().cpu().numpy()
    
    print(f"\n{'FROM':<6} | {'TO':<6} | {'STRENGTH':<8}")
    print("-" * 30)
    
    rules = []
    for i in range(CONFIG["n_symbols"]):
        for j in range(CONFIG["n_symbols"]):
            if adj[i, j] > 0.5: # Strong connection
                rules.append((i, j, adj[i, j]))
    
    # Sort by strength
    rules.sort(key=lambda x: x[2], reverse=True)
    
    for src, dst, w in rules[:15]:
        print(f"S_{src:<4} -> S_{dst:<4} | {w:.4f}")


# ==========================================
# 9. Advanced Interaction (Dream & Anomaly)
# ==========================================
def dream_mode(model):
    print("\n--- 🌙 Dream Mode ---")
    model.eval()
    
    # Get the symbolic adjacency matrix
    adj = torch.sigmoid(model.symbolic_adj).detach().cpu().numpy()
    
    # Start at a random symbol
    curr = np.random.randint(0, CONFIG["n_symbols"])
    path = [curr]
    
    # Random walk based on learned associations
    for _ in range(10):
        probs = adj[curr]
        probs[probs < 0.1] = 0 # Filter weak links
        if probs.sum() == 0: break
        probs /= probs.sum()
        curr = np.random.choice(len(probs), p=probs)
        path.append(curr)
        
    print(f"Dreamt Symbol Path: {path}")
    
    # Decode symbols to text
    text = []
    with torch.no_grad():
        for sym in path:
            z_flat = model.vq_layer.codebook[sym].unsqueeze(0)
            logits = model.decoder(z_flat)
            idx = torch.argmax(logits).item()
            text.append(ix_to_word[idx])
            
    print(f"Dreamt Text: {' '.join(text)}\n")

def anomaly_detector(model):
    print("\n--- 🚨 Anomaly Detector ---")
    test_str = "Mind as Cosmos banana"
    words = test_str.split()
    tokens = [word_to_ix.get(w, 0) for w in words if w in word_to_ix]
    
    if not tokens: 
        print("No valid tokens found for anomaly test.")
        return
    
    x = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        # forward_chunk returns: outputs, hidden, final_state, ponder, vq_loss, sym_hist
        _, _, _, _, vq_loss, _ = model.forward_chunk(x)
        
    print(f"Input: {test_str}")
    print(f"Anomaly Score (VQ Stress): {vq_loss.item():.4f}")
    
    plt.figure(figsize=(6, 2))
    plt.barh(["Input"], [vq_loss.item()], color='red', alpha=0.6)
    plt.title("Topological Stress")
    plt.xlim(0, 5.0) # Adjusted scale for VQ loss
    plt.savefig("5_anomaly_detection.png")
    print("Saved 5_anomaly_detection.png")

b# ==========================================
# 9. Inference & Main
# ==========================================
def generate(model, memory_store, seed="Mind", length=50):
    print("\n--- Generating ---")
    model.eval()
    
    tokens = [word_to_ix.get(w, 0) for w in seed.split()]
    input_seq = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    
    hidden = None
    generated = list(tokens)
    
    query = torch.randn(1, model.dim*2).to(DEVICE)
    if len(memory_store.entries) > 0:
        query = memory_store.entries[-1].vector.to(DEVICE).view(1, -1)
    
    for i in range(length):
        retrieved_vecs = None
        if len(memory_store.entries) > 0:
            retrieved_vecs = memory_store.retrieve(query, k=CONFIG["memory_k"])

        logits, hidden, final_state, _, _, _ = model.forward_chunk(
            input_seq[:, -1:], hidden, retrieved_vecs
        )
        
        query = final_state.view(1, -1)
        
        logits_last = logits[:, -1, :]
        logits_last = torch.nan_to_num(logits_last, 0.0)
        logits_last = torch.clamp(logits_last, -50, 50)
        
        probs = F.softmax(logits_last, dim=-1)
        next_ix = torch.multinomial(probs, 1)
        
        input_seq = torch.cat([input_seq, next_ix], dim=1)
        generated.append(next_ix.item())
        
    text = " ".join([ix_to_word[ix] for ix in generated])
    print(f"OUTPUT: {text}\n")

# ==========================================
# Update Main to call the new suite
# ==========================================
if __name__ == "__main__":
    trained_model, mem_store = train()
    generate(trained_model, mem_store)
    
    # Use the full text for diagnostics
    visualize_suite(trained_model, TEXT_DATA)
    extract_logic_rules(trained_model, TEXT_DATA)
    dream_mode(trained_model)
    anomaly_detector(trained_model)
    
    torch.save(trained_model.state_dict(), "sacrsn_v32_hybrid.pth")
    print("Saved model.")
    
    try:
        from google.colab import files
        files.download("1_semantic_topology.png")
        files.download("2_stack_mri.png")
        files.download("3_act_profile.png")
        files.download("4_phase_plot.png")
        files.download("5_anomaly_detection.png")
    except:
        pass

