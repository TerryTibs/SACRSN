# ============================================================
# SACRSN v32: Unified Script (Fixed In-Place Operations)
# Features: Complex Recursion, Soft Stack, GraphMemoryVQ, Ethical/Diversity Loss
# Includes: Training, Visualization, Dream Mode, Logic Extraction, Anomaly Detection
# ============================================================

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# ============================================================
# 0. Determinism
# ============================================================
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

# ============================================================
# 1. Configuration
# ============================================================
CONFIG = {
    "seq_len": 32,
    "embedding_dim": 128,
    "n_symbols": 128,
    "max_recursion_depth": 8,
    "act_threshold": 0.9999,
    "ponder_penalty": 0.0001,
    "use_stack": True,
    "stack_size": 16,
    "commitment_cost": 0.01,
    "graph_bias_scale": 0.8,
    "symbol_consistency_weight": 0.01,
    "ethical_weight": 0.005,
    "diversity_weight": 0.5,
    "epochs": 3000,
    "learning_rate": 1e-3,
    "grad_clip": 0.5,
    "eps": 1e-6,
    "warmup_epochs": 0
}

# ============================================================
# 2. Data
# ============================================================
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

chars = sorted(list(set(TEXT_DATA)))
vocab_size = len(chars)
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}
data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(DEVICE)

# ============================================================
# 3. Complex Primitives
# ============================================================
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        mag = torch.abs(z)+CONFIG["eps"]
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, keepdim=True)
        norm_mag = (mag-mean)/torch.sqrt(var+CONFIG["eps"])
        norm_mag = norm_mag*self.scale + self.shift
        phase = torch.angle(z)
        return torch.complex(norm_mag*torch.cos(phase), norm_mag*torch.sin(phase))

class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        norm = torch.abs(z)+CONFIG["eps"]
        scale = F.relu(norm+self.bias)/norm
        return z*scale

class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.fc_real.weight)
        nn.init.orthogonal_(self.fc_imag.weight)
    def forward(self, z):
        return torch.complex(
            self.fc_real(z.real)-self.fc_imag(z.imag),
            self.fc_real(z.imag)+self.fc_imag(z.real)
        )

# ============================================================
# 4. Memory Modules
# ============================================================
class DifferentiableStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
    def forward(self, z, memory, ptr, control):
        ptr = ptr.clone()  # avoid in-place
        push, pop, noop = control[:,0].view(-1,1), control[:,1].view(-1,1), control[:,2].view(-1,1)
        ptr_up = torch.roll(ptr,1, dims=1)
        ptr_down = torch.roll(ptr,-1,dims=1)
        new_ptr = (push*ptr_up)+(pop*ptr_down)+(noop*ptr)
        new_ptr = new_ptr/(new_ptr.sum(dim=1, keepdim=True)+CONFIG["eps"])
        z_flat = torch.cat([z.real,z.imag],dim=-1)
        write_mask = push*ptr_up
        write_val = write_mask.unsqueeze(2)*z_flat.unsqueeze(1)
        retain_mask = 1.0-write_mask.unsqueeze(2)
        new_memory = write_val + (memory*retain_mask)
        read_mask = new_ptr.unsqueeze(2)
        read_flat = torch.sum(new_memory*read_mask, dim=1)
        read_z = torch.complex(read_flat[:,:self.dim], read_flat[:,self.dim:])
        return read_z, new_memory, new_ptr

class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols
        self.codebook = nn.Parameter(torch.empty(n_symbols, latent_dim*2))
        nn.init.uniform_(self.codebook, -0.5, 0.5)
        self.adjacency = nn.Parameter(torch.zeros(n_symbols,n_symbols))
    def forward(self, z, prev_symbol_idx=None):
        z_flat = torch.cat([z.real,z.imag],dim=-1)
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + torch.sum(self.codebook**2, dim=-1) - 2*torch.matmul(z_flat, self.codebook.t())
        if prev_symbol_idx is not None:
            graph_prior = self.adjacency[prev_symbol_idx]
            bias = CONFIG["graph_bias_scale"]*torch.sigmoid(graph_prior)
            d = d - bias
        min_indices = torch.argmin(d, dim=-1)
        z_q = F.embedding(min_indices, self.codebook)
        loss_vq = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_flat)
        z_q = z_flat + (z_q - z_flat).detach()
        z_complex = torch.complex(z_q[...,:z.shape[-1]], z_q[...,z.shape[-1]:])
        return z_complex, loss_vq + loss_commit*CONFIG["commitment_cost"], min_indices

# ============================================================
# 5. Core Processor
# ============================================================
class ComplexAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = ComplexLinear(dim)
        self.k_proj = ComplexLinear(dim)
        self.v_proj = ComplexLinear(dim)
        self.scale = dim**-0.5
    def forward(self,z):
        q=self.q_proj(z)
        k=self.k_proj(z)
        v=self.v_proj(z)
        q_flat = torch.cat([q.real,q.imag],dim=-1)
        k_flat = torch.cat([k.real,k.imag],dim=-1)
        attn_scores = torch.matmul(q_flat,k_flat.transpose(-2,-1))*self.scale
        attn_weights = F.softmax(attn_scores,dim=-1)
        v_real = torch.matmul(attn_weights,v.real)
        v_imag = torch.matmul(attn_weights,v.imag)
        return torch.complex(v_real, v_imag)

class EthicalConstraint(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, prev_sym, curr_sym, adjacency):
        if prev_sym is None: return torch.tensor(0.0,device=adjacency.device)
        row_logits = adjacency[prev_sym]
        return F.cross_entropy(row_logits.view(-1,CONFIG["n_symbols"]), curr_sym.view(-1))

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.halt_linear = nn.Linear(dim*2,1)
        self.stack_ctrl = nn.Linear(dim*2,3)
        self.attention = ComplexAttention(dim)
        nn.init.constant_(self.halt_linear.bias,-2.0)
    def forward(self,z):
        z_proc = self.act(self.norm(self.linear(z)))
        z_proc = self.attention(z_proc)
        z_flat = torch.cat([z_proc.real,z_proc.imag],dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        stack_probs = F.softmax(self.stack_ctrl(z_flat),dim=-1)
        return z_proc, halt_prob, stack_probs

# ============================================================
# 6. Master Model
# ============================================================
class UberCRSN(nn.Module):
    def __init__(self,vocab_size,dim):
        super().__init__()
        self.dim=dim
        self.emb_mag = nn.Embedding(vocab_size,dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size,dim))
        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        if CONFIG["use_stack"]:
            self.stack = DifferentiableStack(dim, CONFIG["stack_size"])
        self.ethics = EthicalConstraint()
        self.register_buffer("prev_sym_soft", torch.zeros(CONFIG["n_symbols"]))
    def embed(self, idx):
        r=self.emb_mag(idx)
        t=self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))
    def forward(self,input_ids,hidden=None,prev_sym=None):
        batch_size=input_ids.size(0)
        z=self.embed(input_ids).squeeze(1)
        if hidden is None:
            z_prev = torch.zeros_like(z)
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device)
            stack_ptr[:,0] = 1.0
        else:
            z_prev, stack_mem, stack_ptr = hidden
            z = 0.5*z + 0.5*z_prev
        act_step = 0
        halting_probability = torch.zeros(batch_size,1,device=z.device)
        remain = torch.ones(batch_size,1,device=z.device)
        ponder_cost = 0
        stack_history=[]
        z_weighted=torch.zeros_like(z)
        current_sym = prev_sym
        vq_loss_total=0
        ethical_loss_total=0
        for t in range(CONFIG["max_recursion_depth"]):
            act_step +=1
            z_proc, p_halt, stack_ctrl = self.cell(z)
            if CONFIG["use_stack"]:
                stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
                z_combined = z_proc + stack_read
                depth = torch.sum(stack_ptr*torch.arange(CONFIG["stack_size"],device=z.device),dim=1)
                stack_history.append(depth)
            else:
                z_combined = z_proc
                stack_history.append(torch.zeros(1,device=z.device))
            z_sym, vq_loss, sym_idx = self.vq_layer(z_combined, current_sym)
            eth_loss = self.ethics(current_sym, sym_idx, self.vq_layer.adjacency)
            ethical_loss_total += eth_loss
            current_sym = sym_idx
            z = 0.7*z_combined + 0.3*z_sym
            still_running = (halting_probability<CONFIG["act_threshold"]).float()
            p = p_halt*still_running
            if t==CONFIG["max_recursion_depth"]-1: p = remain
            z_weighted = z_weighted + (p*z)
            halting_probability += p
            remain = remain - p
            ponder_cost += still_running.mean()
            vq_loss_total += vq_loss
        features = torch.cat([z_weighted.real,z_weighted.imag],dim=-1)
        logits = self.decoder(features)
        next_hidden = (z_weighted, stack_mem, stack_ptr)
        avg_stack = torch.stack(stack_history).mean() if len(stack_history)>0 else torch.tensor(0.0)
        return logits, next_hidden, current_sym, ponder_cost, vq_loss_total, ethical_loss_total, avg_stack

# ============================================================
# 7. Training
# ============================================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-5)
    print("\n--- Training SACRSN v32 ---")
    try:
        for epoch in range(CONFIG["epochs"]):
            hidden, prev_sym = None, None
            total_loss = 0
            total_ponder = 0
            total_ppx = 0
            entropy_weight = 0.01*(1 - epoch/CONFIG["epochs"])
            for i in range(len(data_tensor)-1):
                x = data_tensor[i].view(1,1)
                y = data_tensor[i+1].view(1)
                logits, hidden, sym_idx, ponder, vq_loss, eth_loss, _ = model(x, hidden, prev_sym)
                h_z, h_mem, h_ptr = hidden
                hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach())
                prev_sym = sym_idx.detach()
                loss_pred = F.cross_entropy(logits,y)
                loss_ponder = CONFIG["ponder_penalty"]*ponder
                probs = F.softmax(logits,dim=-1)
                loss_entropy = -entropy_weight*((-(probs*torch.log(probs+1e-8)).sum()))
                curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float()
                if curr_onehot.dim()>1: curr_onehot = curr_onehot.view(-1)
                # Fixed in-place update
                model.prev_sym_soft = (model.prev_sym_soft*0.9 + curr_onehot*0.1).detach()
                loss_diversity = CONFIG["diversity_weight"]*(model.prev_sym_soft*torch.log(model.prev_sym_soft+1e-9)).sum()
                loss_ethics = CONFIG["ethical_weight"]*eth_loss
                loss = loss_pred + loss_ponder + 0.1*vq_loss + loss_entropy + loss_diversity + loss_ethics
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                total_loss += loss.item()
                total_ponder += ponder.item()
                usage_dist = model.prev_sym_soft.detach()+1e-10
                entropy_val = -(usage_dist*torch.log(usage_dist)).sum()
                total_ppx += torch.exp(entropy_val).item()
            scheduler.step()
            if epoch%50==0:
                avg_loss = total_loss/len(data_tensor)
                avg_ponder = total_ponder/len(data_tensor)
                avg_ppx = total_ppx/len(data_tensor)
                lr = scheduler.get_last_lr()[0]
                print(f"Ep {epoch:04d} | Loss: {avg_loss:.4f} | Steps: {avg_ponder:.2f} | Usage(PPX): {avg_ppx:.1f} | LR: {lr:.6f}")
                if avg_loss<0.01:
                    print("\n--- PERFECT CONVERGENCE ---")
                    return model
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return model

# ============================================================
# 8. Dream Mode
# ============================================================
def dream_mode(model, start_char=None, length=100, temp=1.0):
    print("\n--- 🌙 Dream Mode ---")
    if start_char is None: start_char = random.choice(list(char_to_ix.keys()))
    idx = torch.tensor([[char_to_ix[start_char]]], device=DEVICE)
    hidden, prev_sym = None, None
    dream_text = start_char
    for _ in range(length):
        with torch.no_grad():
            logits, hidden, prev_sym, ponder, vq_loss, eth_loss, avg_stack = model(idx, hidden, prev_sym)
            probs = F.softmax(logits/temp, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            dream_text += ix_to_char[int(idx_next)]
            idx = idx_next.view(1,1)
    print(f"Dreamed Text:\n{dream_text}\n")
    return dream_text

# ============================================================
# 9. Symbolic Logic Extraction
# ============================================================
def extract_logic_rules(model,data_tensor):
    print("\n--- Extracting Symbolic Logic Rules ---")
    transitions = defaultdict(list)
    hidden, prev_sym = None, None
    for i in range(len(data_tensor)-1):
        x = data_tensor[i].view(1,1)
        with torch.no_grad():
            logits, hidden, sym_idx, ponder, vq_loss, eth_loss, avg_stack = model(x, hidden, prev_sym)
        if prev_sym is not None:
            transitions[int(prev_sym)].append(int(sym_idx))
        prev_sym = sym_idx
    for k,v in transitions.items():
        counts = np.bincount(v,minlength=CONFIG["n_symbols"])
        most_common = np.argmax(counts)
        print(f"Symbol {k} -> Symbol {most_common} ({counts[most_common]} occurrences)")

# ============================================================
# 10. Anomaly Detector
# ============================================================
def anomaly_detector(model, threshold=0.2):
    print("\n--- Anomaly Detection ---")
    hidden, prev_sym = None, None
    anomalies=[]
    for i in range(len(data_tensor)-1):
        x = data_tensor[i].view(1,1)
        with torch.no_grad():
            logits, hidden, sym_idx, ponder, vq_loss, eth_loss, avg_stack = model(x, hidden, prev_sym)
            prob = F.softmax(logits,dim=-1)
            max_prob = prob.max().item()
            if max_prob < threshold:
                anomalies.append((i, ix_to_char[int(data_tensor[i])], max_prob))
            prev_sym = sym_idx
    print(f"Detected {len(anomalies)} anomalies with threshold={threshold}")
    return anomalies

# ============================================================
# 11. Visualization
# ============================================================
def visualize_all(model):
    z_real_list,z_imag_list,mag_list=[],[],[]
    hidden, prev_sym=None,None
    for idx in range(len(data_tensor)-1):
        x=data_tensor[idx].view(1,1)
        with torch.no_grad():
            logits, hidden, sym_idx, ponder, vq_loss, eth_loss, avg_stack = model(x, hidden, prev_sym)
            z_real_list.append(logits.real.cpu().numpy())
            z_imag_list.append(logits.imag.cpu().numpy())
            mag_list.append(np.abs(logits.cpu().numpy()))
            prev_sym=sym_idx
    plt.figure(figsize=(14,5))
    plt.subplot(1,3,1); plt.title("Real"); plt.imshow(np.concatenate(z_real_list,axis=0).T,aspect='auto',cmap='bwr')
    plt.subplot(1,3,2); plt.title("Imag"); plt.imshow(np.concatenate(z_imag_list,axis=0).T,aspect='auto',cmap='bwr')
    plt.subplot(1,3,3); plt.title("Magnitude"); plt.imshow(np.concatenate(mag_list,axis=0).T,aspect='auto',cmap='viridis')
    plt.show()

# ============================================================
# 12. Main Execution
# ============================================================
if __name__=="__main__":
    start_time = time.time()
    model = train()
    visualize_all(model)
    dream_mode(model, length=200)
    extract_logic_rules(model,data_tensor)
    anomalies = anomaly_detector(model,threshold=0.2)
    print(f"\nExecution finished in {time.time()-start_time:.1f} seconds")

