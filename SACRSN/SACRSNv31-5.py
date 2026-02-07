# ============================================================
# SACRSN v34: THE BI-CAMERAL SUITE (COMPLETE MONOLITH)
# Architecture: Dual-VQ Neuro-Symbolic Network (Syntax + Semantics)
# Tokenization: Custom BPE (Sub-word abstraction)
# Training: Masked Bidirectional Prediction
# Diagnostics: Dual Topology, Logic Extraction, Dream Overlay
# ============================================================

import os
import time
import random
import requests
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
SEED = 1337
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
    "seq_len": 64,                 # BPE allows shorter seq for same content
    "embedding_dim": 128,
    "n_heads": 4,
    
    # Dual VQ Settings
    "n_syntax_symbols": 64,        # Fast Grammar Rules
    "n_semantic_symbols": 128,     # Slow Concept Rules
    
    # Reasoning
    "max_recursion_depth": 10,
    "act_threshold": 0.99,
    "ponder_penalty": 0.0005,
    
    # Memory
    "use_stack": True,
    "stack_size": 32,
    
    # Training
    "mask_prob": 0.15,             # 15% Masking for robustness
    "epochs": 15,                  # 15 epochs is usually sufficient for BPE
    "batch_size": 64,
    "learning_rate": 8e-4,
    "grad_clip": 1.0,
    "warmup_epochs": 1
}

# ==========================================
# 2. Custom BPE Tokenizer (Robust)
# ==========================================
class SimpleBPE:
    def __init__(self, vocab_size=800):
        self.target_vocab_size = vocab_size
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)} # Base byte vocab
        self.special_tokens = {"<PAD>": 256, "<UNK>": 257, "<MASK>": 258}
        self.next_id = 259

    def get_stats(self, ids):
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def train(self, text):
        print("Training BPE Tokenizer...")
        ids = list(text.encode("utf-8"))
        num_merges = self.target_vocab_size - self.next_id
        
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats: break
            pair = max(stats, key=stats.get)
            idx = self.next_id
            self.merges[pair] = idx
            
            # Update vocab map for decoding
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.next_id += 1
            
            # Recursive replace
            new_ids = []
            skip = False
            for j in range(len(ids)):
                if skip:
                    skip = False
                    continue
                if j < len(ids) - 1 and ids[j] == pair[0] and ids[j+1] == pair[1]:
                    new_ids.append(idx)
                    skip = True
                else:
                    new_ids.append(ids[j])
            ids = new_ids
            
            if i % 100 == 0: print(f"Merge {i}: {pair} -> {idx}")
        
        self.vocab_size = self.next_id
        print(f"Final Vocab Size: {self.vocab_size}")

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for pair, idx in self.merges.items():
            new_ids = []
            skip = False
            for j in range(len(ids)):
                if skip:
                    skip = False
                    continue
                if j < len(ids) - 1 and ids[j] == pair[0] and ids[j+1] == pair[1]:
                    new_ids.append(idx)
                    skip = True
                else:
                    new_ids.append(ids[j])
            ids = new_ids
        return ids

    def decode(self, ids):
        out_bytes = b""
        for idx in ids:
            if idx in self.vocab:
                out_bytes += self.vocab[idx]
            elif idx == self.special_tokens["<MASK>"]:
                out_bytes += b"[MASK]"
            else:
                out_bytes += b"?"
        return out_bytes.decode("utf-8", errors="replace")

# --- LOAD DATA ---
FILE_PATH = "tinyshakespeare.txt"
if not os.path.exists(FILE_PATH):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(FILE_PATH, 'w') as f: f.write(requests.get(url).text)

with open(FILE_PATH, 'r') as f: raw_data = f.read()

# Train BPE
tokenizer = SimpleBPE(vocab_size=800)
# Train on a subset to save startup time
tokenizer.train(raw_data[:50000])

print("Encoding full dataset...")
tokenized_data = tokenizer.encode(raw_data)
data_tensor = torch.tensor(tokenized_data, dtype=torch.long).to(DEVICE)
VOCAB_SIZE = tokenizer.vocab_size

# ==========================================
# 3. Complex Primitives
# ==========================================
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        mag = torch.abs(z) + 1e-6
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, keepdim=True)
        norm_mag = (mag - mean) / torch.sqrt(var + 1e-6)
        norm_mag = norm_mag * self.scale + self.shift
        phase = torch.angle(z)
        return torch.complex(norm_mag * torch.cos(phase), norm_mag * torch.sin(phase))

class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        norm = torch.abs(z) + 1e-6
        scale = F.relu(norm + self.bias) / norm
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_real = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_imag = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, z):
        return torch.complex(
            self.fc_real(z.real) - self.fc_imag(z.imag),
            self.fc_real(z.imag) + self.fc_imag(z.real)
        )

# ==========================================
# 4. Multi-Scale VQ (Syntax & Semantics)
# ==========================================
class DualScaleVQ(nn.Module):
    def __init__(self, latent_dim, n_syntax, n_semantic, decay=0.99):
        super().__init__()
        self.dim = latent_dim * 2
        self.decay = decay
        
        # Codebooks
        self.cb_syn = nn.Parameter(torch.randn(n_syntax, self.dim))
        self.cb_sem = nn.Parameter(torch.randn(n_semantic, self.dim))
        
        # EMA Buffers
        self.register_buffer("cl_syn", torch.zeros(n_syntax))
        self.register_buffer("avg_syn", self.cb_syn.clone())
        self.register_buffer("cl_sem", torch.zeros(n_semantic))
        self.register_buffer("avg_sem", self.cb_sem.clone())
        
        # Topology Tracking (For Visualization & Logic Extraction)
        self.register_buffer("adj_syn", torch.zeros(n_syntax, n_syntax))
        self.register_buffer("adj_sem", torch.zeros(n_semantic, n_semantic))

    def vq_step(self, z, codebook, cluster_size, embed_avg, adj_matrix, prev_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        d = torch.sum(z_flat**2, dim=1, keepdim=True) + \
            torch.sum(codebook**2, dim=1) - \
            2 * torch.matmul(z_flat, codebook.t())
        
        idx = torch.argmin(d, dim=1)
        z_q = F.embedding(idx, codebook)
        
        # Update Topology (Hebbian-ish counting)
        if prev_idx is not None and self.training:
             with torch.no_grad():
                # vectorized update or loop for safety
                for i in range(len(idx)):
                    r, c = prev_idx[i].item(), idx[i].item()
                    adj_matrix[r, c] = adj_matrix[r, c] * 0.99 + 1.0 # Moving avg count

        if self.training:
            enc = F.one_hot(idx, codebook.size(0)).float()
            cluster_size.data.mul_(self.decay).add_(enc.sum(0), alpha=1-self.decay)
            embed_sum = torch.matmul(enc.t(), z_flat.detach())
            embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
            n = cluster_size.sum()
            cs = (cluster_size + 1e-6) / (n + codebook.size(0)*1e-6) * n
            codebook.data.copy_(embed_avg / cs.unsqueeze(1))
            
        loss = F.mse_loss(z_q.detach(), z_flat) + 0.25 * F.mse_loss(z_q, z_flat.detach())
        z_q = z_flat + (z_q - z_flat).detach()
        return torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:]), loss, idx

    def forward(self, z_fast, z_slow, prev_indices=None):
        prev_syn, prev_sem = (None, None) if prev_indices is None else prev_indices
        
        # 1. Syntax VQ (RNN State)
        zq_syn, loss_syn, idx_syn = self.vq_step(z_fast, self.cb_syn, self.cl_syn, self.avg_syn, self.adj_syn, prev_syn)
        
        # 2. Semantic VQ (Stack State)
        zq_sem, loss_sem, idx_sem = self.vq_step(z_slow, self.cb_sem, self.cl_sem, self.avg_sem, self.adj_sem, prev_sem)
        
        return zq_syn, zq_sem, loss_syn + loss_sem, idx_syn, idx_sem

# ==========================================
# 5. Core Model (Bi-Cameral Architecture)
# ==========================================
class BiCameralCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        
        self.linear = ComplexLinear(dim, dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.halt = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        
        self.vq = DualScaleVQ(dim, CONFIG["n_syntax_symbols"], CONFIG["n_semantic_symbols"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        self.mask_token = nn.Parameter(torch.randn(dim*2))

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, x_idx, hidden=None, mask_mask=None, prev_vq_indices=None):
        batch_size = x_idx.size(0)
        z = self.embed(x_idx).squeeze(1)
        
        # Masking Logic
        if mask_mask is not None:
            z_flat = torch.cat([z.real, z.imag], dim=-1)
            mask_vec = self.mask_token.unsqueeze(0).expand(batch_size, -1)
            z_masked_flat = z_flat * (1 - mask_mask) + mask_vec * mask_mask
            z = torch.complex(z_masked_flat[..., :self.dim], z_masked_flat[..., self.dim:])
            
        if hidden is None:
            z_prev = torch.zeros_like(z)
            stack = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
            ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device)
            ptr[:, 0] = 1.0
        else:
            z_prev, stack, ptr = hidden
            z = 0.5 * z + 0.5 * z_prev

        halt_prob = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        ponder = torch.zeros((), device=z.device)
        
        z_out = torch.zeros_like(z)
        final_stack_read = torch.zeros_like(z)

        # ACT Loop
        for t in range(CONFIG["max_recursion_depth"]):
            z_proc = self.act(self.norm(self.linear(z)))
            z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
            
            # Stack Logic
            ctrl = F.softmax(self.stack_ctrl(z_flat), dim=-1)
            ptr_up = torch.roll(ptr, 1, 1)
            ptr_down = torch.roll(ptr, -1, 1)
            new_ptr = (ctrl[:,0:1]*ptr_up) + (ctrl[:,1:2]*ptr_down) + (ctrl[:,2:3]*ptr)
            new_ptr = new_ptr / (new_ptr.sum(1, keepdim=True) + 1e-6)
            
            write_mask = ctrl[:,0:1] * ptr_up
            stack_new = stack * (1 - write_mask.unsqueeze(2)) + z_flat.unsqueeze(1) * write_mask.unsqueeze(2)
            
            read_vec = (stack_new * new_ptr.unsqueeze(2)).sum(1)
            stack_read = torch.complex(read_vec[..., :self.dim], read_vec[..., self.dim:])
            
            z_combined = z_proc + stack_read
            final_stack_read = stack_read

            h = torch.sigmoid(self.halt(z_flat))
            still_running = (halt_prob < CONFIG["act_threshold"]).float()
            p = torch.minimum(h * still_running, remain)
            if t == CONFIG["max_recursion_depth"]-1: p = remain
            
            z_out = z_out + p * z_combined
            halt_prob = halt_prob + p
            remain = remain - p
            ponder = ponder + still_running.mean()
            
            z = z_combined
            stack = stack_new
            ptr = new_ptr
            
            if remain.max() < 1e-6: break

        # Dual VQ
        zq_syn, zq_sem, vq_loss, idx_syn, idx_sem = self.vq(z_out, final_stack_read, prev_vq_indices)
        
        z_final = z_out + zq_syn + zq_sem
        features = torch.cat([z_final.real, z_final.imag], dim=-1)
        logits = self.decoder(features)
        
        return logits, (z_out, stack, ptr), (idx_syn, idx_sem), ponder, vq_loss

# ==========================================
# 6. Training (Masked & Logged)
# ==========================================
def train():
    model = BiCameralCRSN(VOCAB_SIZE, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    # Data prep
    num_samples = len(data_tensor) // CONFIG["seq_len"]
    trim = num_samples * CONFIG["seq_len"]
    x_data = data_tensor[:trim].view(num_samples, CONFIG["seq_len"])
    y_data = torch.roll(data_tensor, -1)[:trim].view(num_samples, CONFIG["seq_len"])
    
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=CONFIG["batch_size"], shuffle=True)
    
    print("--- Training SACRSN v34 (Bi-Cameral BPE) ---")
    
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        start_time = time.time()
        
        for i, (x, y) in enumerate(loader):
            hidden, prev_indices = None, None
            loss_batch = 0
            ponder_accum = 0
            ppx_accum = 0
            
            # Create random mask
            mask_mask = (torch.rand_like(x.float()) < CONFIG["mask_prob"]).float().unsqueeze(-1).to(DEVICE)
            
            for t in range(CONFIG["seq_len"]):
                xt = x[:, t:t+1]
                yt = y[:, t]
                mt = mask_mask[:, t, :]
                
                logits, hidden, curr_indices, ponder, vq_loss = model(xt, hidden, mt, prev_indices)
                
                # Detach indices to stop gradient through topology update
                prev_indices = (curr_indices[0].detach(), curr_indices[1].detach())
                
                loss_pred = F.cross_entropy(logits, yt)
                loss_batch += loss_pred + CONFIG["ponder_penalty"]*ponder + vq_loss
                
                ponder_accum += ponder.mean().item()
                ppx_accum += torch.exp(loss_pred).item()
                
            loss_batch = loss_batch / CONFIG["seq_len"]
            
            opt.zero_grad()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
            
            # Full Detach
            hz, hs, hp = hidden
            hidden = (hz.detach(), hs.detach(), hp.detach())
            
            total_loss += loss_batch.item()
            
            if i % 20 == 0:
                avg_steps = 1.0 + (ponder_accum / CONFIG["seq_len"])
                avg_ppx = ppx_accum / CONFIG["seq_len"]
                print(f"Ep {epoch} | Batch {i:03d} | Loss: {loss_batch.item():.4f} | Steps: {avg_steps:.2f} | PPX: {avg_ppx:.1f}")
                
        print(f"Epoch {epoch} Done. Avg Loss: {total_loss/len(loader):.4f} | Time: {time.time()-start_time:.1f}s")
        if epoch % 1 == 0: print(generate_sample(model))
        
    return model

def generate_sample(model):
    model.eval()
    # Start with <UNK> as a seed
    x = torch.tensor([[tokenizer.special_tokens["<UNK>"]]], device=DEVICE)
    hidden, prev_idx = None, None
    out_ids = []
    with torch.no_grad():
        for _ in range(100):
            logits, hidden, idx, _, _ = model(x, hidden, None, prev_idx)
            prev_idx = idx
            probs = F.softmax(logits, dim=-1)
            x = torch.multinomial(probs, 1)
            out_ids.append(x.item())
    model.train()
    return f"Sample: {tokenizer.decode(out_ids)}\n"

# ==========================================
# 7. Complete Visualization Suite
# ==========================================
def visualize_all(model):
    print("\n--- Visualizing Bi-Cameral Topology ---")
    model.eval()
    
    # 1. Dual Topology Plot (Heatmaps of Adjacency)
    adj_syn = model.vq.adj_syn.cpu().detach().numpy()
    adj_sem = model.vq.adj_sem.cpu().detach().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(np.log1p(adj_syn), cmap='Blues')
    ax1.set_title("Syntax (Grammar)")
    
    ax2.imshow(np.log1p(adj_sem), cmap='Reds')
    ax2.set_title("Semantics (Stack)")
    
    plt.tight_layout()
    plt.savefig("1_bicameral_topology.png")
    plt.close()

    # 2. Logic Extraction
    print("\n--- Syntax Logic (Fast System) ---")
    # Get top transitions
    indices = np.argsort(adj_syn.flatten())[::-1][:5]
    for idx in indices:
        r, c = np.unravel_index(idx, adj_syn.shape)
        if adj_syn[r, c] > 0: 
            print(f"Syn_{r} -> Syn_{c} (Strength: {adj_syn[r,c]:.0f})")

    print("\n--- Semantic Logic (Slow System) ---")
    indices = np.argsort(adj_sem.flatten())[::-1][:5]
    for idx in indices:
        r, c = np.unravel_index(idx, adj_sem.shape)
        if adj_sem[r, c] > 0: 
            print(f"Sem_{r} -> Sem_{c} (Strength: {adj_sem[r,c]:.0f})")

    # 3. Dream Diagnostic & 4D Heatmap
    print("\nGenerating Dream Data...")
    # Encode a starter prompt
    start_ids = tokenizer.encode("The King")
    x = torch.tensor([start_ids], device=DEVICE)
    hidden, prev_idx = None, None
    
    # Warmup
    with torch.no_grad():
        for t in range(x.size(1)):
             logits, hidden, idx, _, _ = model(x[:, t:t+1], hidden, None, prev_idx)
             prev_idx = idx
    
    # Generate
    out_tokens = ["The", "King"]
    x = torch.multinomial(F.softmax(logits, dim=-1), 1)
    
    syn_indices, sem_indices = [], []
    stack_depths, act_loads = [], []
    
    with torch.no_grad():
        for _ in range(50):
            logits, hidden, idx, ponder, _ = model(x, hidden, None, prev_idx)
            prev_idx = idx
            
            syn_indices.append(idx[0][0].item())
            sem_indices.append(idx[1][0].item())
            act_loads.append(ponder.item())
            
            # Stack depth
            _, _, ptr = hidden
            depth = (ptr[0] * torch.arange(CONFIG["stack_size"], device=DEVICE)).sum().item()
            stack_depths.append(depth)
            
            probs = F.softmax(logits, dim=-1)
            x = torch.multinomial(probs, 1)
            
            decoded = tokenizer.decode([x.item()])
            out_tokens.append(decoded)
            
    print(f"Dream Sequence: {''.join(out_tokens)}")
    
    # Plot 4D Heatmap
    def normalize(lst):
        arr = np.array(lst)
        if arr.max() == arr.min(): return arr
        return (arr - arr.min()) / (arr.max() - arr.min())
        
    matrix = np.vstack([normalize(syn_indices), normalize(sem_indices), normalize(stack_depths), normalize(act_loads)])
    
    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, aspect='auto', cmap='inferno', interpolation='nearest')
    plt.yticks(range(4), ['Syntax ID', 'Semantic ID', 'Stack Depth', 'Cognitive Load'])
    plt.xticks(range(len(out_tokens)-2), out_tokens[2:], rotation=45, ha='right')
    plt.title("Bi-Cameral Logic Heatmap")
    plt.tight_layout()
    plt.savefig("2_bicameral_heatmap.png")
    plt.close()

# ==========================================
# 8. Main
# ==========================================
if __name__ == "__main__":
    FILENAME = "sacrsn_v34_bicameral_complete.pth"
    model = train()
    
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({'model': model.state_dict(), 'config': CONFIG, 'vocab': tokenizer.vocab}, FILENAME)
    
    visualize_all(model)
