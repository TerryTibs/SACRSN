# ============================================================
# SACRSN v31 Unified TinyShakespeare Edition
# Features: GPU-batched training, live dashboard, dream mode, anomaly detection
# ============================================================

import os, time, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx

# ==========================================
# 0. Determinism
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# ==========================================
# 1. Configuration
# ==========================================
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

# ==========================================
# 2. TinyShakespeare Data
# ==========================================
with open("tinyshakespeare.txt", "r", encoding="utf-8") as f: TEXT_DATA = f.read()
chars = sorted(list(set(TEXT_DATA)))
vocab_size = len(chars)
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}
data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(DEVICE)

# ==========================================
# 3. Complex Primitives
# ==========================================
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.scale=nn.Parameter(torch.ones(dim)); self.shift=nn.Parameter(torch.zeros(dim))
    def forward(self,z):
        mag=torch.abs(z)+CONFIG["eps"]; mean=mag.mean(dim=-1,keepdim=True); var=mag.var(dim=-1,keepdim=True)
        norm_mag=(mag-mean)/torch.sqrt(var+CONFIG["eps"]); norm_mag=norm_mag*self.scale+self.shift
        phase=torch.angle(z)
        return torch.complex(norm_mag*torch.cos(phase), norm_mag*torch.sin(phase))

class ModReLU(nn.Module):
    def __init__(self,dim): super().__init__(); self.bias=nn.Parameter(torch.zeros(dim))
    def forward(self,z):
        norm=torch.abs(z)+CONFIG["eps"]; scale=F.relu(norm+self.bias)/norm; return z*scale

class ComplexLinear(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.fc_real=nn.Linear(dim,dim,bias=False); self.fc_imag=nn.Linear(dim,dim,bias=False)
        nn.init.orthogonal_(self.fc_real.weight); nn.init.orthogonal_(self.fc_imag.weight)
    def forward(self,z):
        return torch.complex(
            self.fc_real(z.real)-self.fc_imag(z.imag),
            self.fc_real(z.imag)+self.fc_imag(z.real)
        )

# ==========================================
# 4. Memory Modules
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self,dim,size): super().__init__(); self.dim=dim; self.size=size
    def forward(self,z,memory,ptr,control):
        push,pop,noop=control[:,0].view(-1,1),control[:,1].view(-1,1),control[:,2].view(-1,1)
        ptr_up=torch.roll(ptr,1,dims=1); ptr_down=torch.roll(ptr,-1,dims=1)
        new_ptr=(push*ptr_up)+(pop*ptr_down)+(noop*ptr); new_ptr=new_ptr/(new_ptr.sum(dim=1,keepdim=True)+CONFIG["eps"])
        z_flat=torch.cat([z.real,z.imag],dim=-1)
        write_mask=push*ptr_up; write_val=write_mask.unsqueeze(2)*z_flat.unsqueeze(1); retain_mask=1.0-write_mask.unsqueeze(2)
        new_memory=write_val+(memory*retain_mask)
        read_mask=new_ptr.unsqueeze(2); read_flat=torch.sum(new_memory*read_mask,dim=1)
        read_z=torch.complex(read_flat[:,:self.dim], read_flat[:,self.dim:])
        return read_z,new_memory,new_ptr

class GraphMemoryVQ(nn.Module):
    def __init__(self,latent_dim,n_symbols):
        super().__init__(); self.n_symbols=n_symbols
        self.codebook=nn.Parameter(torch.empty(n_symbols,latent_dim*2)); nn.init.uniform_(self.codebook,-0.5,0.5)
        self.adjacency=nn.Parameter(torch.zeros(n_symbols,n_symbols))
    def forward(self,z,prev_symbol_idx=None):
        z_flat=torch.cat([z.real,z.imag],dim=-1)
        d=torch.sum(z_flat**2,dim=-1,keepdim=True)+torch.sum(self.codebook**2,dim=-1)-2*torch.matmul(z_flat,self.codebook.t())
        if prev_symbol_idx is not None:
            graph_prior=self.adjacency[prev_symbol_idx]; bias=CONFIG["graph_bias_scale"]*torch.sigmoid(graph_prior); d=d-bias
        min_indices=torch.argmin(d,dim=-1); z_q=F.embedding(min_indices,self.codebook)
        loss_vq=F.mse_loss(z_q,z_flat.detach()); loss_commit=F.mse_loss(z_q.detach(),z_flat)
        z_q=z_flat+(z_q-z_flat).detach(); z_complex=torch.complex(z_q[...,:z.shape[-1]], z_q[...,z.shape[-1]:])
        return z_complex, loss_vq+loss_commit*CONFIG["commitment_cost"], min_indices

# ==========================================
# 5. Core Processor
# ==========================================
class ComplexAttention(nn.Module):
    def __init__(self,dim): super().__init__(); self.q_proj=self.k_proj=self.v_proj=ComplexLinear(dim); self.scale=dim**-0.5
    def forward(self,z):
        q=self.q_proj(z); k=self.k_proj(z); v=self.v_proj(z)
        q_flat=torch.cat([q.real,q.imag],dim=-1); k_flat=torch.cat([k.real,k.imag],dim=-1)
        attn_scores=torch.matmul(q_flat,k_flat.transpose(-2,-1))*self.scale
        attn_weights=F.softmax(attn_scores,dim=-1)
        v_real=torch.matmul(attn_weights,v.real); v_imag=torch.matmul(attn_weights,v.imag)
        return torch.complex(v_real,v_imag)

class EthicalConstraint(nn.Module):
    def __init__(self): super().__init__()
    def forward(self,prev_sym,curr_sym,adjacency):
        if prev_sym is None: return torch.tensor(0.0).to(adjacency.device)
        row_logits=adjacency[prev_sym]; return F.cross_entropy(row_logits.view(-1,CONFIG["n_symbols"]), curr_sym.view(-1))

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.linear=ComplexLinear(dim); self.norm=ComplexLayerNorm(dim); self.act=ModReLU(dim)
        self.halt_linear=nn.Linear(dim*2,1); self.stack_ctrl=nn.Linear(dim*2,3); self.attention=ComplexAttention(dim)
        nn.init.constant_(self.halt_linear.bias,-2.0)
    def forward(self,z):
        z_proc=self.act(self.norm(self.linear(z))); z_proc=self.attention(z_proc)
        z_flat=torch.cat([z_proc.real,z_proc.imag],dim=-1)
        halt_prob=torch.sigmoid(self.halt_linear(z_flat)); stack_probs=F.softmax(self.stack_ctrl(z_flat),dim=-1)
        return z_proc, halt_prob, stack_probs

# ==========================================
# 6. Master Model
# ==========================================
class UberCRSN(nn.Module):
    def __init__(self,vocab_size,dim):
        super().__init__(); self.dim=dim; self.emb_mag=nn.Embedding(vocab_size,dim); self.emb_phase=nn.Parameter(torch.randn(vocab_size,dim))
        self.cell=AdaptiveRecursiveCell(dim); self.vq_layer=GraphMemoryVQ(dim,CONFIG["n_symbols"]); self.decoder=nn.Linear(dim*2,vocab_size)
        if CONFIG["use_stack"]: self.stack=DifferentiableStack(dim,CONFIG["stack_size"])
        self.ethics=EthicalConstraint(); self.register_buffer("prev_sym_soft",torch.zeros(CONFIG["n_symbols"]))
    def embed(self,idx): r=self.emb_mag(idx); t=self.emb_phase[idx]; return torch.complex(r*torch.cos(t), r*torch.sin(t))
    def forward(self,input_ids,hidden=None,prev_sym=None):
        batch_size=input_ids.size(0); z=self.embed(input_ids).squeeze(1)
        if hidden is None: z_prev=torch.zeros_like(z); stack_mem=torch.zeros(batch_size,CONFIG["stack_size"],self.dim*2,device=z.device); stack_ptr=torch.zeros(batch_size,CONFIG["stack_size"],device=z.device); stack_ptr[:,0]=1.0
        else: z_prev, stack_mem, stack_ptr=hidden; z=0.5*z+0.5*z_prev
        act_step=0; halting_probability=torch.zeros(batch_size,1).to(z.device); remain=torch.ones(batch_size,1).to(z.device); ponder_cost=0; stack_history=[]; z_weighted=torch.zeros_like(z); current_sym=prev_sym; vq_loss_total=0; ethical_loss_total=0
        for t in range(CONFIG["max_recursion_depth"]):
            act_step+=1; z_proc,p_halt,stack_ctrl=self.cell(z)
            if CONFIG["use_stack"]:
                stack_read,stack_mem,stack_ptr=self.stack(z_proc,stack_mem,stack_ptr,stack_ctrl); z_combined=z_proc+stack_read; stack_history.append(torch.sum(stack_ptr*torch.arange(CONFIG["stack_size"],device=z.device),dim=1))
            else: z_combined=z_proc; stack_history.append(torch.zeros(1).to(z.device))
            z_sym,vq_loss,sym_idx=self.vq_layer(z_combined,current_sym); eth_loss=self.ethics(current_sym,sym_idx,self.vq_layer.adjacency); ethical_loss_total+=eth_loss; current_sym=sym_idx
            z=0.7*z_combined+0.3*z_sym; still_running=(halting_probability<CONFIG["act_threshold"]).float(); p=p_halt*still_running; p=(remain if t==CONFIG["max_recursion_depth"]-1 else p); z_weighted=z_weighted+(p*z); halting_probability=halting_probability+p; remain=remain-p; ponder_cost+=still_running.mean(); vq_loss_total+=vq_loss
        features=torch.cat([z_weighted.real,z_weighted.imag],dim=-1); logits=self.decoder(features); next_hidden=(z_weighted,stack_mem,stack_ptr); avg_stack=(torch.stack(stack_history).mean() if len(stack_history)>0 else torch.tensor(0.0))
        return logits,next_hidden,current_sym,ponder_cost,vq_loss_total,ethical_loss_total,avg_stack

# ==========================================
# 7. Batch Anomaly Helper
# ==========================================
def batch_anomaly_ppx(model,input_texts):
    model.eval(); anomalies=[]
    for text in input_texts:
        tensor=torch.tensor([char_to_ix.get(c,0) for c in text],dtype=torch.long).to(DEVICE)
        hidden,prev_sym=None,None
        with torch.no_grad():
            for i in range(len(tensor)-1): x=tensor[i].view(1,1); _,hidden,prev_sym,_,_,eth_loss,_=model(x,hidden,prev_sym); anomalies.append(eth_loss.item())
    print(f"Batch anomaly scores: {anomalies[:min(10,len(anomalies))]}...")

# ==========================================
# 8. Dream Mode Batch
# ==========================================
def dream_mode_batch(model,start_chars="T",batch_size=5,max_steps=100):
    model.eval(); outputs=[c for c in start_chars[:batch_size]]
    x=torch.tensor([[char_to_ix[c]] for c in outputs],device=DEVICE)
    hidden,prev_sym=None,None
    for _ in range(max_steps):
        logits,hidden,prev_sym,_,_,_,_=model(x,hidden,prev_sym)
        probs=F.softmax(logits,dim=-1); next_ix=torch.multinomial(probs,1)
        chars=[ix_to_char[ix.item()] for ix in next_ix]; outputs=[o+c for o,c in zip(outputs,chars)]; x=next_ix
    print(f"Dream outputs: {outputs}")

# ==========================================
# 9. Live Dashboard Training
# ==========================================
def train_with_live_dashboard(batch_size=8, refresh_every=10):
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-5)

    data_len = len(data_tensor) - 1
    num_batches = (data_len + batch_size - 1) // batch_size

    # Initialize live dashboard
    fig, axs = plt.subplots(2,2, figsize=(12,8))
    axs[0,0].set_title("Loss"); axs[0,1].set_title("Avg PPX")
    axs[1,0].set_title("Avg Ponder Steps"); axs[1,1].set_title("Diversity/Entropy")
    plt.tight_layout()
    losses, ppx_vals, ponder_vals, diversity_vals = [], [], [], []

    for epoch in range(CONFIG["epochs"]):
        total_loss, total_ponder, avg_ppx, avg_diversity = 0,0,0,0
        entropy_weight = 0.01*(1-epoch/CONFIG["epochs"])

        for b in range(num_batches):
            start_idx, end_idx = b*batch_size, min((b+1)*batch_size, data_len)
            cur_batch_size = end_idx - start_idx
            x = data_tensor[start_idx:end_idx].view(cur_batch_size,1)
            y = data_tensor[start_idx+1:end_idx+1]

            logits, hidden, sym_idx, ponder, vq_loss, eth_loss, _ = model(x)
            h_z,h_mem,h_ptr = hidden
            hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach()); sym_idx = sym_idx.detach()

            loss_pred = F.cross_entropy(logits,y)
            loss_ponder = CONFIG["ponder_penalty"]*ponder
            probs = F.softmax(logits, dim=-1)
            loss_entropy = -entropy_weight*(-(probs*torch.log(probs+1e-8)).sum())

            # Correct prev_sym_soft update for batch
            curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float()           # [batch_size, n_symbols]
            curr_onehot_mean = curr_onehot.mean(dim=0)                                 # average -> [n_symbols]
            with torch.no_grad():
                model.prev_sym_soft.copy_(model.prev_sym_soft*0.9 + curr_onehot_mean*0.1)

            buffer_usage = model.prev_sym_soft
            loss_diversity = CONFIG["diversity_weight"]*(buffer_usage*torch.log(buffer_usage+1e-9)).sum()
            loss_ethics = CONFIG["ethical_weight"]*eth_loss

            loss = loss_pred + loss_ponder + 0.1*vq_loss + loss_entropy + loss_diversity + loss_ethics

            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"]); opt.step()

            total_loss += loss.item(); total_ponder += ponder.item()
            entropy_val = -(buffer_usage*torch.log(buffer_usage+1e-10)).sum()
            avg_ppx += torch.exp(entropy_val).item()
            avg_diversity += loss_diversity.item()

        scheduler.step()
        losses.append(total_loss/num_batches); ppx_vals.append(avg_ppx/num_batches)
        ponder_vals.append(total_ponder/num_batches); diversity_vals.append(avg_diversity/num_batches)

        # Update live dashboard every refresh_every epochs
        if epoch % refresh_every == 0:
            axs[0,0].cla(); axs[0,0].plot(losses, color='red'); axs[0,0].set_title("Loss")
            axs[0,1].cla(); axs[0,1].plot(ppx_vals, color='blue'); axs[0,1].set_title("Avg PPX")
            axs[1,0].cla(); axs[1,0].plot(ponder_vals, color='green'); axs[1,0].set_title("Avg Ponder Steps")
            axs[1,1].cla(); axs[1,1].plot(diversity_vals, color='purple'); axs[1,1].set_title("Diversity / Entropy")
            # plt.pause(0.01)

        # Optional: run batch anomaly check for first batch every 50 epochs
        if epoch % 50 == 0:
            input_texts = ["".join([ix_to_char[idx.item()] for idx in data_tensor[:batch_size]])]
            batch_anomaly_ppx(model, input_texts)

    plt.ioff()
    return model

# ==========================================
# 10. Main Execution
# ==========================================
if __name__ == "__main__":
    FILENAME = "crsn_tinyshakespeare_model.pth"
    
    # Training with live dashboard
    print("\n--- Starting Training with Live Dashboard ---")
    trained_model = train_with_live_dashboard(batch_size=8, refresh_every=5)
    
    # Save trained model
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
    }, FILENAME)
    print("Model saved successfully.")
    
    # Visualize full diagnostics
    print("\n--- Running Full Visualization Suite ---")
    visualize_all(trained_model)
    
    # Extract explicit logic rules from training data
    extract_logic_rules(trained_model, data_tensor)
    
    # Dream Mode batch generation
    print("\n--- 🌙 Dream Mode (Batch Generation) ---")
    dream_mode_batch(trained_model, start_chars="T", batch_size=5, max_steps=120)
    
    # Batch anomaly detection
    print("\n--- 🚨 Batch Anomaly Detection ---")
    input_texts = [
        "ROMEO: O, she doth teach the torches to burn bright!",
        "JULIET: O Romeo, Romeo! wherefore art thou Romeo?",
        "HAMLET: To be, or not to be, that is the question."
    ]
    batch_anomaly_ppx(trained_model, input_texts, max_len=100)
    
    # Optional: Attempt file download if in Colab
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


