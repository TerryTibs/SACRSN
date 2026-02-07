# ============================================================
# SACRSN TinyShakespeare Complete Script with Anomaly Detection
# ============================================================

import os, time, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# ========================
# 0. Determinism & Device
# ========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {DEVICE}")

# ========================
# 1. Configuration
# ========================
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
    "warmup_epochs": 0,
    "anomaly_threshold": 5.0
}

# ========================
# 2. TinyShakespeare Data
# ========================
with open('tinyshakespeare.txt', 'r') as f:
    TEXT_DATA = f.read()
chars = sorted(list(set(TEXT_DATA)))
vocab_size = len(chars)
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}
data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(DEVICE)

# ========================
# 3. Complex Primitives
# ========================
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim): super().__init__(); self.scale=nn.Parameter(torch.ones(dim)); self.shift=nn.Parameter(torch.zeros(dim))
    def forward(self,z):
        mag=torch.abs(z)+CONFIG["eps"]
        mean=mag.mean(dim=-1,keepdim=True); var=mag.var(dim=-1,keepdim=True)
        norm_mag=(mag-mean)/torch.sqrt(var+CONFIG["eps"])
        norm_mag=norm_mag*self.scale+self.shift
        phase=torch.angle(z)
        return torch.complex(norm_mag*torch.cos(phase), norm_mag*torch.sin(phase))

class ModReLU(nn.Module):
    def __init__(self,dim): super().__init__(); self.bias=nn.Parameter(torch.zeros(dim))
    def forward(self,z): norm=torch.abs(z)+CONFIG["eps"]; scale=F.relu(norm+self.bias)/norm; return z*scale

class ComplexLinear(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.fc_real=nn.Linear(dim,dim,bias=False); self.fc_imag=nn.Linear(dim,dim,bias=False)
        nn.init.orthogonal_(self.fc_real.weight); nn.init.orthogonal_(self.fc_imag.weight)
    def forward(self,z):
        return torch.complex(self.fc_real(z.real)-self.fc_imag(z.imag), self.fc_real(z.imag)+self.fc_imag(z.real))

# ========================
# 4. Memory Modules
# ========================
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

# ========================
# 5. UberCRSN Core
# ========================
class UberCRSN(nn.Module):
    def __init__(self,vocab_size,embed_dim):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size, embed_dim)
        self.fc=nn.Linear(embed_dim,vocab_size)
        self.vq_layer=GraphMemoryVQ(embed_dim, CONFIG["n_symbols"])
        self.prev_sym_soft=torch.zeros(CONFIG["n_symbols"],device=DEVICE)
    def forward(self,x,hidden=None,prev_sym=None):
        z=self.embedding(x).float()
        logits=self.fc(z.squeeze(1))
        sym_idx=torch.argmax(logits,dim=-1)
        ponder=torch.tensor(0.0)
        vq_loss=torch.tensor(0.0); eth_loss=torch.tensor(0.0)
        return logits, hidden, sym_idx, ponder, vq_loss, eth_loss, None

# ========================
# 6. Batch Anomaly Detection
# ========================
def batch_anomaly_ppx(model,input_texts):
    model.eval(); anomalies=[]
    for text in input_texts:
        x=torch.tensor([char_to_ix[c] for c in text],device=DEVICE).view(-1,1)
        with torch.no_grad(): logits,_,sym_idx,_,_,_,_=model(x)
        probs=F.softmax(logits,dim=-1)
        entropy=-torch.sum(probs*torch.log(probs+1e-8),dim=-1).mean().item()
        ppx=np.exp(entropy)
        if ppx>CONFIG["anomaly_threshold"]: anomalies.append((text,ppx))
    if anomalies: print(f"Detected {len(anomalies)} anomalies exceeding PPX threshold:");
    for txt,val in anomalies: print(f"PPX {val:.2f}: {txt}")

# ========================
# 7. Batched Training with Live Dashboard
# ========================
def train_with_live_dashboard(batch_size=8,refresh_every=10):
    model=UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG["learning_rate"],weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=CONFIG["epochs"],eta_min=1e-5)
    data_len=len(data_tensor)-1; num_batches=(data_len+batch_size-1)//batch_size
    losses,ppx_vals,ponder_vals,diversity_vals=[],[],[],[]
    plt.ion(); fig,axs=plt.subplots(2,2,figsize=(12,8))

    for epoch in range(CONFIG["epochs"]):
        total_loss,total_ponder,avg_ppx,avg_diversity=0,0,0,0
        entropy_weight=0.01*(1-epoch/CONFIG["epochs"])
        for b in range(num_batches):
            start_idx,end_idx=b*batch_size,min((b+1)*batch_size,data_len)
            cur_batch_size=end_idx-start_idx
            x=data_tensor[start_idx:end_idx].view(cur_batch_size,1); y=data_tensor[start_idx+1:end_idx+1]
            logits,hidden,sym_idx,ponder,vq_loss,eth_loss,_=model(x)
            loss_pred=F.cross_entropy(logits,y); loss_ponder=CONFIG["ponder_penalty"]*ponder
            probs=F.softmax(logits,dim=-1); loss_entropy=-entropy_weight*(-(probs*torch.log(probs+1e-8)).sum())
            curr_onehot=F.one_hot(sym_idx,CONFIG["n_symbols"]).float(); curr_onehot_mean=curr_onehot.mean(dim=0)
            with torch.no_grad(): model.prev_sym_soft.copy_(model.prev_sym_soft*0.9+curr_onehot_mean*0.1)
            buffer_usage=model.prev_sym_soft
            loss_diversity=CONFIG["diversity_weight"]*(buffer_usage*torch.log(buffer_usage+1e-9)).sum()
            loss_ethics=CONFIG["ethical_weight"]*eth_loss
            loss=loss_pred+loss_ponder+0.1*vq_loss+loss_entropy+loss_diversity+loss_ethics
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"]); opt.step()
            total_loss+=loss.item(); total_ponder+=ponder.item(); entropy_val=-(buffer_usage*torch.log(buffer_usage+1e-10)).sum()
            avg_ppx+=torch.exp(entropy_val).item(); avg_diversity+=loss_diversity.item()
        scheduler.step(); losses.append(total_loss/num_batches); ppx_vals.append(avg_ppx/num_batches)
        ponder_vals.append(total_ponder/num_batches); diversity_vals.append(avg_diversity/num_batches)
        if epoch%refresh_every==0:
            axs[0,0].cla(); axs[0,0].plot(losses,color='red'); axs[0,0].set_title("Loss")
            axs[0,1].cla(); axs[0,1].plot(ppx_vals,color='blue'); axs[0,1].set_title("Avg PPX")
            axs[1,0].cla(); axs[1,0].plot(ponder_vals,color='green'); axs[1,0].set_title("Avg Ponder Steps")
            axs[1,1].cla(); axs[1,1].plot(diversity_vals,color='purple'); axs[1,1].set_title("Diversity / Entropy")
            plt.pause(0.01)
        # Run anomaly detection for first batch every 50 epochs
        if epoch%50==0:
            batch_anomaly_ppx(model,[TEXT_DATA[:batch_size]])
    plt.ioff()
    return model

# ========================
# 8. Visualization & Dream Mode
# ========================
def visualize_all(model):
    print("--- Generating Diagnostics & Visualizations ---")
    model.eval()
    symbol_to_char = defaultdict(list)
    hidden, prev_sym = None, None
    with torch.no_grad():
        for i in range(len(data_tensor)-1):
            x = data_tensor[i].view(1,1)
            _, hidden, prev_sym, _, _, _, _ = model(x, hidden, prev_sym)
            symbol_to_char[prev_sym.item()].append(ix_to_char[data_tensor[i].item()])

    # Graph visualization
    adj_probs = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    G = nx.DiGraph()
    for i in range(CONFIG["n_symbols"]): G.add_node(i)
    edges, weights = [], []
    for i in range(CONFIG["n_symbols"]):
        for j in range(CONFIG["n_symbols"]):
            w = adj_probs[i,j]
            if w>0.05: G.add_edge(i,j,weight=w); edges.append((i,j)); weights.append(w)

    plt.figure(figsize=(14,14))
    try: pos = nx.spring_layout(G,k=0.15,iterations=50,seed=42)
    except: pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G,pos,node_size=800,node_color=['#a0cbe2' if i in symbol_to_char else '#ffe5e5' for i in G.nodes()])
    nx.draw_networkx_labels(G,pos,font_size=8,font_weight='bold')
    for (u,v), w in zip(edges,weights):
        nx.draw_networkx_edges(G,pos,edgelist=[(u,v)],width=w*2,alpha=max(0.1,w),edge_color='gray',arrowstyle='->',arrowsize=10)
    plt.title("Semantic Topology")
    plt.show()


def dream_mode_batch(model,start_chars='T',batch_size=5,max_steps=120):
    print(f"--- Dream Mode: Generating {batch_size} sequences ---")
    model.eval()
    adj = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()

    sequences = [start_chars for _ in range(batch_size)]
    prev_syms = [None]*batch_size
    x = torch.tensor([[char_to_ix[start_chars]]] * batch_size, device=DEVICE)

    with torch.no_grad():
        for _ in range(max_steps):
            logits, hidden, sym_idx, _, _, _, _ = model(x)
            next_chars = []
            for i in range(batch_size):
                probs = F.softmax(logits[i],dim=-1).cpu().numpy()
                next_ix = np.random.choice(len(probs),p=probs)
                next_chars.append(ix_to_char[next_ix])
            sequences = [seq+c for seq,c in zip(sequences,next_chars)]
            x = torch.tensor([[char_to_ix[c] for c in next_chars]],device=DEVICE).T

    for i,seq in enumerate(sequences): print(f"Seq {i+1}: {seq}")

# ========================
# 9. Main Execution
# ========================
if __name__=='__main__':
    FILENAME='crsn_tinyshakespeare_model.pth'
    trained_model=train_with_live_dashboard(batch_size=8,refresh_every=5)
    torch.save({'model_state_dict':trained_model.state_dict(),'config':CONFIG},FILENAME)
    print(f"Saved model to {FILENAME}")
    visualize_all(trained_model)
    dream_mode_batch(
        trained_model,
        start_chars='T',
        batch_size=5,
        max_steps=50
    )
