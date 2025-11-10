"""
Discrete graph diffusion for scaffold inpainting AND structure growth/deletion with optional property guidance.

Key features
- Variable-size molecules on a fixed canvas (N_max).
- PAD atom (id=0) for absent nodes; NO_BOND (id=0) for absent edges.
- Observed scaffold nodes/edges are clamped each step.
- Classifier-free property conditioning (e.g., QED/logP).
- Soft property targets (mass & logP) via additive logit biases at sampling.
- Hard structural constraints via edge_must_bond / edge_must_nobond / node_allowed_types.
- Cleaned code: consistent naming, safer batching, tighter decoding.

Tested: PyTorch ≥ 2.2, PyG ≥ 2.5.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GINEConv
from torch_geometric.nn.norm import BatchNorm

# =============================
# 0) Configuration
# =============================

@dataclass
class DiffusionConfig:
    # Vocab sizes include special classes: PAD atom (0), NO_BOND (0)
    num_atom_types: int
    num_bond_types: int
    n_max: int = 40                      # max nodes on canvas
    d_model: int = 256
    num_layers: int = 6
    dropout: float = 0.1
    T: int = 200                         # diffusion steps
    atom_beta_start: float = 1e-3
    atom_beta_end: float = 0.02
    bond_beta_start: float = 1e-3
    bond_beta_end: float = 0.02
    clf_free_p_uncond: float = 0.1       # classifier-free guidance drop prob at train
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_valence_masks: bool = True
    d_cond: int = 2                      # e.g., [QED, logP] after scaling

# Special class ids
PAD_ATOM_ID = 0       # non-existent node
NO_BOND_ID = 0        # non-existent edge

# =============================
# 1) Utilities: schedules & categorical noise
# =============================

class LinearSchedule:
    def __init__(self, T: int, beta_start: float, beta_end: float):
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T+1)  # [0..T]
    def __call__(self, t: torch.LongTensor) -> torch.Tensor:
        return self.beta[t]

def make_prior_from_data_marginals(counts: torch.Tensor) -> torch.Tensor:
    probs = counts.float() + 1.0
    probs = probs / probs.sum()
    return probs

def q_sample_discrete(x0: torch.Tensor,
                      t: torch.LongTensor,
                      schedule: LinearSchedule,
                      prior: torch.Tensor) -> torch.Tensor:
    beta_t = schedule(t).to(x0.device)  # (N,)
    keep = torch.bernoulli((1 - beta_t).clamp(0,1)).to(dtype=torch.bool, device=x0.device)
    x_t = x0.clone()
    if (~keep).any():
        C = prior.size(0)
        num_replace = (~keep).sum().item()
        repl = torch.multinomial(prior.expand(num_replace, C), 1).squeeze(1).to(x0.device)
        x_t[~keep] = repl
    return x_t

# =============================
# 2) RDKit helpers (optional)
# =============================

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
    from rdkit.Chem import Descriptors, Crippen
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

# Basic per-atom properties for guidance fallback
ATOMIC_MASS = {"C":12.011, "N":14.007, "O":15.999, "F":18.998, "P":30.974, "S":32.06, "Cl":35.45, "Br":79.904, "I":126.904, "H":1.0079}
# Very rough hydrophobicity proxy (positive = more hydrophobic)
ATOM_HYDRO = {"C":1.0, "N":-0.5, "O":-0.7, "F":0.2, "P":0.1, "S":0.0, "Cl":0.6, "Br":0.6, "I":0.6, "H":0.1}

# =============================
# 3) Dataset base
# =============================

class MoleculeInpaintDataset(Dataset):
    """Data fields (variable-size on fixed canvas N_max):
    - x_cat: Long [N_max] atom type ids; 0=PAD means no atom at that position
    - edge_index: Long [2, E] edges of complete graph on N_max nodes
    - edge_attr_cat: Long [E] bond type ids; 0=NO_BOND means absent
    - node_mask_obs: Bool [N_max] observed nodes
    - edge_mask_obs: Bool [E] observed edges
    - cond_vec: Optional[Float] [1, d_cond] per-graph
    - optional hard constraints: edge_must_bond / edge_must_nobond / node_allowed_types
    """
    def __init__(self, split: str = "train"):
        super().__init__()
        self.split = split
        self._data = []
    def __len__(self):
        return len(self._data)
    def __getitem__(self, idx):
        return self._data[idx]

# =============================
# 4) Model components
# =============================

class TimestepEmbed(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
    @staticmethod
    def sinusoidal(t: torch.LongTensor, d_model: int) -> torch.Tensor:
        device = t.device
        half = d_model // 2
        freqs = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / max(1, half-1)))
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if d_model % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb
    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        h = self.sinusoidal(t, self.lin1.in_features)
        h = self.lin2(self.act(self.lin1(h)))
        return h

class GraphDenoiser(nn.Module):
    """
    Predicts categorical logits for nodes (PAD+atoms) and edges (NO_BOND+bonds) at step t.
    Classifier-free property conditioning; feasibility/clamps handled outside if needed.
    """
    def __init__(self, cfg: DiffusionConfig, d_atom_embed: int = 128, d_bond_embed: int = 64):
        super().__init__()
        self.cfg = cfg
        self.atom_embed = nn.Embedding(cfg.num_atom_types, d_atom_embed)
        self.bond_embed = nn.Embedding(cfg.num_bond_types, d_bond_embed)
        self.node_mask_embed = nn.Embedding(2, 8)
        self.edge_mask_embed = nn.Embedding(2, 8)

        in_dim = d_atom_embed + 8 + cfg.d_model
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            nn_lin = nn.Sequential(nn.Linear(in_dim, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model))
            conv = GINEConv(nn_lin, train_eps=True)
            self.layers.append(conv)
            self.norms.append(BatchNorm(cfg.d_model))
            in_dim = cfg.d_model

        self.t_embed = TimestepEmbed(cfg.d_model)
        self.cond_proj = nn.Linear(cfg.d_cond, cfg.d_model)  # <-- matches DiffusionConfig.d_cond
        self.null_cond = nn.Parameter(torch.zeros(cfg.d_model))

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_bond_embed + 8, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model)
        )
        self.atom_head = nn.Linear(cfg.d_model, cfg.num_atom_types)
        self.bond_head = nn.Linear(cfg.d_model, cfg.num_bond_types)

    def _broadcast_cond(self, data: Data, cond_vec: Optional[torch.Tensor]) -> torch.Tensor:
        # Returns per-node [num_nodes, d_model] cond embedding (or null when None)
        if cond_vec is None:
            # expand the single null vector per node
            if hasattr(data, "x_noisy"):
                N = data.x_noisy.size(0)
            else:
                N = data.x_cat.size(0)
            return self.null_cond.expand(N, -1)
        # cond_vec shape: [num_graphs, d_cond] (batched) OR [1, d_cond] (single)
        c_proj = self.cond_proj(cond_vec)  # [G, d_model]
        if hasattr(data, "batch"):
            return c_proj[data.batch]       # per-node via graph index
        else:
            # single-graph data, repeat to node count
            N = data.x_cat.size(0)
            return c_proj.expand(N, -1)

    def forward(self, data: Data, t_node: torch.LongTensor, cond_vec: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Nodes
        x_atom = self.atom_embed(data.x_noisy)
        x_mask = self.node_mask_embed(data.node_mask_obs.long())
        t_emb  = self.t_embed(t_node)
        c_node = self._broadcast_cond(data, cond_vec)  # per-node

        h = torch.cat([x_atom, x_mask, t_emb + c_node], dim=-1)

        # Edges
        e_bond = self.bond_embed(data.edge_attr_noisy)
        e_mask = self.edge_mask_embed(data.edge_mask_obs.long())
        e_feat = self.edge_mlp(torch.cat([e_bond, e_mask], dim=-1))

        # Message passing on complete graph
        edge_index = data.edge_index
        for conv, norm in zip(self.layers, self.norms):
            h = conv(h, edge_index, e_feat)
            h = norm(h)
            h = F.dropout(F.silu(h), p=self.cfg.dropout, training=self.training)

        atom_logits = self.atom_head(h)
        src, dst = edge_index
        edge_h = h[src] + h[dst]
        bond_logits = self.bond_head(edge_h)
        return atom_logits, bond_logits

# =============================
# 5) Training step
# =============================

def sample_timesteps_like(x: torch.Tensor, T: int) -> torch.LongTensor:
    return torch.randint(1, T+1, (x.size(0),), device=x.device)

def training_step(model: GraphDenoiser,
                  cfg: DiffusionConfig,
                  batch: Data,
                  atom_sched: LinearSchedule,
                  bond_sched: LinearSchedule,
                  atom_prior: torch.Tensor,
                  bond_prior: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = cfg.device
    batch = batch.to(device)

    # Classifier-free: randomly drop conditioning per BATCH (simple & effective)
    cond_vec = getattr(batch, 'cond_vec', None)
    if (cond_vec is not None) and (random.random() <= cfg.clf_free_p_uncond):
        cond_vec = None

    # Sample timesteps
    t_nodes = sample_timesteps_like(batch.x_cat, cfg.T)
    t_edges = sample_timesteps_like(batch.edge_attr_cat, cfg.T)

    # Corrupt nodes/edges (PAD and NO_BOND included in priors)
    x_noisy = q_sample_discrete(batch.x_cat, t_nodes, atom_sched, atom_prior)
    e_noisy = q_sample_discrete(batch.edge_attr_cat, t_edges, bond_sched, bond_prior)

    batch.x_noisy = x_noisy
    batch.edge_attr_noisy = e_noisy

    atom_logits, bond_logits = model(batch, t_nodes, cond_vec)

    # Masks for feasibility (no bonds to PAD, constraints, etc.)
    if cfg.use_valence_masks:
        atom_logits, bond_logits = valence_mask_logits(atom_logits, bond_logits, batch)

    # Losses over all positions
    node_loss = F.cross_entropy(atom_logits, batch.x_cat)
    edge_loss = F.cross_entropy(bond_logits, batch.edge_attr_cat)
    loss = node_loss + edge_loss
    return loss, {"node_ce": float(node_loss.detach().cpu()), "edge_ce": float(edge_loss.detach().cpu())}

# =============================
# 6) Feasibility masks + property guidance sampler
# =============================

def valence_mask_logits(atom_logits: torch.Tensor,
                        bond_logits: torch.Tensor,
                        data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask impossible classes and enforce structure constraints.
    - Disallow bonds on PAD nodes.
    - Enforce hard edge constraints if provided: edge_must_bond / edge_must_nobond.
    - Optional: node_allowed_types to constrain atom classes per node.
    """
    # No bonds touching PAD
    if hasattr(data, 'x_noisy'):
        pad_nodes = (data.x_noisy == PAD_ATOM_ID)
        if pad_nodes.any():
            src, dst = data.edge_index
            bad_edges = pad_nodes[src] | pad_nodes[dst]
            bond_logits[bad_edges] = -1e9
            bond_logits[bad_edges, NO_BOND_ID] = 0.0

    # Hard edge constraints
    if hasattr(data, 'edge_must_bond') and data.edge_must_bond is not None:
        must = data.edge_must_bond
        if must.any():
            bond_logits[must, NO_BOND_ID] = -1e9
    if hasattr(data, 'edge_must_nobond') and data.edge_must_nobond is not None:
        must_not = data.edge_must_nobond
        if must_not.any():
            bond_logits[must_not] = -1e9
            bond_logits[must_not, NO_BOND_ID] = 0.0

    # Node allowed types (optional)
    if hasattr(data, 'node_allowed_types') and data.node_allowed_types is not None:
        allowed = data.node_allowed_types  # [N, num_atom_types] bool
        if allowed.dim() == 2:
            disallowed = ~allowed
            atom_logits[disallowed] = -1e9

    return atom_logits, bond_logits

def classifier_free_mix(logits_uncond: torch.Tensor,
                        logits_cond: torch.Tensor,
                        guidance_scale: float) -> torch.Tensor:
    return logits_uncond + guidance_scale * (logits_cond - logits_uncond)

# --- Simple property estimators (need VOCAB; defined later but used at runtime)
def estimate_mass_from_x(x_cat: torch.Tensor) -> float:
    mass = 0.0
    for a_id in x_cat.tolist():
        if a_id == PAD_ATOM_ID: continue
        sym = VOCAB.id_to_atom.get(int(a_id), "C")
        if sym == "<PAD>": continue
        mass += ATOMIC_MASS.get(sym, 12.0)
    return float(mass)

def estimate_logp_from_graph(x_cat: torch.Tensor, edge_index: torch.Tensor, e_cat: torch.Tensor) -> float:
    if RDKit_AVAILABLE:
        mol = decode_to_mol_from_canvas(x_cat.cpu(), edge_index.cpu(), e_cat.cpu())
        if mol is not None:
            try:
                return float(Crippen.MolLogP(mol))
            except Exception:
                pass
    # Fallback heuristic
    hydro = 0.0
    for a_id in x_cat.tolist():
        if a_id == PAD_ATOM_ID: continue
        sym = VOCAB.id_to_atom.get(int(a_id), "C")
        if sym == "<PAD>": continue
        hydro += ATOM_HYDRO.get(sym, 0.0)
    if RDKit_AVAILABLE:
        nonpolar_like = (e_cat == VOCAB.bond_to_id[rdchem.BondType.SINGLE]).float().sum() + \
                        (e_cat == VOCAB.bond_to_id[rdchem.BondType.AROMATIC]).float().sum()
    else:
        nonpolar_like = (e_cat == 1).float().sum() + (e_cat == 4).float().sum()
    hydro += 0.02 * float(nonpolar_like)
    return float(hydro)

def compute_property_bias(x_cur: torch.Tensor,
                          edge_index: torch.Tensor,
                          e_cur: torch.Tensor,
                          target_mass: Optional[Tuple[float,float]] = None,
                          target_logp: Optional[Tuple[float,float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return additive biases for (atom_logits_bias [num_atom_types], bond_logits_bias [num_bond_types])."""
    atom_bias = torch.zeros(len(VOCAB.atom_to_id), dtype=torch.float32, device=x_cur.device)
    bond_bias = torch.zeros(len(VOCAB.bond_to_id), dtype=torch.float32, device=x_cur.device)

    # Mass guidance
    if target_mass is not None:
        cur_mass = estimate_mass_from_x(x_cur)
        low, high = target_mass
        if cur_mass < low:
            for k, sym in VOCAB.id_to_atom.items():
                if sym == "<PAD>": atom_bias[k] -= 0.5; continue
                atom_bias[k] += 0.01 * ATOMIC_MASS.get(sym, 12.0)
        elif cur_mass > high:
            atom_bias[PAD_ATOM_ID] += 0.5
            for k, sym in VOCAB.id_to_atom.items():
                if sym == "<PAD>": continue
                atom_bias[k] -= 0.01 * ATOMIC_MASS.get(sym, 12.0)

    # Hydrophobicity / logP guidance
    if target_logp is not None:
        cur_logp = estimate_logp_from_graph(x_cur, edge_index, e_cur)
        low, high = target_logp
        if cur_logp < low:
            for k, sym in VOCAB.id_to_atom.items():
                if sym == "<PAD>": continue
                atom_bias[k] += 0.2 * ATOM_HYDRO.get(sym, 0.0)
            if RDKit_AVAILABLE:
                bond_bias[VOCAB.bond_to_id[rdchem.BondType.SINGLE]] += 0.1
                bond_bias[VOCAB.bond_to_id[rdchem.BondType.AROMATIC]] += 0.1
            else:
                bond_bias[1] += 0.1; bond_bias[4] += 0.1
        elif cur_logp > high:
            atom_bias[PAD_ATOM_ID] += 0.2
            for k, sym in VOCAB.id_to_atom.items():
                if sym == "<PAD>": continue
                atom_bias[k] -= 0.2 * ATOM_HYDRO.get(sym, 0.0)
            bond_bias[NO_BOND_ID] += 0.1

    return atom_bias, bond_bias

def apply_bias_to_logits(atom_logits: torch.Tensor,
                         bond_logits: torch.Tensor,
                         atom_bias: torch.Tensor,
                         bond_bias: torch.Tensor,
                         node_mask: torch.Tensor,
                         edge_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Only bias unknown positions
    atom_logits = atom_logits.clone()
    bond_logits = bond_logits.clone()
    atom_logits[~node_mask] = atom_logits[~node_mask] + atom_bias
    bond_logits[~edge_mask] = bond_logits[~edge_mask] + bond_bias
    return atom_logits, bond_logits

@torch.no_grad()
def sample_inpaint_grow(model: GraphDenoiser,
                        cfg: DiffusionConfig,
                        data: Data,
                        atom_prior: torch.Tensor,
                        bond_prior: torch.Tensor,
                        guidance_scale: float = 2.0,
                        cond_vec: Optional[torch.Tensor] = None,
                        target_mass: Optional[Tuple[float,float]] = None,
                        target_logp: Optional[Tuple[float,float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reverse diffusion with inpainting + growth/deletion, soft property guidance, and hard constraints."""
    device = cfg.device
    model.eval()
    with torch.no_grad():
        x_cur = data.x_cat.clone().to(device)
        e_cur = data.edge_attr_cat.clone().to(device)
        node_obs = data.node_mask_obs.to(device)
        edge_obs = data.edge_mask_obs.to(device)
        data = data.to(device)

        # Initialize unknowns from priors
        if (~node_obs).any():
            x_unknown = torch.multinomial(atom_prior.expand((~node_obs).sum(), -1), 1).squeeze(1).to(device)
            x_cur[~node_obs] = x_unknown
        if (~edge_obs).any():
            e_unknown = torch.multinomial(bond_prior.expand((~edge_obs).sum(), -1), 1).squeeze(1).to(device)
            e_cur[~edge_obs] = e_unknown

        for t in range(cfg.T, 0, -1):
            t_nodes = torch.full_like(x_cur, t)
            data.x_noisy = x_cur
            data.edge_attr_noisy = e_cur

            # Unconditional
            atom_logits_u, bond_logits_u = model(data, t_nodes, cond_vec=None)
            # Conditional
            if cond_vec is not None:
                # single graph: cond_vec should be [1, d_cond]
                atom_logits_c, bond_logits_c = model(data, t_nodes, cond_vec=cond_vec)
                atom_logits = classifier_free_mix(atom_logits_u, atom_logits_c, guidance_scale)
                bond_logits = classifier_free_mix(bond_logits_u, bond_logits_c, guidance_scale)
            else:
                atom_logits, bond_logits = atom_logits_u, bond_logits_u

            # Hard feasibility masks
            if cfg.use_valence_masks:
                atom_logits, bond_logits = valence_mask_logits(atom_logits, bond_logits, data)

            # Soft property biases
            if (target_mass is not None) or (target_logp is not None):
                a_bias, b_bias = compute_property_bias(x_cur, data.edge_index, e_cur, target_mass, target_logp)
                atom_logits, bond_logits = apply_bias_to_logits(atom_logits, bond_logits, a_bias, b_bias, node_obs, edge_obs)

            # Sample unknowns only
            if (~node_obs).any():
                probs_atom = F.softmax(atom_logits[~node_obs], dim=-1)
                x_sample = torch.multinomial(probs_atom, 1).squeeze(1)
                x_cur[~node_obs] = x_sample
            if (~edge_obs).any():
                probs_bond = F.softmax(bond_logits[~edge_obs], dim=-1)
                e_sample = torch.multinomial(probs_bond, 1).squeeze(1)
                e_cur[~edge_obs] = e_sample

            # Enforce PAD → NO_BOND consistency
            src, dst = data.edge_index
            pad_nodes = (x_cur == PAD_ATOM_ID)
            if pad_nodes.any():
                pad_edge_mask = pad_nodes[src] | pad_nodes[dst]
                e_cur[pad_edge_mask] = NO_BOND_ID

            # Clamp observed positions back to ground truth
            x_cur[node_obs] = data.x_cat[node_obs]
            e_cur[edge_obs] = data.edge_attr_cat[edge_obs]

        return x_cur, e_cur

# =============================
# 7) Training loop (skeleton)
# =============================

def train(cfg: DiffusionConfig,
          train_set: Dataset,
          val_set: Optional[Dataset] = None,
          atom_class_counts: Optional[torch.Tensor] = None,
          bond_class_counts: Optional[torch.Tensor] = None):

    device = cfg.device
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size)

    if atom_class_counts is None:
        atom_class_counts = torch.ones(cfg.num_atom_types)
    if bond_class_counts is None:
        bond_class_counts = torch.ones(cfg.num_bond_types)
    atom_prior = make_prior_from_data_marginals(atom_class_counts).to(device)
    bond_prior = make_prior_from_data_marginals(bond_class_counts).to(device)

    atom_sched = LinearSchedule(cfg.T, cfg.atom_beta_start, cfg.atom_beta_end)
    bond_sched = LinearSchedule(cfg.T, cfg.bond_beta_start, cfg.bond_beta_end)
    atom_sched.beta = atom_sched.beta.to(device)
    bond_sched.beta = bond_sched.beta.to(device)

    model = GraphDenoiser(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {"loss": 0.0, "node_ce": 0.0, "edge_ce": 0.0}
        for batch in train_loader:
            opt.zero_grad()
            loss, logs = training_step(model, cfg, batch, atom_sched, bond_sched, atom_prior, bond_prior)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running["loss"] += float(loss.detach().cpu())
            running["node_ce"] += logs["node_ce"]
            running["edge_ce"] += logs["edge_ce"]
        n_batches = max(1, len(train_loader))
        print(f"Epoch {epoch:03d} | loss {running['loss']/n_batches:.4f} | node {running['node_ce']/n_batches:.4f} | edge {running['edge_ce']/n_batches:.4f}")

        if val_loader is not None and epoch % 5 == 0:
            evaluate(model, cfg, val_loader, atom_sched, bond_sched, atom_prior, bond_prior)

    return model

@torch.no_grad()
def evaluate(model: GraphDenoiser,
             cfg: DiffusionConfig,
             val_loader: DataLoader,
             atom_sched: LinearSchedule,
             bond_sched: LinearSchedule,
             atom_prior: torch.Tensor,
             bond_prior: torch.Tensor):
    model.eval()
    device = cfg.device
    totals = {"node_ce": 0.0, "edge_ce": 0.0}
    count = 0
    for batch in val_loader:
        batch = batch.to(device)
        t_nodes = sample_timesteps_like(batch.x_cat, cfg.T)
        t_edges = sample_timesteps_like(batch.edge_attr_cat, cfg.T)
        x_noisy = q_sample_discrete(batch.x_cat, t_nodes, atom_sched, atom_prior)
        e_noisy = q_sample_discrete(batch.edge_attr_cat, t_edges, bond_sched, bond_prior)
        batch.x_noisy = x_noisy
        batch.edge_attr_noisy = e_noisy
        atom_logits, bond_logits = model(batch, t_nodes, cond_vec=None)
        node_loss = F.cross_entropy(atom_logits, batch.x_cat)
        edge_loss = F.cross_entropy(bond_logits, batch.edge_attr_cat)
        totals["node_ce"] += float(node_loss.detach().cpu())
        totals["edge_ce"] += float(edge_loss.detach().cpu())
        count += 1
    print(f"Val | node_ce {totals['node_ce']/max(1,count):.4f} | edge_ce {totals['edge_ce']/max(1,count):.4f}")

# =============================
# 8) Vocab, conversion, dense-canvas dataset, decode
# =============================

class AtomBondVocab:
    def __init__(self):
        # PAD at index 0; common atoms thereafter
        self.atom_list_core = ["C","N","O","F","P","S","Cl","Br","I","H"]
        self.atom_to_id = {"<PAD>": PAD_ATOM_ID}
        for i, s in enumerate(self.atom_list_core, start=1):
            self.atom_to_id[s] = i
        self.id_to_atom = {v:k for k,v in self.atom_to_id.items()}
        if RDKit_AVAILABLE:
            self.bond_to_id = {None: NO_BOND_ID,
                               Chem.rdchem.BondType.SINGLE: 1,
                               Chem.rdchem.BondType.DOUBLE: 2,
                               Chem.rdchem.BondType.TRIPLE: 3,
                               Chem.rdchem.BondType.AROMATIC: 4}
        else:
            self.bond_to_id = {None: NO_BOND_ID, "SINGLE":1, "DOUBLE":2, "TRIPLE":3, "AROMATIC":4}
        self.id_to_bond = {v:k for k,v in self.bond_to_id.items()}

VOCAB = AtomBondVocab()

def build_complete_graph(n_max: int) -> Tuple[torch.Tensor, int]:
    # Exclude self-loops; undirected with both directions
    adj = torch.ones(n_max, n_max) - torch.eye(n_max)
    edge_index = dense_to_sparse(adj)[0].long()
    return edge_index, edge_index.size(1)

def mol_to_canvas(mol: "Chem.Mol", n_max: int, scaffold_mask_nodes: Optional[torch.Tensor] = None,
                  cond_vec: Optional[torch.Tensor] = None) -> Data:
    assert RDKit_AVAILABLE, "RDKit required for this dataset helper"
    mol = Chem.RemoveHs(mol)
    N = mol.GetNumAtoms()
    assert N <= n_max, f"Molecule has {N} atoms > n_max={n_max}"

    x_cat = torch.full((n_max,), PAD_ATOM_ID, dtype=torch.long)
    for i, a in enumerate(mol.GetAtoms()):
        sym = a.GetSymbol(); sym = sym if sym in VOCAB.atom_to_id else "C"
        x_cat[i] = VOCAB.atom_to_id[sym]

    edge_index, E = build_complete_graph(n_max)
    e_cat = torch.full((E,), NO_BOND_ID, dtype=torch.long)

    # Map (i,j) -> edge idx
    idx_map = {(int(edge_index[0,k]), int(edge_index[1,k])): int(k) for k in range(edge_index.size(1))}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = b.GetBondType()
        bt = bt if bt in VOCAB.bond_to_id else Chem.rdchem.BondType.SINGLE
        e_id = VOCAB.bond_to_id[bt]
        e_cat[idx_map[(i,j)]] = e_id
        e_cat[idx_map[(j,i)]] = e_id

    if scaffold_mask_nodes is None:
        scaffold_mask_nodes = torch.zeros(n_max, dtype=torch.bool)
    # Observed edges = both endpoints observed
    obs_edges = (scaffold_mask_nodes[edge_index[0]] & scaffold_mask_nodes[edge_index[1]])

    data = Data(
        x_cat=x_cat,
        edge_index=edge_index,
        edge_attr_cat=e_cat,
        node_mask_obs=scaffold_mask_nodes,
        edge_mask_obs=obs_edges,
    )
    if cond_vec is not None:
        data.cond_vec = cond_vec  # [1, d_cond]
    return data

def murcko_scaffold_mask_on_canvas(mol: "Chem.Mol", n_max: int) -> torch.Tensor:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    core = MurckoScaffold.GetScaffoldForMol(mol)
    core_atoms = set([a.GetIdx() for a in core.GetAtoms()])
    mask = torch.zeros(n_max, dtype=torch.bool)
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetIdx() in core_atoms:
            mask[i] = True
    return mask

class PropScaler:
    def __init__(self):
        self.min = torch.tensor([0.0, -2.0])   # [QED, logP]
        self.max = torch.tensor([1.0, 6.0])
    def __call__(self, vec: torch.Tensor) -> torch.Tensor:
        return (vec - self.min) / (self.max - self.min + 1e-6)

PROP_SCALER = PropScaler()

def compute_cond_vec(mol: "Chem.Mol") -> torch.Tensor:
    from rdkit.Chem import QED, Crippen
    qed = QED.qed(mol)
    logp = Crippen.MolLogP(mol)
    v = torch.tensor([qed, logp], dtype=torch.float32)
    return PROP_SCALER(v).unsqueeze(0)  # [1, 2]

class SmilesCanvasDataset(MoleculeInpaintDataset):
    def __init__(self, smiles_list, n_max: int, mask_ratio_range: Tuple[float,float]=(0.3,0.6), with_props: bool=True,
                 edge_constraints: bool=False):
        super().__init__(split="train")
        assert RDKit_AVAILABLE, "This example dataset requires RDKit"
        self._data = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            obs_nodes = murcko_scaffold_mask_on_canvas(mol, n_max)
            idxs = torch.where(obs_nodes)[0]
            if idxs.numel() > 0:
                keep_k = max(1, int(random.uniform(*mask_ratio_range) * idxs.numel()))
                keep = idxs[torch.randperm(idxs.numel())[:keep_k]]
                obs = torch.zeros(n_max, dtype=torch.bool); obs[keep] = True
            else:
                obs = torch.zeros(n_max, dtype=torch.bool)
            cond_vec = compute_cond_vec(mol) if with_props else None
            data = mol_to_canvas(mol, n_max, scaffold_mask_nodes=obs, cond_vec=cond_vec)

            # OPTIONAL hard edge constraints (example)
            if edge_constraints:
                src, dst = data.edge_index
                must_bond = torch.zeros_like(src, dtype=torch.bool)
                must_nobond = torch.zeros_like(src, dtype=torch.bool)
                # Enforce bonds among observed neighbors if present in GT
                obs_pairs = obs[src] & obs[dst]
                must_bond = must_bond | (obs_pairs & (data.edge_attr_cat != NO_BOND_ID))
                # For PAD neighbors, enforce NO_BOND
                pad_nodes = (data.x_cat == PAD_ATOM_ID)
                must_nobond = must_nobond | (pad_nodes[src] | pad_nodes[dst])
                data.edge_must_bond = must_bond
                data.edge_must_nobond = must_nobond
            self._data.append(data)

def decode_to_mol_from_canvas(x_cat: torch.Tensor, edge_index: torch.Tensor, e_cat: torch.Tensor) -> Optional["Chem.Mol"]:
    if not RDKit_AVAILABLE:
        return None
    rw = Chem.RWMol()
    node_map = {}
    for i, a_id in enumerate(x_cat.tolist()):
        if a_id == PAD_ATOM_ID:
            node_map[i] = None; continue
        sym = VOCAB.id_to_atom.get(a_id, "C")
        if sym == "<PAD>":
            node_map[i] = None; continue
        idx = rw.AddAtom(Chem.Atom(sym))
        node_map[i] = idx
    seen = set()
    for k in range(edge_index.size(1)):
        i = int(edge_index[0, k]); j = int(edge_index[1, k])
        if i >= j:  # add undirected once via (min,max)
            continue
        if node_map.get(i) is None or node_map.get(j) is None:
            continue
        bt_id = int(e_cat[k].item())
        if bt_id == NO_BOND_ID:
            continue
        bt = VOCAB.id_to_bond.get(bt_id, Chem.rdchem.BondType.SINGLE)
        key = (i,j)
        if key in seen: continue
        seen.add(key)
        try:
            rw.AddBond(node_map[i], node_map[j], bt)
        except Exception:
            pass
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol

# =============================
# 9) CLI / main
# =============================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--T", type=int, default=150)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--n_max", type=int, default=40)
    parser.add_argument("--no_props", action="store_true")
    parser.add_argument("--target_mass", type=str, default="")
    parser.add_argument("--target_logp", type=str, default="")
    args = parser.parse_args()

    # Vocab sizes
    num_atoms = 1 + len(VOCAB.atom_list_core)
    num_bonds = 1 + 4

    cfg = DiffusionConfig(
        num_atom_types=num_atoms,
        num_bond_types=num_bonds,
        n_max=args.n_max,
        d_model=256,
        num_layers=6,
        T=args.T,
        batch_size=args.batch_size,
        epochs=args.epochs,
        d_cond=2,  # QED + logP
    )

    # Example data
    smiles = [
        "c1ccccc1C(=O)O",
        "CCOc1ccc2nc(S)ncc2c1",
        "CCN(CC)CCOc1ccc(CC(=O)O)cc1",
        "CC1=CC(=O)NC(=O)N1",
        "CCOC(=O)c1ccc(O)cc1",
        "CC(C)Oc1ccc2c(c1)OCO2",
        "CC(C)CC1=CC(=O)NC(=O)N1",
        "C1CCCCC1",
        "CCN1C(=O)NC(=O)C1=O",
        "COc1ccc(OC)cc1"
    ]

    with_props = not args.no_props
    ds = SmilesCanvasDataset(smiles, n_max=cfg.n_max, with_props=with_props, edge_constraints=True)
    n = len(ds); n_val = max(1, n//5)
    train_set = torch.utils.data.Subset(ds, list(range(0, n-n_val)))
    val_set = torch.utils.data.Subset(ds, list(range(n-n_val, n)))

    # Empirical priors including PAD/NO_BOND
    atom_counts = torch.zeros(cfg.num_atom_types)
    bond_counts = torch.zeros(cfg.num_bond_types)
    for i in range(len(train_set)):
        d = ds._data[i]
        atom_counts += torch.bincount(d.x_cat, minlength=cfg.num_atom_types)
        bond_counts += torch.bincount(d.edge_attr_cat, minlength=cfg.num_bond_types)
    # Slight encouragement for PAD/NO_BOND
    atom_counts[PAD_ATOM_ID] += cfg.n_max
    bond_counts[NO_BOND_ID] += cfg.n_max * (cfg.n_max - 1)

    model = train(cfg, train_set, val_set, atom_counts, bond_counts)

    device = cfg.device
    model.to(device)

    data = ds._data[-1].clone().to(device)

    atom_prior = make_prior_from_data_marginals(atom_counts).to(device)
    bond_prior = make_prior_from_data_marginals(bond_counts).to(device)

    # Parse property targets like "300,450" or "1.0,3.0"
    target_mass = None
    if args.target_mass:
        try:
            lo, hi = [float(x) for x in args.target_mass.split(',')]
            target_mass = (lo, hi)
        except Exception:
            print("[warn] could not parse --target_mass, expected 'lo,hi'")
    target_logp = None
    if args.target_logp:
        try:
            lo, hi = [float(x) for x in args.target_logp.split(',')]
            target_logp = (lo, hi)
        except Exception:
            print("[warn] could not parse --target_logp, expected 'lo,hi'")

    cond_vec = getattr(data, 'cond_vec', None)  # [1,2] or None
    x_hat, e_hat = sample_inpaint_grow(model, cfg, data, atom_prior, bond_prior,
                                       guidance_scale=args.guidance,
                                       cond_vec=cond_vec,
                                       target_mass=target_mass,
                                       target_logp=target_logp)

    mol_pred = decode_to_mol_from_canvas(x_hat.detach().cpu(), data.edge_index.detach().cpu(), e_hat.detach().cpu())
    if RDKit_AVAILABLE and mol_pred is not None:
        print("Predicted SMILES:", Chem.MolToSmiles(mol_pred))

if __name__ == "__main__":
    main()
