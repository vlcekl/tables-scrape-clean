import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import random_split


class GraphDatasetPG(Dataset):
    """
    PyG Dataset that mirrors your DGL-based GraphDataset.
    """
    def __init__(self, root: str = './data', filename: str = 'dataset_graph.npz',
                 transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.filename = filename
        arr = np.load(os.path.join(self.root, self.filename), allow_pickle=True)['data'][0]

        # unpack
        self.n_node    = arr['n_node']
        self.n_edge    = arr['n_edge']
        self.node_attr = arr['node_attr']
        self.edge_attr = arr['edge_attr']
        self.src       = arr['src']
        self.dst       = arr['dst']
        self.shift     = arr['shift']
        self.mask      = arr['mask']
        self.smi       = arr['smi']

        # cumulative sums for slicing
        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])

    def len(self):
        return int(self.n_node.shape[0])

    def get(self, idx):
        # edge slice
        e0, e1 = int(self.e_csum[idx]), int(self.e_csum[idx+1])
        src = self.src[e0:e1]
        dst = self.dst[e0:e1]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # node slice
        n0, n1 = int(self.n_csum[idx]), int(self.n_csum[idx+1])
        x = torch.from_numpy(self.node_attr[n0:n1]).float()
        edge_attr = torch.from_numpy(self.edge_attr[e0:e1]).float()

        # targets & masks
        shift = torch.from_numpy(self.shift[n0:n1]).float()
        mask  = torch.from_numpy(self.mask[n0:n1]).bool()
        n_node = torch.tensor([n1 - n0], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=shift, mask=mask)
        data.num_nodes = int(n_node)

        return data, n_node, shift, mask


def split_dataset(dataset, splits, shuffle=True, random_state=None):
    """
    Splits a torch Dataset into subsets.
    - splits: sequence of fractions summing to 1.0, e.g. [0.8, 0.1, 0.1]
    Returns: list of Subset objects [train, val, test]
    """
    total_len = len(dataset)
    # compute integer lengths
    lengths = [int(frac * total_len) for frac in splits]
    # adjust last to ensure sum == total_len
    lengths[-1] = total_len - sum(lengths[:-1])

    if shuffle:
        generator = torch.Generator().manual_seed(random_state or 0)
        return random_split(dataset, lengths, generator=generator)
    else:
        return random_split(dataset, lengths)


def collate_reaction_graphs_pyg(batch):
    """
    Collate function for PyG DataLoader to batch graphs and auxiliary tensors.
    """
    data_list, n_nodes, shifts, masks = zip(*batch)
    batched = Batch.from_data_list(data_list)
    n_nodes = torch.cat(n_nodes, dim=0)
    shifts   = torch.cat(shifts,   dim=0)
    masks    = torch.cat(masks,    dim=0)
    return batched, n_nodes, shifts, masks


if __name__ == "__main__":
    # --- User parameters ---
    data_split   = [0.8, 0.1, 0.1]  # fractions for train/val/test
    batch_size   = 32
    random_seed  = 42

    # --- Instantiate dataset ---
    dataset = GraphDatasetPG(root='./data', filename='dataset_graph.npz')

    # --- Split into train/val/test ---
    train_set, val_set, test_set = split_dataset(
        dataset, data_split, shuffle=True, random_state=random_seed
    )

    # --- DataLoaders ---
    train_loader = PyGDataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_reaction_graphs_pyg, drop_last=True
    )
    val_loader = PyGDataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_reaction_graphs_pyg
    )
    test_loader = PyGDataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_reaction_graphs_pyg
    )

    # --- Quick sanity check ---
    print(f"Total graphs: {len(dataset)}")
    print(f"  Train / Val / Test sizes: {len(train_set)}, {len(val_set)}, {len(test_set)}")

    for batch_graph, n_nodes, shifts, masks in train_loader:
        print(batch_graph)
        print("n_nodes:", n_nodes.shape, "shifts:", shifts.shape, "masks:", masks.shape)
        break
