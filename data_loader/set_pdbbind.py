import os
import dgl
import torch
import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from sklearn.cluster import AffinityPropagation
from rdkit.DataStructs import FingerprintSimilarity
from torch.utils.data import Dataset, DataLoader

from .pdbbind import PDBBind

class SetPDBBind(object):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.gen_datasets()

    def gen_datasets(self):
        np.random.seed(1)   # fix dataset
        V_size, S_size = self.params.v_size, self.params.s_size
        self.dataset = load_pdbbind(self.params)

        data_root = '/root/dataset/pdbbind'
        data_path = os.path.join(data_root, 'pdbbind_set_data.pkl')
        if os.path.exists(data_path):
            print(f'load data from {data_path}')
            trainData, valData, testData = pickle.load(open(data_path, "rb"))
            self.V_train, self.S_train = trainData['V_train'], trainData['S_train']
            self.V_val, self.S_val = valData['V_train'], valData['S_train']
            self.V_test, self.S_test = testData['V_train'], testData['S_train']
        else:
            self.V_train, self.S_train = get_set_pdbbind_dataset_activate(self.dataset, V_size, S_size, size=1000)
            self.V_val, self.S_val = get_set_pdbbind_dataset_activate(self.dataset, V_size, S_size, size=100)
            self.V_test, self.S_test = get_set_pdbbind_dataset_activate(self.dataset, V_size, S_size, size=100)

            trainData = {'V_train': self.V_train, 'S_train': self.S_train}
            valData = {'V_train': self.V_val, 'S_train': self.S_val}
            testData = {'V_train': self.V_test, 'S_train': self.S_test}
            if not os.path.exists(data_root):
                os.makedirs(data_root)
            pickle.dump((trainData, valData, testData), open(data_path, "wb"))

    def get_loaders(self, batch_size, num_workers, shuffle_train=False, get_test=True):
        train_dataset = SetDataset(self.dataset, self.V_train, self.S_train, self.params, is_train=True)
        val_dataset = SetDataset(self.dataset, self.V_val, self.S_val, self.params)
        test_dataset = SetDataset(self.dataset, self.V_test, self.S_test, self.params)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                    collate_fn=collate_train, pin_memory=True, shuffle=shuffle_train, num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                    collate_fn=collate_val_and_test, pin_memory=True, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                    collate_fn=collate_val_and_test, pin_memory=True, shuffle=False, num_workers=num_workers) if get_test else None
        
        return train_loader, val_loader, test_loader

def collate_train(data):
    U, S, neg_S = map(list, zip(*data))
    bs, vs = len(U), U[0].shape[0]

    bg = dgl.batch([ gs[idx] for gs in U for idx in range(vs)])
    for nty in bg.ntypes:
            bg.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
    for ety in bg.canonical_etypes:
        bg.set_e_initializer(dgl.init.zero_initializer, etype=ety)

    S = torch.cat(S, dim=0).reshape(bs, -1)
    neg_S = torch.cat(neg_S, dim=0).reshape(bs, -1)
    return bg, S, neg_S

def collate_val_and_test(data):
    U, S = map(list, zip(*data))
    bs, vs = len(U), U[0].shape[0]

    bg = dgl.batch([ gs[idx] for gs in U for idx in range(vs)])
    for nty in bg.ntypes:
            bg.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
    for ety in bg.canonical_etypes:
        bg.set_e_initializer(dgl.init.zero_initializer, etype=ety)

    S = torch.cat(S, dim=0).reshape(bs, -1)
    return bg, S

class SetDataset(Dataset):
    def __init__(self, dataset, V_idxs, S_idxs, params, is_train=False):
        _, self.graphs, _ = dataset
        self.V_idxs, self.S_idxs = V_idxs, S_idxs
        self.is_train = is_train
        self.neg_num = params.neg_num
        self.v_size = params.v_size
    
    def __getitem__(self, index):
        V_idxs, S = np.array(self.V_idxs[index]), np.array(self.S_idxs[index])
        V_graphs = np.array([self.graphs[item] for item in V_idxs])
        
        S_mask = torch.zeros([self.v_size])
        S_mask[S] = 1
        if self.is_train:
            idxs = (S_mask == 0).nonzero(as_tuple=True)[0]
            neg_S = idxs[torch.randperm(idxs.shape[0])[:S.shape[0] * self.neg_num]]
            neg_S_mask = torch.zeros([self.v_size])
            neg_S_mask[S] = 1
            neg_S_mask[neg_S] = 1
            return V_graphs, S_mask, neg_S_mask
        
        return V_graphs, S_mask

    def __len__(self):
        return len(self.V_idxs)

def load_pdbbind(params):
    params.subset = 'core'
    dataset = PDBBind(subset=params.subset)

    # decompose dataset
    _, ligand_mols, protein_mols, graphs, labels = map(list, zip(*dataset))
    ligand_mols = np.array(ligand_mols)
    graphs = np.array(graphs)
    labels = torch.stack(labels, dim=0)
    return (ligand_mols, graphs, labels)

def get_set_pdbbind_dataset_activate(dataset, V_size, S_size, size=1000):
    """
    Generate dataset for compound selection with only the bioactivity filter
    """
    mols, _, labels = dataset
    data_size = len(mols)

    V_list, S_list = [], []
    for _ in tqdm(range(size)):
        V_idxs = np.random.permutation(data_size)[:V_size]
        sub_labels = labels[V_idxs].squeeze(dim=-1)
        _, idxs = torch.topk(sub_labels, S_size)

        V_list.append(V_idxs)
        S_list.append(idxs)
    return np.array(V_list), np.array(S_list)

def get_set_pdbbind_dataset(dataset, V_size, S_size, size=1000):
    """
    Generate dataset for compound selection with bioactivity and diversity filters
    """
    mols, _, labels = dataset
    data_size = len(mols)

    V_list, S_list = [], []
    pbar = tqdm(total=size)
    num = 0
    while True:
        if num == size: break

        V_idxs = np.random.permutation(data_size)[:V_size]
        sub_labels = labels[V_idxs].squeeze(dim=-1)
        _, idxs = torch.topk(sub_labels, V_size // 3)
        filter_idxs = V_idxs[idxs]
        S_idxs = get_os_oracle(mols, filter_idxs)

        if len(S_idxs) == 0: continue
        
        V_list.append(V_idxs)
        S_list.append([np.where(V_idxs == item)[0][0] for item in S_idxs])
        
        num += 1
        pbar.update(1)
    return np.array(V_list), np.array(S_list)

def get_os_oracle(mols, filter_idxs):
    # reference: https://nyxflower.com/2020/07/17/molecule-clustering/
    mol_list = mols[filter_idxs]
    n = len(mol_list)

    sm = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            m1, m2 = mol_list[i], mol_list[j]
            sm[i, j] = FingerprintSimilarity(Chem.RDKFingerprint(m1), Chem.RDKFingerprint(m2))
    sm = sm + sm.T - np.eye(n)
    af = AffinityPropagation().fit(sm)
    cluster_centers_indices = af.cluster_centers_indices_
    return filter_idxs[cluster_centers_indices]
