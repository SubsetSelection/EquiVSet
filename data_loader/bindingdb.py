import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from tdc.multi_pred import DTI
from multiprocessing import Pool
from rdkit.DataStructs import FingerprintSimilarity
from sklearn.cluster import AffinityPropagation

from torch.utils.data import Dataset, DataLoader

acid_list = ['H', 'M', 'C', 'P', 'L', 'A', 'R', 'F', 'D', 'T', 'K', 'E', 'S', 'V', 'G', 'Y', 'N', 'W', 'I', 'Q']
CHARPROTLEN  = len(acid_list)
CHARPROTDIC = { acid_list[idx]: idx for idx in range(len(acid_list))}

smile_list = ['o', 'N', '/', 'H', '#', 'C', 'i', '+', 'l', '@', '8', '-', '6', '3', '\\', '2', 'B', 'P', '.', 
            'e', '9', '7', 'a', 's', 'O', ')', '0', 'n', '1', '4', 'I', 'F', ']', 'S', '5', '(', '[', '=', '%', 'c', 'r']
CHARCANSMILEN = len(smile_list)
CHARCANSMIDIC = { smile_list[idx]: idx for idx in range(len(smile_list))}

class Tokenizer():
    @staticmethod
    def seq_tokenizer(seq, type_):
        if type_ == 'drug':
            max_length = 100
            mask = torch.zeros([max_length, CHARCANSMILEN])
            seq = np.array([CHARCANSMIDIC[item] for item in seq.split(" ")[:max_length]])
        elif type_ == 'protein':
            max_length = 1000
            mask = torch.zeros([max_length, CHARPROTLEN])
            seq = np.array([CHARPROTDIC[item] for item in seq.split(" ")[:max_length]])
        
        length = seq.shape[0]
        mask[range(length), seq] = 1
        return mask.transpose_(0, 1).unsqueeze(0)

    @staticmethod
    def tokenizer(bt, type_):
        bt = [Tokenizer.seq_tokenizer(seq, type_) for seq in bt]
        return bt

class SetBindingDB(object):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.gen_datasets()

    def gen_datasets(self):
        np.random.seed(1)   # fix dataset
        V_size, S_size = self.params.v_size, self.params.s_size
        self.dataset = load_bindingdb(self.params)

        data_root = '/root/dataset/bindingdb'
        data_path = os.path.join(data_root, 'bindingdb_set_data.pkl')
        if os.path.exists(data_path):
            print(f'load data from {data_path}')
            trainData, valData, testData = pickle.load(open(data_path, "rb"))
            self.V_train, self.S_train = trainData['V_train'], trainData['S_train']
            self.V_val, self.S_val = valData['V_train'], valData['S_train']
            self.V_test, self.S_test = testData['V_train'], testData['S_train']
        else:
            self.V_train, self.S_train = get_set_bindingdb_dataset_activate(self.dataset, V_size, S_size, self.params, size=1000)
            self.V_val, self.S_val = get_set_bindingdb_dataset_activate(self.dataset, V_size, S_size, self.params, size=100)
            self.V_test, self.S_test = get_set_bindingdb_dataset_activate(self.dataset, V_size, S_size, self.params, size=100)
            
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
    V_drug, V_target, S, neg_S = map(list, zip(*data))
    bs, vs = len(V_drug), V_drug[0].shape[0]

    b_D = [ gs[idx] for gs in V_drug for idx in range(vs)]
    b_P = [ gt[idx] for gt in V_target for idx in range(vs)]
    b_D = Tokenizer.tokenizer(b_D, 'drug')
    b_P = Tokenizer.tokenizer(b_P, 'protein')

    b_D = torch.cat(b_D, dim=0)
    b_P = torch.cat(b_P, dim=0)
    S = torch.cat(S, dim=0).reshape(bs, -1)
    neg_S = torch.cat(neg_S, dim=0).reshape(bs, -1)
    return (b_D, b_P), S, neg_S

def collate_val_and_test(data):
    V_drug, V_target, S = map(list, zip(*data))
    bs, vs = len(V_drug), V_drug[0].shape[0]

    b_D = [ gs[idx] for gs in V_drug for idx in range(vs)]
    b_P = [ gt[idx] for gt in V_target for idx in range(vs)]
    b_D = Tokenizer.tokenizer(b_D, 'drug')
    b_P = Tokenizer.tokenizer(b_P, 'protein')

    b_D = torch.cat(b_D, dim=0)
    b_P = torch.cat(b_P, dim=0)
    S = torch.cat(S, dim=0).reshape(bs, -1)
    return (b_D, b_P), S

class SetDataset(Dataset):
    def __init__(self, dataset, V_idxs, S_idxs, params, is_train=False):
        self.drugs, self.targets = dataset['Drug'], dataset['Target']
        self.V_idxs, self.S_idxs = V_idxs, S_idxs
        self.is_train = is_train
        self.neg_num = params.neg_num
        self.v_size = params.v_size

    def __getitem__(self, index):
        V_idxs, S = np.array(self.V_idxs[index]), np.array(self.S_idxs[index])
        V_drug = np.array([" ".join(item) for item in self.drugs[V_idxs].tolist()])
        V_target = np.array([" ".join(item) for item in self.targets[V_idxs].tolist()])

        S_mask = torch.zeros([self.v_size])
        S_mask[S] = 1
        if self.is_train:
            idxs = (S_mask == 0).nonzero(as_tuple=True)[0]
            neg_S = idxs[torch.randperm(idxs.shape[0])[:S.shape[0] * self.neg_num]]
            neg_S_mask = torch.zeros([self.v_size])
            neg_S_mask[S] = 1
            neg_S_mask[neg_S] = 1
            return V_drug, V_target, S_mask, neg_S_mask

        return V_drug, V_target, S_mask

    def __len__(self):
        return len(self.V_idxs)

def load_bindingdb(params):
    data = DTI(name = 'BindingDB_Kd', path='/root/dataset/bindingdb')
    data.harmonize_affinities(mode = 'mean')
    return data.get_data()

def load_dt_pair(dataset, setdata):
    drugs, targets = dataset['Drug'], dataset['Target']
    V, S = setdata['V_train'], setdata['S_train']

    V_drugs_list, V_targets_list = [], []
    for V_idxs in V:
        V_drugs = [" ".join(item) for item in drugs[V_idxs].tolist()]
        V_drugs_list.append(V_drugs)
        V_targets = [" ".join(item) for item in targets[V_idxs].tolist()]
        V_targets_list.append(V_targets)
    
    V_drug = np.array(V_drugs_list)
    V_target = np.array(V_targets_list)
    S = torch.Tensor(S).type(torch.long)
    return (V_drug, V_target), S

def get_set_bindingdb_dataset_activate(dataset, V_size, S_size, params, size=1000):
    """
    Generate dataset for compound selection with only the bioactivity filter
    """
    _, _, labels = dataset['Drug'], dataset['Target'], dataset['Y']
    data_size = len(labels)

    V_list, S_list = [], []
    for _ in tqdm(range(size)):
        V_idxs = np.random.permutation(data_size)[:V_size]
        sub_labels = torch.from_numpy(labels[V_idxs].to_numpy())
        _, idxs = torch.topk(sub_labels, S_size)

        V_list.append(V_idxs)
        S_list.append(idxs)
    return np.array(V_list), np.array(S_list)

def get_set_bindingdb_dataset(dataset, V_size, S_size, params, size=1000):
    """
    Generate dataset for compound selection with bioactivity and diversity filters
    """
    drugs, targets, labels = dataset['Drug'], dataset['Target'], dataset['Y']
    data_size = len(labels)

    V_list, S_list = [], []
    pbar = tqdm(total=size)
    num = 0
    while True:
        if num == size: break

        V_idxs = np.random.permutation(data_size)[:V_size]
        sub_labels = torch.from_numpy(labels[V_idxs].to_numpy())
        _, idxs = torch.topk(sub_labels, V_size // 3)
        filter_idxs = V_idxs[idxs]
        S_idxs = get_os_oracle(drugs, filter_idxs)

        if len(S_idxs) == 0: continue

        V_list.append(V_idxs)
        S_list.append([np.where(V_idxs == item)[0][0] for item in S_idxs])

        num += 1
        pbar.update(1)
    return np.array(V_list), np.array(S_list)

def get_os_oracle(drugs, filter_idxs):
    smile_list = drugs[filter_idxs].to_numpy()
    n = len(smile_list)
    
    sm = np.zeros((n, n))
    max_cpu = os.cpu_count()
    ij_list = [(i, j, smile_list) for i in range(n) for j in range(i+1, n)]
    with Pool(max_cpu) as p:
        similarity = p.starmap(cal_fingerprint_similarity, ij_list)
    i, j, _ = zip(*ij_list)
    sm[i,j] = similarity
    sm = sm + sm.T + np.eye(n)

    af = AffinityPropagation().fit(sm)
    cluster_centers_indices = af.cluster_centers_indices_
    return filter_idxs[cluster_centers_indices]

def cal_fingerprint_similarity(i, j, smiles):
    m1, m2 = Chem.MolFromSmiles(smiles[i]), Chem.MolFromSmiles(smiles[j])
    return FingerprintSimilarity(Chem.RDKFingerprint(m1), Chem.RDKFingerprint(m2))
