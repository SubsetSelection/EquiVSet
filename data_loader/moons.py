import torch
import numpy as np
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader

class Data:
    def __init__(self, params):
        self.params = params
        self.gen_datasets()
    
    def gen_datasets(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        raise NotImplementedError

class TwoMoons(Data):
    def __init__(self, params):
        super().__init__(params)   

    def gen_datasets(self):
        np.random.seed(1)   # fix dataset
        V_size, S_size = self.params.v_size, self.params.s_size
        
        self.V_train, self.S_train = get_two_moons_dataset(V_size, S_size, rand_seed=0)
        self.V_val, self.S_val = get_two_moons_dataset(V_size, S_size, rand_seed=1)
        self.V_test, self.S_test = get_two_moons_dataset(V_size, S_size, rand_seed=2)

        self.fea_size = 2
        self.x_lim, self.y_lim = 4, 2

    def get_loaders(self, batch_size, num_workers, shuffle_train=False, get_test=True):
        train_dataset = SetDataset(self.V_train, self.S_train, self.params, is_train=True)
        val_dataset = SetDataset(self.V_val, self.S_val, self.params)
        test_dataset = SetDataset(self.V_test, self.S_test, self.params)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                    shuffle=shuffle_train, num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers) if get_test else None
        return train_loader, val_loader, test_loader

class SetDataset(Dataset):
    def __init__(self, V, S, params, is_train=False):
        self.data = V
        self.labels = S
        self.is_train = is_train
        self.neg_num = params.neg_num
        self.v_size = params.v_size

    def __getitem__(self, index):
        V = self.data[index]
        S = self.labels[index]

        S_mask = torch.zeros([self.v_size])
        S_mask[S] = 1
        if self.is_train:
            idxs = (S_mask == 0).nonzero(as_tuple=True)[0]
            neg_S = idxs[torch.randperm(idxs.shape[0])[:S.shape[0] * self.neg_num]]
            neg_S_mask = torch.zeros([self.v_size])
            neg_S_mask[S] = 1
            neg_S_mask[neg_S] = 1
            return V, S_mask, neg_S_mask
        
        return V, S_mask
    
    def __len__(self):
        return len(self.data)

def gen_moons(batch_size, rand_seed):
    data, Y = datasets.make_moons(n_samples=batch_size, noise=0.1, random_state=rand_seed)
    data = data.astype("float32")
    data = data * 2 + np.array([-1, -0.2])

    noise_label = np.random.randint(2)
    noise = data[Y == noise_label]
    data = data[Y == (1 - noise_label)]
    return data, noise

def get_two_moons_dataset(V_size, S_size, rand_seed):
    V_list, S_list = [], []
    for idx in range(1000):
        data, noise = gen_moons(V_size*2, rand_seed*1000+idx)

        V = data[:V_size]
        S = np.random.choice(list(range(0,V_size)), S_size, replace=False)
        V[S, :] = noise[:S_size]
        
        V_list.append(V)
        S_list.append(S)
    
    V = torch.FloatTensor(np.array(V_list))
    S = torch.LongTensor(np.array(S_list))

    return V, S
