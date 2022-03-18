import os
import torch
import gdown
import pickle
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

image_size = 64
img_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
])
img_root_path = '/root/dataset/celeba/img_align_celeba/'

class Data:
    def __init__(self, params):
        self.params = params
    
    def gen_datasets(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        raise NotImplementedError

class CelebA(Data):
    def __init__(self, params):
        super().__init__(params)
        data_root = self.download_celeba()

        torch.manual_seed(1)    # fix dataset
        np.random.seed(1)
        self.gen_datasets(data_root)
    
    def download_celeba(self):
        url_img = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
        url_anno = 'https://drive.google.com/uc?id=1p0-TEiW4HgT8MblB399ep4YM3u5A0Edc'
        data_root = '/root/dataset/celeba'
        download_path_img = f'{data_root}/img_align_celeba.zip'
        download_path_anno = f'{data_root}/list_attr_celeba.txt'
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        
        if not os.listdir(data_root):
            gdown.download(url_anno, download_path_anno, quiet=False)
            gdown.download(url_img, download_path_img, quiet=False)
            with zipfile.ZipFile(download_path_img, 'r') as ziphandler:
                ziphandler.extractall(data_root)
        return data_root

    def load_data(self, data_root):
        data_path = data_root + '/list_attr_celeba.txt'
        df = pd.read_csv(data_path, sep="\s+", skiprows=1)
        label_names = list(df.columns)[:-1]
        df = df.to_numpy()[:, :-1]
        df = np.maximum(df, 0)  # -1 -> 0
        return df, label_names

    def gen_datasets(self, data_root):
        data_path = data_root + '/celebA_set_data.pkl'
        if os.path.exists(data_path):
            print(f'load data from {data_path}')
            label_names, trainData, valData, testData = pickle.load(open(data_path, "rb"))
            self.V_train, self.S_train, self.labels_train = trainData['V_train'], trainData['S_train'], trainData['labels_train']
            self.V_val, self.S_val, self.labels_val = valData['V_train'], valData['S_train'], valData['labels_train']
            self.V_test, self.S_test, self.labels_test = testData['V_train'], testData['S_train'], testData['labels_train']
        else:
            data, label_names = self.load_data(data_root)
            self.V_train, self.S_train, self.labels_train = get_set_celeba_dataset(data, data_size=10000, v_size=self.params.v_size)
            self.V_val, self.S_val, self.labels_val = get_set_celeba_dataset(data, data_size=1000, v_size=self.params.v_size)
            self.V_test, self.S_test, self.labels_test = get_set_celeba_dataset(data, data_size=1000, v_size=self.params.v_size)
            
            trainData = {'V_train': self.V_train, 'S_train': self.S_train, 'labels_train': self.labels_train}
            valData = {'V_train': self.V_val, 'S_train': self.S_val, 'labels_train': self.labels_val}
            testData = {'V_train': self.V_test, 'S_train': self.S_test, 'labels_train': self.labels_test}
            pickle.dump((label_names, trainData, valData, testData), open(data_path, "wb"))

    def get_loaders(self, batch_size, num_workers, shuffle_train=False, get_test=True):
        train_dataset = SetDataset(self.V_train, self.S_train, self.params, is_train=True)
        val_dataset = SetDataset(self.V_val, self.S_val, self.params)
        test_dataset = SetDataset(self.V_test, self.S_test, self.params)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                    collate_fn=collate_train, shuffle=shuffle_train, num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                    collate_fn=collate_val_and_test, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                    collate_fn=collate_val_and_test, shuffle=False, num_workers=num_workers) if get_test else None
        return train_loader, val_loader, test_loader

class SetDataset(Dataset):
    def __init__(self, U, S, params, is_train=False):
        self.data = U
        self.labels = S
        self.is_train = is_train
        self.neg_num = params.neg_num
        self.v_size = params.v_size

    def __getitem__(self, index):
        V_id = self.data[index]
        S = self.labels[index]
        V = torch.cat([ load_img(idx.item()) for idx in V_id ], dim=0)

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

def get_set_celeba_dataset(data, data_size, v_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.Tensor(data).to(device)
    img_nums = data.shape[0]

    V_list = []
    S_list = []
    label_list = []
    cur_size = 0
    pbar = tqdm(total=data_size)
    while True:
        if cur_size == data_size: break

        nor_id = np.random.randint(img_nums)
        nor_label = data[nor_id]
        if torch.sum(nor_label).item() < 2:
            continue
        nor_lable_idxs = torch.nonzero(nor_label).reshape(-1)
        perm = torch.randperm(nor_lable_idxs.size(0))
        nor_lable_idxs = nor_lable_idxs[perm[:2]]
        nor_label = torch.zeros(nor_label.shape).to(device)
        nor_label[nor_lable_idxs] = 1
        nor_label = nor_label.reshape(-1, 1)

        s_size = np.random.randint(2, 4)
        nor_res = (torch.nonzero((data @ nor_label).squeeze(-1) == 2)).reshape(-1)
        ano_res = (torch.nonzero((data @ nor_label).squeeze(-1) == 0)).reshape(-1)
        if (nor_res.shape[0] < v_size) or (ano_res.shape[0] < s_size):
            continue
        
        perm = torch.randperm(nor_res.size(0))
        U = nor_res[perm[:v_size]].cpu()
        perm = torch.randperm(ano_res.size(0))
        S = ano_res[perm[:s_size]].cpu()

        S_idx = np.random.choice(list(range(v_size)), s_size, replace=False)
        U[S_idx] = S
        S = torch.Tensor(S_idx).type(torch.int64)
        lable_idxs = nor_lable_idxs.cpu()
        
        V_list.append(U)
        S_list.append(S)
        label_list.append(lable_idxs)
        
        cur_size += 1
        pbar.update(1)
    pbar.close()
    return V_list, S_list, label_list

def collate_train(data):
    V, S, neg_S = map(list, zip(*data))
    bs = len(V)

    V = torch.cat(V, dim=0)
    S = torch.cat(S, dim=0).reshape(bs, -1)
    neg_S = torch.cat(neg_S, dim=0).reshape(bs, -1)
    return V, S, neg_S

def collate_val_and_test(data):
    V, S = map(list, zip(*data))
    bs = len(V)

    V = torch.cat(V, dim=0)
    S = torch.cat(S, dim=0).reshape(bs, -1)
    return V, S

def load_img(img_id):
    img_id = str(img_id + 1)    # 0 -> 1
    img_path =  img_root_path + ( '0' * (6 - len(img_id)) ) + img_id + '.jpg'
    img = Image.open(img_path).convert('RGB')
    img = img_transform(img).unsqueeze(0)
    return img
