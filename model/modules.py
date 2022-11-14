import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F

from model.acnn import ACNN
from utils.config import ACNN_CONFIG
from model.celebaCNN import celebaCNN
from model.deepDTA import DeepDTA_Encoder
from utils.pytorch_helper import FF, normal_cdf

class SetFuction(nn.Module):
    def __init__(self, params):
        super(SetFuction, self).__init__()
        self.params = params
        self.dim_feature = 256

        self.init_layer = self.define_init_layer()
        self.ff = FF(self.dim_feature, 500, 1, self.params.num_layers)

    def define_init_layer(self):
        data_name = self.params.data_name
        if data_name == 'moons':
            return nn.Linear(2, self.dim_feature)
        elif data_name == 'gaussian':
            return nn.Linear(2, self.dim_feature)
        elif data_name == 'amazon':
            return nn.Linear(768, self.dim_feature)
        elif data_name == 'celeba':
            return celebaCNN()
        elif data_name == 'pdbbind':
            return ACNN(hidden_sizes=ACNN_CONFIG['hidden_sizes'],
                        weight_init_stddevs=ACNN_CONFIG['weight_init_stddevs'],
                        dropouts=ACNN_CONFIG['dropouts'],
                        features_to_use=ACNN_CONFIG['atomic_numbers_considered'],
                        radial=ACNN_CONFIG['radial'])
        elif data_name == 'bindingdb':
            return DeepDTA_Encoder()
        else:
            raise ValueError("invalid dataset...")

    def MCsampling(self, q, M):
        bs, vs = q.shape
        q = q.reshape(bs, 1, 1, vs).expand(bs, M, vs, vs)
        sample_matrix = torch.bernoulli(q)

        mask = torch.cat([ torch.eye(vs, vs).unsqueeze(0) for _ in range(M)], dim=0).unsqueeze(0).to(q.device)
        matrix_0 = sample_matrix * (1 - mask)
        matrix_1 = matrix_0 + mask
        return matrix_1, matrix_0

    def mean_field_iteration(self, V, subset_i, subset_not_i):
        F_1 = self.F_S(V, subset_i, fpi=True).squeeze(-1)
        F_0 = self.F_S(V, subset_not_i, fpi=True).squeeze(-1)
        q = torch.sigmoid( (F_1 - F_0).mean(1) )
        return q

    def cross_entropy(self, q, S, neg_S):
        loss = - torch.sum( (S * torch.log(q + 1e-12) + (1 - S) * torch.log(1 - q + 1e-12)) * neg_S, dim=-1 )
        return loss.mean()

    def forward(self, V, S, neg_S, rec_net):
        q = rec_net.get_vardist(V, S.shape[0]).detach()

        for i in range(self.params.RNN_steps):
            sample_matrix_1, sample_matrix_0 = self.MCsampling(q, self.params.num_samples)
            q = self.mean_field_iteration(V, sample_matrix_1, sample_matrix_0)

        loss = self.cross_entropy(q, S, neg_S)
        return loss

    def F_S(self, V, subset_mat, fpi=False):
        if fpi:
            # to fix point iteration (aka mean-field iteration)
            fea = self.init_layer(V).reshape(subset_mat.shape[0], 1, -1, self.dim_feature)
        else:
            # to encode variational dist
            fea = self.init_layer(V).reshape(subset_mat.shape[0], -1, self.dim_feature)
        fea = subset_mat @ fea
        fea  = self.ff(fea)
        return fea

class RecNet(nn.Module):
    def __init__(self, params):
        super(RecNet, self).__init__()
        self.params = params
        self.dim_feature = 256
        num_layers = self.params.num_layers
        
        self.init_layer = self.define_init_layer()
        self.ff = FF(self.dim_feature, 500, 500, num_layers-1 if num_layers>0 else 0)
        self.h_to_mu = nn.Linear(500, 1)
        self.h_to_std = nn.Linear(500, 1)
        self.h_to_U = nn.ModuleList([nn.Linear(500, 1) for i in range(self.params.rank)])

    def define_init_layer(self):
        data_name = self.params.data_name
        if data_name == 'moons':
            return nn.Linear(2, self.dim_feature)
        elif data_name == 'gaussian':
            return nn.Linear(2, self.dim_feature)
        elif data_name == 'amazon':
            return nn.Linear(768, self.dim_feature)
        elif data_name == 'celeba':
            return celebaCNN()
        elif data_name == 'pdbbind':
            return ACNN(hidden_sizes=ACNN_CONFIG['hidden_sizes'],
                        weight_init_stddevs=ACNN_CONFIG['weight_init_stddevs'],
                        dropouts=ACNN_CONFIG['dropouts'],
                        features_to_use=ACNN_CONFIG['atomic_numbers_considered'],
                        radial=ACNN_CONFIG['radial'])
        elif data_name == 'bindingdb':
            return DeepDTA_Encoder()
        else:
            raise ValueError("invalid dataset...")

    def encode(self, V, bs):
        fea = self.init_layer(V).reshape(bs, -1, self.dim_feature)
        h = torch.relu(self.ff(fea))
        
        ber = torch.sigmoid(self.h_to_mu(h)).squeeze(-1)
        std = F.softplus(self.h_to_std(h)).squeeze(-1)
        rs = []
        for i in range(self.params.rank):
            rs.append(torch.tanh(self.h_to_U[i](h)))
        u_perturbation = torch.cat(rs, -1)

        return ber, std, u_perturbation

    def MCsampling(self, ber, std, u_pert, M):
        """
        ber: location parameter (0, 1)               [batch_size, v_size]
        std: standard deviation (0, +infinity)      [batch_size, v_size]
        u_pert: lower rank perturbation (-1, 1)     [batch_size, v_size, rank]
        M: number of MC approximation
        """
        bs, vs = ber.shape

        eps = torch.randn((bs, M, vs)).to(ber.device)
        eps_corr = torch.randn((bs, M, self.params.rank, 1)).to(ber.device)
        g = eps * std.unsqueeze(1) + torch.matmul(u_pert.unsqueeze(1), eps_corr).squeeze(-1)
        u = normal_cdf(g, 0, 1)
        
        ber = ber.unsqueeze(1)
        l = torch.log(ber + 1e-12) - torch.log(1 - ber + 1e-12) + \
                torch.log(u + 1e-12) - torch.log(1 - u + 1e-12)
        
        # understanding Gumbel softmax for binary cases: https://j-zin.github.io/files/Gumbel_Softmax_for_Binary_Case.pdf
        prob = torch.sigmoid(l / self.params.tau)
        r = torch.bernoulli(prob)
        s = prob + (r - prob).detach()  # straight through estimator
        return s

    def cal_elbo(self, V, sample_mat, set_func, q):
        f_mt = set_func.F_S(V, sample_mat).squeeze(-1).mean(-1)
        entropy = - torch.sum(q * torch.log(q + 1e-12) + (1 - q) * torch.log(1 - q + 1e-12), dim=-1)
        elbo = f_mt + entropy
        return elbo.mean()

    def forward(self, V, set_func, bs):
        ber, std, u_perturbation = self.encode(V, bs)
        sample_mat = self.MCsampling(ber, std, u_perturbation, self.params.num_samples)
        elbo = self.cal_elbo(V, sample_mat, set_func, ber)
        return -elbo

    def get_vardist(self, V, bs):
        fea = self.init_layer(V).reshape(bs, -1, self.dim_feature)
        h = torch.relu(self.ff(fea))
        ber = torch.sigmoid(self.h_to_mu(h)).squeeze(-1)
        return ber
