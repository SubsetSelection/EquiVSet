# reference: https://github.com/mims-harvard/TDC/blob/main/examples/multi_pred/dti_dg/domainbed/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x
    
class CNN(nn.Sequential):
    def __init__(self, encoding):
        super(CNN, self).__init__()
        if encoding == 'drug':
            in_ch = [41] + [32,64,96]
            kernels = [4,6,8]
            layer_size = 3
            self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
                                                    out_channels = in_ch[i+1], 
                                                    kernel_size = kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output((41, 100))
            self.fc1 = nn.Linear(n_size_d, 256)
        elif encoding == 'protein':
            in_ch = [20] + [32,64,96]
            kernels = [4,8,12]
            layer_size = 3
            self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
                                                    out_channels = in_ch[i+1], 
                                                    kernel_size = kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_p = self._get_conv_output((20, 1000))
            self.fc1 = nn.Linear(n_size_p, 256)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v
    
class DeepDTA_Encoder(nn.Sequential):
    def __init__(self):
        super(DeepDTA_Encoder, self).__init__()
        self.input_dim_drug = 256
        self.input_dim_protein = 256

        self.model_drug = CNN('drug')
        self.model_protein = CNN('protein')
        self.predictor = nn.Linear(self.input_dim_drug + self.input_dim_protein, 256)

    def forward(self, V):
        v_D, v_P = V
        # each encoding
        v_D = self.model_drug(v_D)
        v_P = self.model_protein(v_P)
        # concatenate and output feature
        v_f = torch.cat((v_D, v_P), 1)
        v_f = self.predictor(v_f)
        return v_f
