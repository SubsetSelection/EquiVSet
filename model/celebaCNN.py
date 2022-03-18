import torch.nn as nn
import torch.nn.functional as F

class celebaCNN(nn.Sequential):
    def __init__(self):
        super(celebaCNN, self).__init__()

        in_ch = [3] + [32,64,128]
        kernels = [3,4,5]
        strides = [2,2,2]
        layer_size = 3
        self.conv = nn.ModuleList([nn.Conv2d(in_channels = in_ch[i], 
                                                out_channels = in_ch[i+1], 
                                                kernel_size = kernels[i],
                                                stride = strides[i]) for i in range(layer_size)])
        self.conv = self.conv.double()
        self.fc1 = nn.Linear(128, 256)

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool2d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v