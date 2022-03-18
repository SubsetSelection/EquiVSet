import torch
import numpy as np

MOONS_CONFIG = {
    'data_name': 'moons',
    'v_size': 100,
    's_size': 10,
    'batch_size': 128
}

GAUSSIAN_CONFIG = {
    'data_name': 'gaussian',
    'v_size': 100,
    's_size': 10,
    'batch_size': 128
}

AMAZON_CONFIG = {
    'data_name': 'amazon',
    'v_size': 30,
    'batch_size': 128
}

CELEBA_CONFIG = {
    'data_name': 'celeba',
    'v_size': 8,
    'batch_size': 128
}

PDBBIND_CONFIG = {
    'data_name': 'pdbbind',
    'v_size': 30,
    's_size': 5,
    'batch_size': 32
}

BINDINGDB_CONFIG = {
    'data_name': 'bindingdb',
    'v_size': 300,
    's_size': 15,
    'batch_size': 4
}

ACNN_CONFIG = {
    'hidden_sizes': [32, 32, 16],
    'weight_init_stddevs': [1. / float(np.sqrt(32)), 1. / float(np.sqrt(32)),
                            1. / float(np.sqrt(16)), 0.01],
    'dropouts': [0., 0., 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[12.0], [0.0, 4.0, 8.0], [4.0]],
}