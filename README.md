# :fire:EquiVSet:fire:

This repo contains PyTorch implementation of the paper "Learning Set Functions Under the Optimal Subset Oracle via Equivariant Variational Inference".

## Installation

Please ensure that:

- Python >= 3.6
- PyTorch >= 1.8.0
- dgl >= 0.7.0

The following pakages are needed if you want to run the `compound selection` experiments:

- **rdkit**: We recommend installing it with `conda install -c rdkit rdkit==2018.09.3`. For other installation recipes, see the [official documentation](https://www.rdkit.org/docs/Install.html).
- **dgllife**: We recommend installing it with `pip install dgllife`. More information is available in the [official documentation](https://lifesci.dgl.ai/install/index.html).
- **tdc**: We recommend installing it with `pip install PyTDC`. See the [official documentation](https://tdc.readthedocs.io/en/main/install.html) for more information.

We provide step-by-step installation commands as follows:

```
conda create -n EquiVSet python=3.7
source activate EquiVSet
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl-cu110 dglgo -f https://data.dgl.ai/wheels/repo.html

# The following commands are used for compound selection:
conda install -c rdkit rdkit==2018.09.3
pip install dgllife
pip install PyTDC
```

## Experiments

This repository implements the synthetic experiments (section 6.1), product recommendation (section 6.2), set anomaly detection (section 6.3), and compound selection (section 6.4).

### Synthetic Experiments

To run on the Two-Moons and Gaussian-Mixture dataset
```
python main.py equivset --train --cuda --data_name <dataset_name>
```
`dataset_name` is chosen in ['moons', 'gaussian'].

### Product Recommendation

To run on the Amazon baby registry dataset
```
python main.py equivset --train --cuda --data_name amazon --amazon_cat <category_name>
```
`category_name` is chosen in ['toys', 'furniture', 'gear', 'carseats', 'bath', 'health', 'diaper', 'bedding', 'safety', 'feeding', 'apparel', 'media'].

### Set Anomaly Detection

To run on the CelebA dataset
```
python main.py equivset --train --cuda --data_name celeba
```

### Compound Selection

To run on the PDBBind and BindingDB dataset
```
python main.py equivset --train --cuda --data_name <dataset_name>
```
`dataset_name` is chosen in ['pdbbind', 'bindingdb'].

## Citation

:smile:If you find this repo is useful, please consider to cite our paper:
```
@article{ou2022learning,
  title={Learning Set Functions Under the Optimal Subset Oracle via Equivariant Variational Inference},
  author={Ou, Zijing and Xu, Tingyang and Su, Qinliang and Li, Yingzhen and Zhao, Peilin and Bian, Yatao},
  journal={arXiv preprint arXiv:2203.01693},
  year={2022}
}
```