## Learning Set Functions Under the Optimal Subset Oracle via Equivariant Variational Inference

<center>
Zijing Ou<sup>1,2</sup>, Tingyang Xu<sup>1</sup>, Qinliang Su<sup>2</sup>, Yingzhen Li<sup>3</sup>, Peilin Zhao<sup>1</sup>, Yatao Bian<sup>1</sup> <br>
<sup>1</sup>Tencent AI Lab, China <br>
<sup>2</sup>Sun Yat-sen University, China <br>
<sup>3</sup>Imperial College London, United Kingdom <br>
</center>

### Project Description

Learning set functions becomes increasingly more important in many applications like product recommendation and compound selection in AI-aided drug discovery. The majority of existing works study methodologies of set function learning under the function value oracle, which, however, requires expensive supervision signals. This renders it impractical for applications with only weak supervisions under the Optimal Subset (OS) oracle, the study of which is surprisingly overlooked. In this work, we present a principled yet practical maximum likelihood learning framework, termed as EquiVSet, that simultaneously meets the following desiderata of learning set functions under the OS oracle: i) permutation invariance of the set mass function being modeled; ii) permission of varying ground set; iii) fully differentiability; iv) minimum prior; and v) scalability. The main components of our framework involve: an energy-based treatment of the set mass function, DeepSet-style architectures to handle permutation invariance, mean-field variational inference, and its amortized variants. Although the framework is embarrassingly simple, empirical studies on three real-world applications (including Amazon product recommendation, set anomaly detection and compound selection for virtual screening) demonstrate that EquiVSet outperforms the baselines by a large margin.

### Paper

Preprint: [Arxiv](https://arxiv.org/abs/2203.01693)

### Slides

The oral presentation can be download [here](files/equivset_slides.pdf).

### Code and Document

To be released.

### Experiments

**contents** <br>
**Exp 1**: [Product Recommendation](#exp1) <br>
**Exp 2**: [Set Anomaly Detection](#exp2) <br>
**Exp 3**: [Compound Selection in AI-aided Drug Discovery](#exp3)

#### Experiment 1: <span id="exp1">Product Recommendation</span>

<center><b>Table 1: Product recommendation results in the MJC metric on the Amazon dataset with different categories.</b></center>

| Categories | Random | PGM | DeepSet | DiffMF (ours) | EquiVSet<sub>ind</sub> (ours) | EquiVSet<sub>copula</sub> (ours) |  
| ------ | ------     | --------- |    ----------- | ------------| -------------|------------|---------------|  
|Toys| 0.0832 | 0.4414 ± 0.0036 | 0.4287 ± 0.0047 | 0.6147 ± 0.0102 | 0.6491 ± 0.0152 | 0.6762 <span>&#177;</span> 0.0221 |  
|Furniture| 0.0651 | 0.1746 ± 0.0069 | 0.1758 ± 0.0072 | 0.1744 ± 0.0121 | 0.1775 ± 0.0108 | 0.1724 ±  0.0091 |  
|Gear| 0.0771 | 0.4712 ± 0.0037 | 0.3806 ± 0.0019 | 0.5622± 0.0171 | 0.6103 ± 0.0193 | 0.6973 ±  0.0119 |
|Carseats| 0.0659 | 0.2330 ± 0.0115 | 0.2121 ± 0.0096 | 0.2229 ± 0.0104 | 0.2141 ± 0.0073 | 0.2149 ±  0.0123 |
|Bath| 0.0763 | 0.5638 ± 0.0077 | 0.4241 ± 0.0058 | 0.6901 ± 0.0061 | 0.6457 ± 0.0200 | 0.7567 ± 0.0095 |
|Health| 0.0758 | 0.4493 ± 0.0024 | 0.4481 ± 0.0041 | 0.5650± 0.0092 | 0.6315 ± 0.0153 | 0.7003 ± 0.0159 |
|Diaper| 0.0839 | 0.5802± 0.0092 | 0.4572 ± 0.0050 | 0.7011 ± 0.0112 | 0.7344 ± 0.0199 | 0.8275 ± 0.0136 |
|Bedding| 0.0791 | 0.4799 ± 0.0061 | 0.4824 ± 0.0081 | 0.6408± 0.0093 | 0.6287 ± 0.0195 | 0.7688 ±  0.0121 |
|Safety| 0.0648 | 0.2495 ± 0.0060 | 0.2211 ± 0.0044 | 0.2007± 0.0527 | 0.2250 ± 0.0287 | 0.2524 ± 0.0285 |
|Feeding| 0.0925 | 0.5596 ± 0.0081 | 0.4295 ± 0.0021 | 0.7496 ± 0.0114 | 0.6955 ± 0.0063 | 0.8101 ± 0.0074 |
|Apparel| 0.0918 | 0.5333 ± 0.0050 | 0.5074 ± 0.0036 | 0.6708± 0.0225 | 0.6465 ± 0.0150 | 0.7521 ±  0.0114 |
|Media| 0.0944 | 0.4406 ± 0.0092 | 0.4241 ± 0.0105 | 0.5145 ± 0.0105 | 0.5506 ± 0.0072 | 0.5694 ± 0.0105 |

#### Experiment 2: <span id="exp2">Set Anomaly Detection</span>

<center><b>Table 2: Set anomaly detection results in the MJC metric.</b></center>

| Method | Double MNIST | CelebA |
| ------ | ------     | --------- |
|Random| 0.0816 | 0.2187 |
|PGM| 0.3031 ± 0.0118 | 0.4812 ± 0.0064 |
|DeepSet| 0.1108 ± 0.0031 | 0.3915 ± 0.0133 |
|DiffMF (ours)| 0.6064 ± 0.0133 | 0.5455 ± 0.0079 |
|EquiVSet<sub>ind</sub> (ours)| 0.4054 ± 0.0122 | 0.5310 ± 0.0123 |
|EquiVSet<sub>copula</sub> (ours)| 0.5878 ± 0.0068 | 0.5549 ± 0.0053 |

#### Experiment 3: <span id="exp3">Compound Selection in AI-aided Drug Discovery</span>

<center><b>Table 3: Compound selection results in the MJC metric.</b></center>

| Method | PDBBind | BindingDB |
| ------ | ------     | --------- |
|Random| 0.0725 | 0.0267 |
|PGM| 0.3499 ± 0.0087 | 0.1760 ± 0.0055 |
|DeepSet| 0.3189 ± 0.0034 | 0.1615 ± 0.0074 |
|DiffMF (ours)| 0.3534 ± 0.0143 | 0.1894 ± 0.0021 |
|EquiVSet<sub>ind</sub> (ours)| 0.3553 ± 0.0049 | 0.1904 ± 0.0034 |
|EquiVSet<sub>copula</sub> (ours)| 0.3536 ± 0.0083 | 0.1875 ± 0.0032 |

<!-- To cite:   -->
