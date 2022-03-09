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
|Toys|89.05 (0.35) |89.91 (1.78) |88.79 (2.23) |89.34 (0.38) |89.62 (2.04 |82.14 (0.86) |  
|Furniture|88.14 (0.17) |82.82 (0.87) |90.67 (0.07) |88.39 (0.25) |83.10 (0.46 |82.41 (0.20) |  
|Gear|88.59 (0.10) |88.10 (1.42) |91.09 (0.15) |88.88 (0.15) |88.23 (1.42 |83.04 (0.08) |  
|Carseats|88.47 (0.20) |83.39 (1.15) |91.32 (0.38) |88.72 (0.16) |83.20 (1.28 |83.22 (0.10) |  
|Bath|88.80 (0.52) |89.01 (2.06) |88.76 (1.92) |89.01 (0.43) |88.95 (2.17 |81.65 (1.06) |  
|Health|88.80 (0.15) |89.42 (0.43) |89.81 (0.41) |88.96 (0.15) |89.24 (0.82 |82.62 (0.23) |  

#### Experiment 2: <span id="exp2">Set Anomaly Detection</span>

#### Experiment 3: <span id="exp3">Compound Selection in AI-aided Drug Discovery</span>

<!-- To cite:   -->
