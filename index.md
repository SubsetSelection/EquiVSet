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

The oral presentation can be download [here](https://github.com/SubsetSelection/EquiVSet/raw/gh-pages/files/equivset_slides.pdf).

### Code and Document

To be released.

### Experiments

**contents** <br>
**Exp 1**: [Product Recommendation](#exp1) <br>
**Exp 2**: [Set Anomaly Detection](#exp2) <br>
**Exp 3**: [Compound Selection in AI-aided Drug Discovery](#exp3)

#### Experiment 1: <span id="exp1">Product Recommendation</span>

#### Experiment 2: <span id="exp2">Set Anomaly Detection</span>

#### Experiment 3: <span id="exp3">Compound Selection in AI-aided Drug Discovery</span>

<!-- To cite:   -->
