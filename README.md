# MeDUET
<a href="https://arxiv.org/abs/2602.17901"><img src='https://img.shields.io/badge/arXiv-MeDUET-red' alt='Paper PDF'></a>

**MeDUET** is a unified pretraining framework for **3D medical image synthesis and analysis** in the **VAE latent space**.

Our core idea is to treat unified pretraining under multi-center style shifts as a **factor identifiability** problem. MeDUET learns to disentangle each 3D medical volume into

- **content**, which captures domain-invariant anatomy
- **style**, which captures acquisition-related appearance

By learning identifiable and transferable content and style factors, MeDUET provides a shared foundation for both controllable generation and robust medical image analysis.

## 📝 TODO
- [x] 📄 Paper released
- [ ] 🧠 Pretraining code
- [ ] 📦 Pretrained model weights
- [ ] 🔧 Downstream code


## 🔎 Overview

In real-world medical imaging, data from different centers often share similar anatomy while exhibiting large appearance variations caused by scanners, protocols, and acquisition conditions. This makes it difficult to directly unify generative modeling and representation learning.

MeDUET addresses this challenge through a disentangled pretraining framework built in the latent space of a pretrained VAE. The framework is designed to support both downstream synthesis and downstream analysis in a unified way.

## 💡 Key Ideas

MeDUET is built on three main components

1. **Demixing for identifiable factor supervision**  
   Token demixing constructs controlled mixtures in latent space and provides explicit supervision for factor separation.

2. **MFTD**  
   Mixed Factor Token Distillation encourages source-faithful factor assignment and reduces factor leakage in mixed regions.

3. **SiQC**  
   Swap-invariance Quadruplet Contrast structures the content and style spaces to improve invariance and discriminability.

Together, these components help MeDUET learn more identifiable content and style representations that can be transferred to both synthesis and analysis tasks.

## ✨ Features

MeDUET aims to bridge two lines of research that are usually developed separately

- **medical image synthesis**
- **medical image analysis**

With disentangled content and style factors, MeDUET enables

- controllable medical image generation
- faster diffusion model convergence
- stronger domain generalization
- better label efficiency for downstream analysis

## 📊 Experimental Scope

According to the current paper, MeDUET is evaluated across **5 datasets**, **4 tasks**, and **2 modalities**. The paper studies both downstream synthesis and downstream analysis settings, including segmentation and classification benchmarks, and shows that the learned content and style factors are useful for controllable diffusion conditioning as well as style-aware transfer. 

## 🚀 Repository Status

**Code coming soon.**

We are currently cleaning and organizing the codebase for public release.

Planned contents include

- pretraining code for MeDUET
- downstream synthesis code
- downstream analysis code
- configs and training scripts
- data preprocessing instructions

## 🙏 Acknowledgement

Our codebase is built upon [MONAI](https://github.com/Project-MONAI/MONAI), an open source framework for medical imaging AI. We sincerely thank the MONAI community for their valuable contributions.

## ✒️ Citation

If you find this project useful, please consider citing our paper.

```bibtex
@article{liu2026meduet,
  title={MeDUET: Disentangled Unified Pretraining for 3D Medical Image Synthesis and Analysis},
  author={Liu, Junkai and Shao, Ling and Zhang, Le},
  journal={arXiv preprint arXiv:2602.17901},
  year={2026}
}
