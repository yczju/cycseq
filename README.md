
# CycSeq: Leveraging Cyclic Data Generation for Accurate Perturbation Prediction in Single-Cell RNA-Seq
**Accepted at IJCAI 2025**

---

## Abstract

Understanding and predicting the effects of cellular perturbations using single-cell technology is a critical problem and remains challenging in biotechnology. 
In this work, we leverage the concept of cyclic data generation and the recent advancements in neural architectures to introduce CycSeq, a deep learning framework capable of predicting single-cell responses under specified perturbation conditions across multiple cell lines and generating corresponding single-cell expression profiles. 
CycSeq addresses the challenge of learning heterogeneous perturbation responses from unpaired single-cell level gene expression observation by generating pseudo-pairs through cyclic data generation. Experimental results show that CycSeq could outperform current methods in perturbation prediction tasks evaluated by computational metrics($R^2$, MAE, etc.). CycSeq also tackles the issue of applying deep learning models to long-tail cell lines by utilizing a unified architecture that integrates information from multiple cell lines, enabling robust predictions even for cell lines with limited training data.

---

## Introduction

CycSeq is a novel deep learning framework based on cycle-consistent generative adversarial networks (CycleGAN) for the integration and translation of single-cell perturbation datasets. This repository provides the official implementation of the methods described in our IJCAI 2025 accepted paper.

---

## Paper

**Title:** Cycle-Consistent Generative Adversarial Networks for Single-Cell Perturbation Data Integration   
**Conference:** International Joint Conference on Artificial Intelligence (IJCAI 2025)  
**Status:** Accepted  
**Paper Link:** (to be updated) 

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yczju/cycseq.git
cd cycseq
```

### 2. Create and activate a Python environment

We recommend using [conda](https://docs.conda.io/) or [venv](https://docs.python.org/3/library/venv.html) for environment management.

```bash
conda create -n cycseq python=3.9
conda activate cycseq
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- torch >= 1.10
- numpy
- pandas
- tqdm
- matplotlib
- scanpy
- scipy
- tensorboard
- pycombat, inmoose

For GPU support, please ensure you have installed the correct CUDA version and compatible PyTorch.

---

## Data Preparation

Please refer to the `script/data_preparation.sh` for detailed instructions on preparing training data.  
The project supports both `.csv` and `.npy` formats for single-cell perturbation expression matrices.

---

## Usage

### Training

```bash
python code/CycleGAN4Seq/ijcai.py --num_res_blocks 2 --use_attention True --use_embedding False --use_batch_norm False --use_dropout True --name cycseq
```

### Main Arguments

- `--num_res_blocks`: Number of residual blocks in encoder/decoder.
- `--use_attention`: Whether to use self-attention.
- `--use_embedding`: Whether to use embedding layers.
- `--use_batch_norm`: Whether to use batch normalization.
- `--use_dropout`: Whether to use dropout.
- `--name`: Model name for output files.

---

## Results

- The model has been evaluated on multiple public single-cell perturbation datasets. Please refer to the paper and the `output_model/` directory for detailed results.
- Training logs and loss curves can be visualized using TensorBoard (`logs/` directory).

---

## Citation

If you find this repository useful for your research, please cite our paper:

(to be updated) 

---

## Contact

For questions or collaboration, please contact:  
- yichengliu@zju.edu.cn

---

## Acknowledgements

This project is inspired by [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and related single-cell analysis tools.