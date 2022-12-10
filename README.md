# NeRF: Neural Radiance Fields for Novel View Synthesis


## A NeRF re-implementation using PyTorch - Machine Learning for 3D Data, CSE291 22Fall UCSD

---

written by [Tianyang Liu](https://leolty.github.io/)


### Quick Start

First, create a conda enviroment named nerf and install corresponding dependencies with the following bash code.

```bash
conda env create -f environment.yml
conda activate nerf
```

Then, open `workflow.ipynb`, just follow the steps.


### NeRF Pipeline

1. Train a single NeRF with data argumentation
2. Drop data argumentation and continue to train the single NeRF.
3. Hard copy the single NeRF to two (one for coarse, one for fine), and use hierarchical samping to fine the outpit.
4. Finetune NeRF system.
   
### Rendering Effect

![rf](src/output.gif)

### Reference

The implementation is based on some Github repos ([nerf](https://github.com/bmild/nerf),  [NeRF](https://github.com/Jiayi-Pan/NeRF), [nerf_pl](https://github.com/kwea123/nerf_pl) and [nerf-pytorch](https://github.com/krrish94/nerf-pytorch)), the example of nerf on [keras](https://keras.io/examples/vision/nerf/), some Google Colabs ([NeRF.ipynb](https://colab.research.google.com/drive/1_51bC5d6m7EFU6U_kkUL2lMYehJqc01R?usp=sharing) and [NeRF_From_Nothing](https://colab.research.google.com/github/aviralksingh/NeRF/blob/main/NeRF_From_Nothing.ipynb#scrollTo=9UXBPUv2407W)), and the article on [Medium](https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666).

Of course, the present implementation also draws heavily on [NeRF original paper](https://arxiv.org/abs/2003.08934).


