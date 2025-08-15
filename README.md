# Parallel MCMC
![Image](https://github.com/user-attachments/assets/1b96011c-bdd4-4ea2-8e2e-eaad35a75bcb)

**Parallelizing MCMC across the Sequence Length**\
*David M. Zoltowski\*, Skyler Wu\*, Xavier Gonzalez, Leo Kozachkov, Scott W. Linderman*\
arXiv

This repository contains code for parallelizing MCMC across the chain length via parallel Newton's method.

## Installation
We recommend installing packages in a virtual environment with Python version `>=3.11`. First, run:
```
pip install -U pip
```
Then install JAX via the instructions [here](https://docs.jax.dev/en/latest/installation.html). The algorithms should be 
run on GPU to achieve efficiency gains. However, the code is supported on CPU for testing purposes. We have tested the 
algorithm on JAX version 0.6.2 that can be installed via:
```
pip install -U "jax[cuda12]==0.6.2" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Finally, install the package and its dependencies with:
```
pip install -e .
```

## Examples


## Citation
