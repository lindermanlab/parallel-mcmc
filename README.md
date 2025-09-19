# Parallel MCMC
![Image](https://github.com/lindermanlab/parallel-mcmc/blob/01192c587cb74cf2c3733402d0a8e8adab637714/static/rosenbrock.gif)

**Parallelizing MCMC across the Sequence Length**\
*David M. Zoltowski\*, Skyler Wu\*, Xavier Gonzalez, Leo Kozachkov, Scott W. Linderman*\
Paper: https://arxiv.org/abs/2508.18413
To appear in NeurIPS 2025

This repository contains code for parallelizing MCMC across the chain length via parallel Newton's method. 

**Colab Demo:** An interactive Colab demo is available [here!](https://colab.research.google.com/drive/1TLd8nOw5VBK8olQLiuSFcY7_yYVYluqd?usp=sharing) 

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

## Repository Structure 
The structure of the primary source code and examples is:
```
src/                    Source code for DEER algorithms and samplers.
├── samplers.py             Defines parallel MALA and HMC samplers.
├── qdeer.py                Optimized stochastic quasi DEER implementation.
├── deer.py                 DEER implementation.
├── elk.py                  Quasi ELK implementation.
├── qdeer_leapfrog.py       Block quasi-DEER for parallel leapfrog. 
├── windowed_qdeer.py       Quasi DEER implementation with windowing.
examples/               Example scripts
```

## Examples

We include four example scripts that we describe below. Additionally, we provide an interactive Google Colab example [here](https://colab.research.google.com/drive/1TLd8nOw5VBK8olQLiuSFcY7_yYVYluqd?usp=sharing) 
demonstrating wall-clock speedups when the Colab is run on an A100 instance. 
- **`examples/run_mala_german_credit.py`** - Runs parallel MALA using stochastic quasi DEER targeting a logistic regression model of the German Credit dataset. 
  ```
  python examples/run_mala_german_credit.py
  ```
- **`examples/run_hmc_rosenbrock.py`** - Runs parallel HMC using DEER with damping (ELK) targeting the Rosenbrock distribution.
  ```
  python examples/run_hmc_rosenbrock.py
  ```
- **`examples/run_gibbs_eight_schools.py`** - Runs a parallel Gibbs sampler targeting the posterior of a hierarchical linear model. 
  ```
  python examples/run_gibbs_eight_schools.py
  ```
- **`examples/run_mala_mog.py`** - Runs parallel MALA using quasi DEER targeting a 2D multimodal mixture of Gaussians.
  ```
  python examples/run_mala_mog.py
  ```

## Citation

```
@article{zoltowski2025parallelmcmc,
      title={Parallelizing MCMC Across the Sequence Length}, 
      author={David M. Zoltowski and Skyler Wu and Xavier Gonzalez and Leo Kozachkov and Scott W. Linderman},
      year={2025},
      eprint={2508.18413},
      archivePrefix={arXiv},
      primaryClass={stat.CO},
}
```
