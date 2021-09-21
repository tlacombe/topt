# TOPT : a Topological OPtimization Toolbox

Authors: [Mathieu Carrière](https://mathieucarriere.github.io/website/) and [Théo Lacombe](https://tlacombe.github.io).

This repository is a work in progress. Comments and feedback appreciated!

# Quick start

A minimal working example is available as a tutorial based on [this paper](https://arxiv.org/abs/2109.00530), see `tutorial_0*.ipynb`.  
This tutorial provides an introduction to generalized gradient descent for 
Topological Data Analysis and showcase its use in the (very simple) case 
of _total persistence_ minimization. 

## Dependencies

This repository was tested on Ubuntu 20.04 and relies on the following libraries:
- `tensorflow 2.2` for automatic differentiation.
- `gudhi` for persistent homology related computations. It was tested with `gudhi 3.4` and `gudhi 3.5` 
(which has not been released yet as of 09/20/2021.)
- `cvxpy` (tested with ` 1.13`, other versions should work as well) to compute the element of minimal norm on a convex hull of sampled gradients. 
- `numpy` (tested with `1.20`, other versions should work as well). 

In addition, to run the notebooks provided in the `./tutorials/` folder, one needs (along with a 
jupyter notebook installation and the aforementioned packages):
- `matplotlib` (tested with `3.4`, other versions should work as well).

# Repository organization

**Note:** the organization is subject to evolution depending on the future additions that will be made to this repository.

The `topt.py` file contains the most important methods for this repository.
In particular, it defines the `TopoMeanModel` class, a natural (tensorflow-like) that compute
(in an autodiff-compatible way) a loss of the form:

$$ x \mapsto \sum_{i=1}^L \mathrm{dist}(\mathrm{Dgm}(x,K), d_i)^\mathrm{order} $$

where the $(d_i)_i$s represent $L$ "target" persistence diagrams, $\mathrm{Dgm}(x)$ is the (ordinary or extended) diagram of a 
current filter function $x$ defined on a given simplicial complex $K$. 
Note that if $L=1$, we retrieve the common problem of minimizing $x \mapsto \mathrm{dist}(\mathrm{Dgm}(x, K), d)$ for 
some target diagram $d$. 

# Related content:
Non-exhaustive, 
feel free to mention code ressources related to optimization with topological descriptors.

- [Mathieu Carriere's difftda notebook](https://github.com/MathieuCarriere/difftda), related to [this paper](http://proceedings.mlr.press/v139/carriere21a/carriere21a.pdf).
- [The code repository](https://github.com/bruel-gabrielsson/TopologyLayer) for [A Topology Layer for machine learning](https://arxiv.org/abs/1905.12200) by Rickard Brüel Gabrielsson and co-authors.
- [The code repository](https://github.com/jisuk1/pllay) for [PLLay: Efficient Topological Layer based on Persistence Landscape](https://arxiv.org/abs/2002.02778) by Kwangho Kim and co-authors.
- TBC

**Note:** no license yet, all rights reserved (will be updated later).