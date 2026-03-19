# KFAC.jl

[![Build Status](https://github.com/oliverdaisey/KFAC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/oliverdaisey/KFAC.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia/Flux implementation of [K-FAC](https://arxiv.org/abs/1503.05671) (Kronecker-Factored Approximate Curvature) and [E-KFAC](https://arxiv.org/abs/1806.03884) (Eigenvalue-corrected KFAC) optimisers for training neural networks.

Ported from [KFAC-Pytorch](https://github.com/alecwangcq/KFAC-Pytorch).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/oliverdaisey/KFAC.jl")
```

## Quick Start

```julia
using KFAC, Flux

# Define model and data
model = Chain(Dense(784 => 128, relu), Dense(128 => 10))
x = randn(Float32, 784, 64)
y = Float32.(Flux.onehotbatch(rand(1:10, 64), 1:10))

# Create optimizer
opt = KFACOptimizer(lr=0.01, damping=0.03, weight_decay=0.003)

# Training loop
loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)
for epoch in 1:100
    loss, model = kfac_step!(opt, model, loss_fn, x, y)
    println("Epoch $epoch: loss = $loss")
end
```

## Optimizers

### `KFACOptimizer`

Standard K-FAC optimizer using Kronecker-factored approximation to the Fisher information matrix.

```julia
opt = KFACOptimizer(
    lr = 0.001,           # learning rate
    momentum = 0.9,       # SGD momentum
    stat_decay = 0.95,    # EMA decay for covariance statistics
    damping = 0.001,      # Tikhonov damping
    kl_clip = 0.001,      # KL-divergence trust region
    weight_decay = 0.0,   # L2 regularization
    TCov = 10,            # covariance recomputation frequency (steps)
    TInv = 100,           # eigendecomposition recomputation frequency (steps)
    batch_averaged = true  # whether loss is batch-averaged
)
```

### `EKFACOptimizer`

Eigenvalue-corrected K-FAC, which maintains per-element scaling factors in the Kronecker eigenbasis for a more accurate Fisher approximation.

```julia
opt = EKFACOptimizer(
    lr = 0.001,
    momentum = 0.9,
    stat_decay = 0.95,
    damping = 0.001,
    kl_clip = 0.001,
    weight_decay = 0.0,
    TCov = 10,
    TScal = 10,           # scaling factor recomputation frequency
    TInv = 100,
    batch_averaged = true
)
```

## Supported Layers

- `Flux.Dense` (with or without bias)
- `Flux.Conv` (with or without bias)

Other layers in the model (e.g., `BatchNorm`, activation functions) are updated using standard gradient descent.

## API

### Training Step Functions

- `kfac_step!(opt, model, loss_fn, x, y)` Perform one KFAC optimization step. Returns `(loss, updated_model)`.
- `ekfac_step!(opt, model, loss_fn, x, y)` Perform one EKFAC optimization step. Returns `(loss, updated_model)`.

### Utility Functions

- `compute_cov_a(a, layer)` Compute the input activation covariance (Kronecker factor A).
- `compute_cov_g(g, layer; batch_averaged)` Compute the gradient output covariance (Kronecker factor G).
- `update_running_stat!(new, running, decay)` Exponential moving average update.
- `extract_patches(x, kernel_size, stride, padding)` Extract convolution patches.
- `compute_mat_grad(input, grad_output, layer)` Per-sample gradient matrices (for EKFAC).

## Examples

See the [`examples/`](examples/) directory:

- `simple_classification.jl` — Train a small MLP on a synthetic classification task with KFAC and EKFAC.

## Algorithm Overview

K-FAC approximates the Fisher information matrix **F** as a block-diagonal matrix, where each block (corresponding to one layer) is further approximated as a Kronecker product:

$$
F_l \approx A_l \otimes G_l
$$

where $$A_l$$ is the covariance of the layer's input activations and $$G_l$$ is the covariance of the gradient of the loss w.r.t. the layer's output. The natural gradient is then:

$$
\Delta \theta_l = \left(A_l \otimes G_l + \lambda I\right)^{-1} \nabla \theta_l
$$

which can be computed efficiently using the eigendecompositions of $$A_l$$ and $$G_l$$.

E-KFAC improves on this by maintaining per-element scaling factors in the Kronecker eigenbasis, providing a better diagonal correction to the Kronecker approximation.

## References

```bibtex
@inproceedings{martens2015optimizing,
  title={Optimizing neural networks with kronecker-factored approximate curvature},
  author={Martens, James and Grosse, Roger},
  booktitle={International conference on machine learning},
  pages={2408--2417},
  year={2015}
}

@inproceedings{grosse2016kronecker,
  title={A kronecker-factored approximate fisher matrix for convolution layers},
  author={Grosse, Roger and Martens, James},
  booktitle={International Conference on Machine Learning},
  pages={573--582},
  year={2016}
}

@inproceedings{george2018fast,
  title={Fast Approximate Natural Gradient Descent in a Kronecker Factored Eigenbasis},
  author={George, Thomas and Laurent, C{\'e}sar and Bouthillier, Xavier and Ballas, Nicolas and Vincent, Pascal},
  booktitle={Advances in Neural Information Processing Systems},
  pages={9550--9560},
  year={2018}
}
```
