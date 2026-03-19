# EKFAC (Eigenvalue-corrected KFAC) Optimizer
#
# Julia/Flux implementation of E-KFAC from:
#   George et al., "Fast Approximate Natural Gradient Descent in a
#   Kronecker Factored Eigenbasis", NeurIPS 2018.
#
# E-KFAC improves upon KFAC by maintaining per-element scaling factors
# in the Kronecker eigenbasis, which better approximates the true Fisher.

"""
    EKFACOptimizer

State container for the EKFAC optimizer.

Extends [`KFACOptimizer`](@ref) with per-element scaling factors `S_l`
that are periodically re-estimated from per-sample gradients projected
into the Kronecker eigenbasis.

# Additional fields (compared to `KFACOptimizer`)
- `TScal`: Frequency (in steps) for recomputing the scaling factors.
- `S_l`: Dict mapping layer index to the scaling matrix in the eigenbasis.
- `cached_A`, `cached_DS`: Cached activations and grad-outputs for scaling updates.
"""
mutable struct EKFACOptimizer
    lr::Float64
    momentum::Float64
    stat_decay::Float64
    damping::Float64
    kl_clip::Float64
    weight_decay::Float64
    TCov::Int
    TScal::Int
    TInv::Int
    batch_averaged::Bool
    steps::Int

    m_aa::Dict{Int, Matrix{Float64}}
    m_gg::Dict{Int, Matrix{Float64}}
    Q_a::Dict{Int, Matrix{Float64}}
    Q_g::Dict{Int, Matrix{Float64}}
    d_a::Dict{Int, Vector{Float64}}
    d_g::Dict{Int, Vector{Float64}}
    S_l::Dict{Int, Matrix{Float64}}

    cached_A::Dict{Int, Any}
    cached_DS::Dict{Int, Any}

    momentum_bufs::Dict{Tuple{Int,Symbol}, Array}
end

"""
    EKFACOptimizer(; lr=0.001, momentum=0.9, stat_decay=0.95, damping=0.001,
                     kl_clip=0.001, weight_decay=0.0, TCov=10, TScal=10,
                     TInv=100, batch_averaged=true)

Construct an EKFAC optimizer.

# Keyword Arguments
All keyword arguments from [`KFACOptimizer`](@ref), plus:
- `TScal`: Frequency for recomputing per-element scaling factors (default: `10`).

# Example
```julia
opt = EKFACOptimizer(lr=0.003, damping=0.03, weight_decay=0.01)
```
"""
function EKFACOptimizer(; lr=0.001, momentum=0.9, stat_decay=0.95, damping=0.001,
                         kl_clip=0.001, weight_decay=0.0, TCov=10, TScal=10,
                         TInv=100, batch_averaged=true)
    lr < 0 && throw(ArgumentError("Invalid learning rate: $lr"))
    momentum < 0 && throw(ArgumentError("Invalid momentum: $momentum"))
    weight_decay < 0 && throw(ArgumentError("Invalid weight_decay: $weight_decay"))

    EKFACOptimizer(
        lr, momentum, stat_decay, damping, kl_clip, weight_decay,
        TCov, TScal, TInv, batch_averaged, 0,
        Dict{Int,Matrix{Float64}}(), Dict{Int,Matrix{Float64}}(),
        Dict{Int,Matrix{Float64}}(), Dict{Int,Matrix{Float64}}(),
        Dict{Int,Vector{Float64}}(), Dict{Int,Vector{Float64}}(),
        Dict{Int,Matrix{Float64}}(),
        Dict{Int,Any}(), Dict{Int,Any}(),
        Dict{Tuple{Int,Symbol},Array}()
    )
end

# ── Eigendecomposition ────────────────────────────────────────────────

function update_inv!(opt::EKFACOptimizer, idx::Int)
    eps = 1e-10
    aa = Symmetric(opt.m_aa[idx])
    gg = Symmetric(opt.m_gg[idx])

    ea = eigen(aa)
    eg = eigen(gg)

    opt.d_a[idx] = max.(ea.values, eps)
    opt.Q_a[idx] = ea.vectors
    opt.d_g[idx] = max.(eg.values, eps)
    opt.Q_g[idx] = eg.vectors

    # Initialize S_l from Kronecker product of eigenvalues
    opt.S_l[idx] = opt.d_g[idx] * opt.d_a[idx]'
end

# ── Scaling factor update ─────────────────────────────────────────────

"""
    update_scale!(opt::EKFACOptimizer, idx, layer)

Re-estimate the per-element scaling matrix `S_l` from cached activations
and gradient outputs.
"""
function update_scale!(opt::EKFACOptimizer, idx::Int, layer)
    haskey(opt.cached_A, idx) || return
    haskey(opt.cached_DS, idx) || return
    haskey(opt.Q_a, idx) || return
    haskey(opt.Q_g, idx) || return

    A = opt.cached_A[idx]
    DS = opt.cached_DS[idx]

    grad_mat = compute_mat_grad(A, DS, layer)  # (out_dim, in_dim, batch)
    if opt.batch_averaged
        grad_mat .*= size(DS, 1)  # un-average
    end

    batch_size = size(grad_mat, 3)

    # Project into eigenbasis: Q_g' * grad * Q_a for each sample
    # grad_mat: (out_dim, in_dim, batch)
    s_l = zeros(Float64, size(opt.Q_g[idx], 2), size(opt.Q_a[idx], 2))
    for b in 1:batch_size
        projected = opt.Q_g[idx]' * grad_mat[:, :, b] * opt.Q_a[idx]
        s_l .+= projected .^ 2
    end
    s_l ./= batch_size

    if !haskey(opt.S_l, idx)
        opt.S_l[idx] = ones(Float64, size(s_l))
    end
    update_running_stat!(s_l, opt.S_l[idx], opt.stat_decay)

    # Clear caches
    delete!(opt.cached_A, idx)
    delete!(opt.cached_DS, idx)
end

# ── Natural gradient with per-element scaling ─────────────────────────

function get_natural_grad(opt::EKFACOptimizer, idx::Int, p_grad_mat::AbstractMatrix,
                          damping::Float64, layer)
    v1 = opt.Q_g[idx]' * p_grad_mat * opt.Q_a[idx]
    v2 = v1 ./ (opt.S_l[idx] .+ damping)
    v  = opt.Q_g[idx] * v2 * opt.Q_a[idx]'

    has_bias = (layer.bias !== false && layer.bias !== nothing)
    if has_bias
        weight_v = v[:, 1:end-1]
        bias_v   = v[:, end]
        if layer isa Conv
            weight_v = reshape(weight_v', size(layer.weight))
        else
            weight_v = reshape(weight_v, size(layer.weight))
        end
        bias_v = reshape(bias_v, size(layer.bias))
        return (weight_v, bias_v)
    else
        if layer isa Conv
            return (reshape(v', size(layer.weight)), nothing)
        else
            return (reshape(v, size(layer.weight)), nothing)
        end
    end
end

# ── Reuse helpers from KFAC module ────────────────────────────────────

function apply_ekfac_update(opt::EKFACOptimizer, layer::Dense, idx, v_w, v_b)
    w = layer.weight .- opt.lr .* _ekfac_momentum!(opt, idx, :weight, v_w, layer.weight)
    if layer.bias !== false && layer.bias !== nothing && v_b !== nothing
        b = layer.bias .- opt.lr .* _ekfac_momentum!(opt, idx, :bias, v_b, layer.bias)
        return Dense(w, b, layer.σ)
    end
    return Dense(w, layer.bias, layer.σ)
end

function apply_ekfac_update(opt::EKFACOptimizer, layer::Conv, idx, v_w, v_b)
    w = layer.weight .- opt.lr .* _ekfac_momentum!(opt, idx, :weight, v_w, layer.weight)
    if layer.bias !== false && layer.bias !== nothing && v_b !== nothing
        b = layer.bias .- opt.lr .* _ekfac_momentum!(opt, idx, :bias, v_b, layer.bias)
        return Conv(layer.σ, w, b, layer.stride, layer.pad, layer.dilation, layer.groups)
    end
    return Conv(layer.σ, w, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups)
end

function _ekfac_momentum!(opt::EKFACOptimizer, idx, which, grad, param)
    d = copy(grad)
    if opt.weight_decay != 0 && opt.steps >= 20 * opt.TCov
        d .+= opt.weight_decay .* param
    end
    if opt.momentum != 0
        key = (idx, which)
        if !haskey(opt.momentum_bufs, key)
            opt.momentum_bufs[key] = copy(d)
        else
            buf = opt.momentum_bufs[key]
            buf .= opt.momentum .* buf .+ d
            opt.momentum_bufs[key] = buf
        end
        d = copy(opt.momentum_bufs[key])
    end
    return d
end

function rebuild_model_ekfac(opt::EKFACOptimizer, model::Chain, kfac_map, grad_tree)
    new_layers = Any[]
    _rebuild_ekfac!(new_layers, opt, model, kfac_map, grad_tree)
    return Chain(new_layers...)
end

function _rebuild_ekfac!(out, opt, model::Chain, kfac_map, grad)
    for (i, layer) in enumerate(model.layers)
        lg = grad === nothing ? nothing : grad.layers[i]
        if layer isa Chain
            inner = Any[]
            _rebuild_ekfac!(inner, opt, layer, kfac_map, lg)
            push!(out, Chain(inner...))
        elseif haskey(kfac_map, objectid(layer))
            idx, v_w, v_b = kfac_map[objectid(layer)]
            push!(out, apply_ekfac_update(opt, layer, idx, v_w, v_b))
        else
            push!(out, layer)
        end
    end
end

# ── Main entry point ──────────────────────────────────────────────────

"""
    ekfac_step!(opt::EKFACOptimizer, model, loss_fn, x, y)

Perform one EKFAC optimisation step, returning `(loss, updated_model)`.

Extends [`kfac_step!`](@ref) with periodic re-estimation of per-element
scaling factors in the Kronecker eigenbasis, which provides a better
approximation to the true Fisher information matrix.

# Arguments
Same as [`kfac_step!`](@ref).

# Returns
`(loss_value, new_model)`.

# Example
```julia
opt = EKFACOptimizer(lr=0.003, damping=0.03, weight_decay=0.01)
model = Chain(Dense(784 => 128, relu), Dense(128 => 10))
loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

for (x, y) in dataloader
    loss, model = ekfac_step!(opt, model, loss_fn, x, y)
end
```
"""
function ekfac_step!(opt::EKFACOptimizer, model, loss_fn, x, y)
    kfac_layers = get_kfac_layers(model)

    # 1. Update covariance statistics
    if opt.steps % opt.TCov == 0
        _update_covs_ekfac!(opt, model, x, kfac_layers)
    end

    # Cache activations for scaling update
    if opt.steps % opt.TScal == 0 && opt.steps > 0
        acts = collect_activations(model, x)
        for (idx, layer) in kfac_layers
            haskey(acts, idx) && (opt.cached_A[idx] = acts[idx])
        end
    end

    # 2. Compute true-loss gradients
    loss_val, grads_tree = Zygote.withgradient(m -> loss_fn(m, x, y), model)
    model_grad = grads_tree[1]

    # Cache gradient outputs for scaling update
    if opt.steps % opt.TScal == 0 && opt.steps > 0
        dy = _initial_grad_output(model, x, y)
        gouts = collect_grad_outputs(model, x, dy)
        for (idx, layer) in kfac_layers
            haskey(gouts, idx) && (opt.cached_DS[idx] = gouts[idx])
        end
    end

    # 3. Eigendecomposition update
    if opt.steps % opt.TInv == 0
        for (idx, _) in kfac_layers
            haskey(opt.m_aa, idx) && haskey(opt.m_gg, idx) && update_inv!(opt, idx)
        end
    end

    # 4. Scaling factor update
    if opt.steps % opt.TScal == 0 && opt.steps > 0
        for (idx, layer) in kfac_layers
            update_scale!(opt, idx, layer)
        end
    end

    # 5. Natural gradient
    layer_grads = Tuple{Any,Any}[]
    updates     = Tuple{Any,Any}[]
    for (idx, layer) in kfac_layers
        r = extract_layer_grads(model_grad, model, layer)
        w_g, b_g = r === nothing ? (zeros(size(layer.weight)),
            layer.bias !== false && layer.bias !== nothing ? zeros(size(layer.bias)) : nothing) : r
        push!(layer_grads, (w_g, b_g))

        if haskey(opt.Q_a, idx) && haskey(opt.S_l, idx)
            p = get_matrix_form_grad(w_g, b_g, layer)
            push!(updates, get_natural_grad(opt, idx, p, opt.damping, layer))
        else
            push!(updates, (w_g, b_g))
        end
    end

    # 6. KL clip
    clipped, _ = kl_clip(updates, layer_grads, opt.lr, opt.kl_clip)

    # 7. Rebuild model
    kfac_map = Dict{UInt,Tuple{Int,Any,Any}}()
    for (i, (idx, layer)) in enumerate(kfac_layers)
        kfac_map[objectid(layer)] = (idx, clipped[i][1], clipped[i][2])
    end
    new_model = rebuild_model_ekfac(opt, model, kfac_map, model_grad)

    opt.steps += 1
    return loss_val, new_model
end

function _update_covs_ekfac!(opt::EKFACOptimizer, model, x, kfac_layers)
    acts = collect_activations(model, x)
    output = model(x)

    probs = softmax(output; dims=1)
    batch_size = size(output, 2)
    sampled_y = zeros(eltype(output), size(output))
    for j in 1:batch_size
        p = probs[:, j]
        cp = cumsum(p)
        r = rand(eltype(p))
        idx = something(findfirst(>=(r), cp), length(p))
        sampled_y[idx, j] = one(eltype(output))
    end

    dy = _initial_grad_output(model, x, sampled_y)
    gouts = collect_grad_outputs(model, x, dy)

    for (idx, layer) in kfac_layers
        if haskey(acts, idx)
            aa = compute_cov_a(acts[idx], layer)
            if !haskey(opt.m_aa, idx)
                opt.m_aa[idx] = Matrix{Float64}(I, size(aa)...)
            end
            update_running_stat!(Float64.(aa), opt.m_aa[idx], opt.stat_decay)
        end

        if haskey(gouts, idx)
            gg = compute_cov_g(gouts[idx], layer; batch_averaged=opt.batch_averaged)
            if !haskey(opt.m_gg, idx)
                opt.m_gg[idx] = Matrix{Float64}(I, size(gg)...)
            end
            update_running_stat!(Float64.(gg), opt.m_gg[idx], opt.stat_decay)
        end
    end
end
