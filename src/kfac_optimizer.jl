# KFAC (Kronecker-Factored Approximate Curvature) Optimizer
#
# Julia/Flux implementation of K-FAC from:
#   Martens & Grosse, "Optimizing Neural Networks with Kronecker-Factored
#   Approximate Curvature", ICML 2015.

"""
    KFACOptimizer

State container for the KFAC optimizer.

# Fields
- `lr`, `momentum`, `stat_decay`, `damping`, `kl_clip`, `weight_decay`: Hyperparameters.
- `TCov`: Recompute covariance statistics every `TCov` steps.
- `TInv`: Recompute eigendecomposition every `TInv` steps.
- `batch_averaged`: Whether gradients are batch-averaged.
- `steps`: Step counter.
- `m_aa`, `m_gg`: Running input/gradient covariance matrices keyed by layer index.
- `Q_a`, `Q_g`: Eigenvectors of A and G.
- `d_a`, `d_g`: Eigenvalues of A and G.
- `momentum_bufs`: Momentum buffers keyed by a `(layer_index, :weight/:bias)` tuple.
"""
mutable struct KFACOptimizer
    lr::Float64
    momentum::Float64
    stat_decay::Float64
    damping::Float64
    kl_clip::Float64
    weight_decay::Float64
    TCov::Int
    TInv::Int
    batch_averaged::Bool
    steps::Int

    m_aa::Dict{Int, Matrix{Float64}}
    m_gg::Dict{Int, Matrix{Float64}}
    Q_a::Dict{Int, Matrix{Float64}}
    Q_g::Dict{Int, Matrix{Float64}}
    d_a::Dict{Int, Vector{Float64}}
    d_g::Dict{Int, Vector{Float64}}

    momentum_bufs::Dict{Tuple{Int,Symbol}, Array}
end

"""
    KFACOptimizer(; lr=0.001, momentum=0.9, stat_decay=0.95, damping=0.001,
                    kl_clip=0.001, weight_decay=0.0, TCov=10, TInv=100,
                    batch_averaged=true)

Construct a KFAC optimizer.

# Keyword Arguments
- `lr`: Learning rate (default: `0.001`).
- `momentum`: Momentum coefficient (default: `0.9`).
- `stat_decay`: Decay rate for running covariance statistics (default: `0.95`).
- `damping`: Tikhonov damping for the Fisher inverse (default: `0.001`).
- `kl_clip`: KL-divergence clipping threshold (default: `0.001`).
- `weight_decay`: L2 regularization coefficient (default: `0.0`).
- `TCov`: Covariance recomputation frequency in steps (default: `10`).
- `TInv`: Eigendecomposition recomputation frequency in steps (default: `100`).
- `batch_averaged`: Whether the loss is batch-averaged (default: `true`).

# Example
```julia
opt = KFACOptimizer(lr=0.01, damping=0.03, weight_decay=0.003)
```
"""
function KFACOptimizer(; lr=0.001, momentum=0.9, stat_decay=0.95, damping=0.001,
                        kl_clip=0.001, weight_decay=0.0, TCov=10, TInv=100,
                        batch_averaged=true)
    lr < 0 && throw(ArgumentError("Invalid learning rate: $lr"))
    momentum < 0 && throw(ArgumentError("Invalid momentum: $momentum"))
    weight_decay < 0 && throw(ArgumentError("Invalid weight_decay: $weight_decay"))

    KFACOptimizer(
        lr, momentum, stat_decay, damping, kl_clip, weight_decay,
        TCov, TInv, batch_averaged, 0,
        Dict{Int,Matrix{Float64}}(), Dict{Int,Matrix{Float64}}(),
        Dict{Int,Matrix{Float64}}(), Dict{Int,Matrix{Float64}}(),
        Dict{Int,Vector{Float64}}(), Dict{Int,Vector{Float64}}(),
        Dict{Tuple{Int,Symbol},Array}()
    )
end

# ── Internal helpers ───────────────────────────────────────────────────

"""
    get_kfac_layers(model) -> Vector{Tuple{Int, Any}}

Return `(index, layer)` pairs for every `Dense` or `Conv` layer in the model.
"""
function get_kfac_layers(model)
    layers = Tuple{Int,Any}[]
    idx = Ref(0)
    _collect_kfac_layers!(layers, model, idx)
    return layers
end

function _collect_kfac_layers!(out, model::Chain, idx)
    for l in model.layers
        _collect_kfac_layers!(out, l, idx)
    end
end

function _collect_kfac_layers!(out, layer, idx)
    if layer isa Dense || layer isa Conv
        idx[] += 1
        push!(out, (idx[], layer))
    elseif hasproperty(layer, :layers)
        for l in layer.layers
            _collect_kfac_layers!(out, l, idx)
        end
    end
end

"""
    collect_activations(model, x) -> Dict{Int, Any}

Run a forward pass layer-by-layer, recording the input to each KFAC layer.
Dense activations are stored as `(batch, features)` matrices;
Conv activations are stored as `(W, H, C, N)` arrays.
"""
function collect_activations(model, x)
    acts = Dict{Int,Any}()
    _collect_acts!(acts, model, x, Ref(0))
    return acts
end

# returns the output of this sub-model
function _collect_acts!(acts, model::Chain, x, idx)
    current = x
    for l in model.layers
        current = _collect_acts!(acts, l, current, idx)
    end
    return current
end

function _collect_acts!(acts, layer, x, idx)
    if layer isa Dense
        idx[] += 1
        # Flux Dense: weight is (out, in), input x is (in, batch)
        acts[idx[]] = Matrix(x')  # store as (batch, in)
    elseif layer isa Conv
        idx[] += 1
        acts[idx[]] = copy(x)    # (W, H, C, N)
    elseif hasproperty(layer, :layers)
        return _collect_acts!(acts, Chain(layer.layers...), x, idx)
    end
    return layer(x)
end

"""
    collect_grad_outputs(model, x, dy) -> Dict{Int, Any}

Manually propagate gradients backwards through the model, recording the
gradient of the loss w.r.t. each KFAC layer's *output* (pre-activation).

For Dense layers the result has shape `(batch, out_features)`;
for Conv layers it has shape `(W, H, C_out, N)`.
"""
function collect_grad_outputs(model, x, dy)
    gouts = Dict{Int,Any}()
    layers = _flatten_model(model)
    # Forward pass to collect all intermediate values
    intermediates = Any[x]
    current = x
    for layer in layers
        current = layer(current)
        push!(intermediates, current)
    end

    # Backward pass
    grad_out = dy
    kfac_idx = count(l -> l isa Dense || l isa Conv, layers)
    for i in length(layers):-1:1
        layer = layers[i]
        inp = intermediates[i]

        if layer isa Dense || layer isa Conv
            # The gradient w.r.t. this layer's output (before activation)
            # We need the gradient *after* activation has been backpropagated
            # For a layer with activation σ: output = σ(W*x + b)
            # grad_out is dL/d(output), we need dL/d(W*x+b) = grad_out ⊙ σ'(W*x+b)
            # However, in Flux, Dense applies σ inside, so we need to handle this.
            # Let's compute the pre-activation output and the activation derivative
            if layer isa Dense
                pre_act = layer.weight * inp
                if layer.bias !== false && layer.bias !== nothing
                    pre_act = pre_act .+ layer.bias
                end
                if layer.σ === identity
                    grad_pre = grad_out
                else
                    # dL/d(pre_act) = dL/d(output) * σ'(pre_act)
                    act_out = layer.σ.(pre_act)
                    _, σ_pullback = Zygote.pullback(z -> layer.σ.(z), pre_act)
                    grad_pre = σ_pullback(grad_out)[1]
                end
                # Store as (batch, out_features)
                gouts[kfac_idx] = Matrix(grad_pre')
            elseif layer isa Conv
                # For Conv, σ is applied element-wise on the output
                pre_act = Flux.conv(inp, layer.weight; stride=layer.stride, pad=layer.pad,
                                    dilation=layer.dilation, groups=layer.groups)
                if layer.bias !== false && layer.bias !== nothing
                    pre_act = pre_act .+ reshape(layer.bias, 1, 1, :, 1)
                end
                if layer.σ === identity
                    grad_pre = grad_out
                else
                    _, σ_pullback = Zygote.pullback(z -> layer.σ.(z), pre_act)
                    grad_pre = σ_pullback(grad_out)[1]
                end
                gouts[kfac_idx] = grad_pre
            end
            kfac_idx -= 1
        end

        # Propagate gradient to previous layer
        _, pb = Zygote.pullback(layer, inp)
        grad_out = pb(grad_out)[1]
    end

    return gouts
end

function _flatten_model(model)
    layers = Any[]
    _flatten!(layers, model)
    return layers
end

function _flatten!(out, model::Chain)
    for l in model.layers
        _flatten!(out, l)
    end
end

function _flatten!(out, layer)
    if hasproperty(layer, :layers)
        for l in layer.layers
            _flatten!(out, l)
        end
    else
        push!(out, layer)
    end
end

"""
    update_inv!(opt::KFACOptimizer, idx)

Eigendecompose the running covariance matrices for layer `idx`.
Small eigenvalues are clamped to `eps` for numerical stability.
"""
function update_inv!(opt::KFACOptimizer, idx::Int)
    eps = 1e-10
    aa = Symmetric(opt.m_aa[idx])
    gg = Symmetric(opt.m_gg[idx])

    ea = eigen(aa)
    eg = eigen(gg)

    opt.d_a[idx] = max.(ea.values, eps)
    opt.Q_a[idx] = ea.vectors
    opt.d_g[idx] = max.(eg.values, eps)
    opt.Q_g[idx] = eg.vectors
end

"""
    get_matrix_form_grad(w_grad, b_grad, layer)

Reshape a layer's weight (and bias) gradient into the standard
`(out_dim, in_dim [+1])` matrix used by the KFAC update rule.
"""
function get_matrix_form_grad(w_grad, b_grad, layer::Dense)
    p = copy(w_grad)  # already (out, in)
    if layer.bias !== false && layer.bias !== nothing && b_grad !== nothing
        p = hcat(p, reshape(b_grad, :, 1))
    end
    return p
end

function get_matrix_form_grad(w_grad, b_grad, layer::Conv)
    # Flux weight: (kw, kh, in_c, out_c) → reshape to (out_c, kw*kh*in_c)
    n_filters = size(layer.weight, 4)
    p = reshape(w_grad, :, n_filters)'  # (out_c, kw*kh*in_c)
    if layer.bias !== false && layer.bias !== nothing && b_grad !== nothing
        p = hcat(p, reshape(b_grad, :, 1))
    end
    return p
end

"""
    get_natural_grad(opt::KFACOptimizer, idx, p_grad_mat, damping, layer)

Compute the natural gradient for one layer:

    v = Q_g * diag(1 ./ (d_g ⊗ d_a .+ λ)) * Q_g' * grad * Q_a * Q_a'

Returns `(weight_update, bias_update_or_nothing)` with shapes matching the layer.
"""
function get_natural_grad(opt::KFACOptimizer, idx::Int, p_grad_mat::AbstractMatrix,
                          damping::Float64, layer)
    v1 = opt.Q_g[idx]' * p_grad_mat * opt.Q_a[idx]
    v2 = v1 ./ (opt.d_g[idx] * opt.d_a[idx]' .+ damping)
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

"""
    kl_clip(updates, grads, lr, kl_clip_val)

Scale the natural-gradient updates so the implied KL divergence stays
below `kl_clip_val`.  Returns `(clipped_updates, nu)`.
"""
function kl_clip(updates, grads, lr, kl_clip_val)
    vg_sum = 0.0
    for i in eachindex(updates)
        v_w, v_b = updates[i]
        g_w, g_b = grads[i]
        vg_sum += sum(v_w .* g_w) * lr^2
        if v_b !== nothing && g_b !== nothing
            vg_sum += sum(v_b .* g_b) * lr^2
        end
    end
    nu = min(1.0, sqrt(kl_clip_val / max(abs(vg_sum), 1e-20)))
    clipped = [(v_w .* nu, v_b === nothing ? nothing : v_b .* nu) for (v_w, v_b) in updates]
    return clipped, nu
end

# ── Gradient extraction from Zygote trees ─────────────────────────────

function extract_layer_grads(grad_tree, model, target_layer)
    return _extract_recursive(grad_tree, model, target_layer)
end

function _extract_recursive(grad, model::Chain, target)
    grad === nothing && return nothing
    for (i, layer) in enumerate(model.layers)
        lg = grad.layers[i]
        if layer === target
            lg === nothing && return (zeros(size(target.weight)),
                target.bias !== false && target.bias !== nothing ? zeros(size(target.bias)) : nothing)
            w = lg.weight === nothing ? zeros(size(target.weight)) : lg.weight
            b = if target.bias !== false && target.bias !== nothing
                lg.bias === nothing ? zeros(size(target.bias)) : lg.bias
            else
                nothing
            end
            return (w, b)
        end
        if layer isa Chain || hasproperty(layer, :layers)
            r = _extract_recursive(lg, layer isa Chain ? layer : Chain(layer.layers...), target)
            r !== nothing && return r
        end
    end
    return nothing
end

# ── Parameter update (pure-functional, returns new model) ─────────────

function apply_kfac_update(opt::KFACOptimizer, layer::Dense, idx, v_w, v_b)
    w = layer.weight .- opt.lr .* _momentum_step!(opt, idx, :weight, v_w, layer.weight)
    if layer.bias !== false && layer.bias !== nothing && v_b !== nothing
        b = layer.bias .- opt.lr .* _momentum_step!(opt, idx, :bias, v_b, layer.bias)
        return Dense(w, b, layer.σ)
    end
    return Dense(w, layer.bias, layer.σ)
end

function apply_kfac_update(opt::KFACOptimizer, layer::Conv, idx, v_w, v_b)
    w = layer.weight .- opt.lr .* _momentum_step!(opt, idx, :weight, v_w, layer.weight)
    if layer.bias !== false && layer.bias !== nothing && v_b !== nothing
        b = layer.bias .- opt.lr .* _momentum_step!(opt, idx, :bias, v_b, layer.bias)
        return Conv(layer.σ, w, b, layer.stride, layer.pad, layer.dilation, layer.groups)
    end
    return Conv(layer.σ, w, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups)
end

function _momentum_step!(opt, idx, which, grad, param)
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

function rebuild_model(opt, model::Chain, kfac_map, grad_tree)
    new_layers = Any[]
    _rebuild!(new_layers, opt, model, kfac_map, grad_tree)
    return Chain(new_layers...)
end

function _rebuild!(out, opt, model::Chain, kfac_map, grad)
    for (i, layer) in enumerate(model.layers)
        lg = grad === nothing ? nothing : grad.layers[i]
        if layer isa Chain
            inner = Any[]
            _rebuild!(inner, opt, layer, kfac_map, lg)
            push!(out, Chain(inner...))
        elseif haskey(kfac_map, objectid(layer))
            idx, v_w, v_b = kfac_map[objectid(layer)]
            push!(out, apply_kfac_update(opt, layer, idx, v_w, v_b))
        elseif layer isa Dense || layer isa Conv
            # Fallback: standard SGD for KFAC layers without computed updates
            r = _extract_recursive(grad, Chain(layer), layer)
            if r !== nothing
                w_g, b_g = r
                push!(out, apply_kfac_update(opt, layer, -objectid(layer) |> Int, w_g, b_g))
            else
                push!(out, layer)
            end
        else
            push!(out, layer)
        end
    end
end

# ── Main entry point ──────────────────────────────────────────────────

"""
    kfac_step!(opt::KFACOptimizer, model, loss_fn, x, y)

Perform one KFAC optimisation step, returning `(loss, updated_model)`.

The function:
1. Optionally recomputes Kronecker-factor covariance statistics (every `TCov` steps)
   using the *sampled* Fisher (samples from the model's predictive distribution).
2. Computes gradients of the true loss.
3. Optionally updates the eigendecomposition of the covariance factors (every `TInv` steps).
4. Transforms gradients into natural gradients via the Kronecker-factored Fisher inverse.
5. Clips updates using a KL-divergence trust region.
6. Applies momentum and weight decay, then updates all parameters.

# Arguments
- `opt`: [`KFACOptimizer`](@ref).
- `model`: Any Flux model whose trainable layers are `Dense` and/or `Conv`.
- `loss_fn`: `(model, x, y) -> scalar` loss function.
- `x`: Input batch.
- `y`: Target batch.

# Returns
`(loss_value, new_model)` — the scalar loss **before** the parameter update
and the model with updated weights.

# Example
```julia
opt = KFACOptimizer(lr=0.01, damping=0.03)
model = Chain(Dense(784 => 128, relu), Dense(128 => 10))
loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

for (x, y) in dataloader
    loss, model = kfac_step!(opt, model, loss_fn, x, y)
end
```
"""
function kfac_step!(opt::KFACOptimizer, model, loss_fn, x, y)
    kfac_layers = get_kfac_layers(model)

    # 1. Update covariance statistics (sampled Fisher)
    if opt.steps % opt.TCov == 0
        _update_covs!(opt, model, x, kfac_layers)
    end

    # 2. Compute true-loss gradients
    loss_val, grads_tree = Zygote.withgradient(m -> loss_fn(m, x, y), model)
    model_grad = grads_tree[1]

    # 3. Eigendecomposition update
    if opt.steps % opt.TInv == 0
        for (idx, _) in kfac_layers
            haskey(opt.m_aa, idx) && haskey(opt.m_gg, idx) && update_inv!(opt, idx)
        end
    end

    # 4. Natural gradient per layer
    layer_grads = Tuple{Any,Any}[]
    updates     = Tuple{Any,Any}[]
    for (idx, layer) in kfac_layers
        r = extract_layer_grads(model_grad, model, layer)
        w_g, b_g = r === nothing ? (zeros(size(layer.weight)),
            layer.bias !== false && layer.bias !== nothing ? zeros(size(layer.bias)) : nothing) : r
        push!(layer_grads, (w_g, b_g))

        if haskey(opt.Q_a, idx) && haskey(opt.Q_g, idx)
            p = get_matrix_form_grad(w_g, b_g, layer)
            push!(updates, get_natural_grad(opt, idx, p, opt.damping, layer))
        else
            push!(updates, (w_g, b_g))
        end
    end

    # 5. KL clip
    clipped, _ = kl_clip(updates, layer_grads, opt.lr, opt.kl_clip)

    # 6. Rebuild model with updated weights
    kfac_map = Dict{UInt,Tuple{Int,Any,Any}}()
    for (i, (idx, layer)) in enumerate(kfac_layers)
        kfac_map[objectid(layer)] = (idx, clipped[i][1], clipped[i][2])
    end
    new_model = rebuild_model(opt, model, kfac_map, model_grad)

    opt.steps += 1
    return loss_val, new_model
end

# ── Covariance update (sampled Fisher) ────────────────────────────────

function _update_covs!(opt::KFACOptimizer, model, x, kfac_layers)
    # Collect input activations
    acts = collect_activations(model, x)

    # Forward pass to get logits
    output = model(x)

    # Sample from the predictive distribution
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

    # Backprop sampled loss to get gradient-output signals
    sampled_loss(m) = Flux.logitcrossentropy(m(x), sampled_y)
    _, s_grads = Zygote.withgradient(sampled_loss, model)
    s_grad = s_grads[1]

    # Collect gradient outputs via manual backward
    gouts = collect_grad_outputs(model, x, _initial_grad_output(model, x, sampled_y))

    for (idx, layer) in kfac_layers
        # ── A factor ──
        if haskey(acts, idx)
            aa = compute_cov_a(acts[idx], layer)
            if !haskey(opt.m_aa, idx)
                opt.m_aa[idx] = Matrix{Float64}(I, size(aa)...)
            end
            update_running_stat!(Float64.(aa), opt.m_aa[idx], opt.stat_decay)
        end

        # ── G factor ──
        if haskey(gouts, idx)
            gg = compute_cov_g(gouts[idx], layer; batch_averaged=opt.batch_averaged)
            if !haskey(opt.m_gg, idx)
                opt.m_gg[idx] = Matrix{Float64}(I, size(gg)...)
            end
            update_running_stat!(Float64.(gg), opt.m_gg[idx], opt.stat_decay)
        end
    end
end

"""
    _initial_grad_output(model, x, sampled_y)

Compute `dL/d(model_output)` for logitcrossentropy with one-hot `sampled_y`.
"""
function _initial_grad_output(model, x, sampled_y)
    output = model(x)
    # gradient of logitcrossentropy w.r.t. logits = softmax(logits) - y  (batch averaged)
    batch_size = size(output, 2)
    return (softmax(output; dims=1) .- sampled_y) ./ batch_size
end
