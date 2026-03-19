# Utility functions for KFAC computations.

"""
    extract_patches(x, kernel_size, stride, padding)

Extract patches from a 4D input tensor for convolution covariance computation.

Given input feature maps of shape `(width, height, channels, batch)` (Flux convention),
extract sliding patches corresponding to the convolution operation defined by the given
`kernel_size`, `stride`, and `padding`.

Returns a matrix of shape `(in_c * kw * kh, n_patches)` where `n_patches = out_w * out_h * batch`.

# Arguments
- `x`: Input tensor of shape `(W, H, C, N)`.
- `kernel_size`: Tuple `(kw, kh)` specifying the convolution kernel dimensions.
- `stride`: Tuple `(sw, sh)` specifying the stride.
- `padding`: Tuple `(pw, ph)` specifying the padding on each side.
"""
function extract_patches(x::AbstractArray{T,4}, kernel_size::Tuple{Int,Int},
                         stride::Tuple{Int,Int}, padding::Tuple{Int,Int}) where T
    W, H, C, N = size(x)

    # Apply padding if needed
    if padding[1] > 0 || padding[2] > 0
        pw, ph = padding
        x_padded = zeros(T, W + 2pw, H + 2ph, C, N)
        x_padded[pw+1:pw+W, ph+1:ph+H, :, :] = x
        x = x_padded
    end

    Wp, Hp = size(x, 1), size(x, 2)
    kw, kh = kernel_size
    sw, sh = stride
    out_w = div(Wp - kw, sw) + 1
    out_h = div(Hp - kh, sh) + 1

    # Each patch is in_c * kw * kh, there are out_w * out_h * N patches
    patch_dim = C * kw * kh
    n_patches = out_w * out_h * N

    patches = zeros(T, patch_dim, n_patches)
    idx = 1
    for n in 1:N
        for j in 1:out_h
            for i in 1:out_w
                w_start = (i - 1) * sw + 1
                h_start = (j - 1) * sh + 1
                patch = @view x[w_start:w_start+kw-1, h_start:h_start+kh-1, :, n]
                patches[:, idx] = vec(patch)
                idx += 1
            end
        end
    end

    return patches
end

"""
    update_running_stat!(new_stat, running_stat, decay)

Update a running statistic using exponential moving average, modifying `running_stat` in place.

Implements: `running_stat .= decay * running_stat + (1 - decay) * new_stat`

This is equivalent to the PyTorch implementation which uses a reparameterized in-place update
for memory efficiency.

# Arguments
- `new_stat`: The newly computed statistic.
- `running_stat`: The running average to update (modified in place).
- `decay`: The decay factor (e.g., 0.95).
"""
function update_running_stat!(new_stat::AbstractArray, running_stat::AbstractArray, decay::Real)
    running_stat .*= decay / (1 - decay)
    running_stat .+= new_stat
    running_stat .*= (1 - decay)
    return running_stat
end

"""
    compute_cov_a(a, layer::Dense)

Compute the input activation covariance (Kronecker factor A) for a Dense layer.

For a Dense layer, the covariance of the input activations is `A = a' * a / batch_size`.
If the layer has bias, a column of ones is appended to `a`.

# Arguments
- `a`: Input activations of shape `(batch_size, in_dim)`.
- `layer`: A `Flux.Dense` layer.

# Returns
A symmetric matrix of shape `(in_dim [+1], in_dim [+1])`.
"""
function compute_cov_a(a::AbstractMatrix, layer::Dense)
    batch_size = size(a, 1)
    if layer.bias !== false && layer.bias !== nothing
        a = hcat(a, ones(eltype(a), batch_size, 1))
    end
    return a' * (a ./ batch_size)
end

"""
    compute_cov_a(a, layer::Conv; kernel_size, stride, padding)

Compute the input activation covariance (Kronecker factor A) for a Conv layer.

Extracts patches from the input and computes their outer product covariance.

# Arguments
- `a`: Input activations of shape `(W, H, C, N)`.
- `layer`: A `Flux.Conv` layer.
"""
function compute_cov_a(a::AbstractArray{T,4}, layer::Conv) where T
    batch_size = size(a, 4)
    ks = layer.stride
    ksize = size(layer.weight)[1:2]
    pad = layer.pad[1:2]

    patches = extract_patches(a, ksize, ks, pad)
    spatial_size = let
        W, H = size(a, 1), size(a, 2)
        pw, ph = pad
        kw, kh = ksize
        sw, sh = ks
        out_w = div(W + 2pw - kw, sw) + 1
        out_h = div(H + 2ph - kh, sh) + 1
        out_w * out_h
    end

    # patches is (patch_dim, n_patches) where n_patches = spatial_size * batch_size
    # Append ones for bias
    if layer.bias !== false && layer.bias !== nothing
        patches = vcat(patches, ones(T, 1, size(patches, 2)))
    end

    patches ./= spatial_size
    return (patches * patches') ./ batch_size
end

"""
    compute_cov_g(g, layer::Dense; batch_averaged=true)

Compute the gradient output covariance (Kronecker factor G) for a Dense layer.

# Arguments
- `g`: Gradient output of shape `(batch_size, out_dim)`.
- `layer`: A `Flux.Dense` layer.
- `batch_averaged`: Whether the gradient is already averaged over the batch.

# Returns
A symmetric matrix of shape `(out_dim, out_dim)`.
"""
function compute_cov_g(g::AbstractMatrix, layer::Dense; batch_averaged::Bool=true)
    batch_size = size(g, 1)
    if batch_averaged
        return g' * (g .* batch_size)
    else
        return g' * (g ./ batch_size)
    end
end

"""
    compute_cov_g(g, layer::Conv; batch_averaged=true)

Compute the gradient output covariance (Kronecker factor G) for a Conv layer.

# Arguments
- `g`: Gradient output of shape `(W, H, C_out, N)`.
- `layer`: A `Flux.Conv` layer.
- `batch_averaged`: Whether the gradient is already averaged over the batch.

# Returns
A symmetric matrix of shape `(out_channels, out_channels)`.
"""
function compute_cov_g(g::AbstractArray{T,4}, layer::Conv; batch_averaged::Bool=true) where T
    spatial_size = size(g, 1) * size(g, 2)
    batch_size = size(g, 4)

    # Reshape to (spatial_size * batch_size, n_filters)
    n_filters = size(g, 3)
    g_reshaped = reshape(permutedims(g, (1, 2, 4, 3)), :, n_filters)

    if batch_averaged
        g_reshaped = g_reshaped .* batch_size
    end
    g_reshaped = g_reshaped .* spatial_size

    return g_reshaped' * (g_reshaped ./ size(g_reshaped, 1))
end

"""
    compute_mat_grad(input, grad_output, layer::Dense)

Compute per-sample gradient matrices for a Dense layer (used in EKFAC).

# Arguments
- `input`: Input activations `(batch_size, in_dim)`.
- `grad_output`: Gradient of the output `(batch_size, out_dim)`.
- `layer`: A `Flux.Dense` layer.

# Returns
A 3D array of shape `(out_dim, in_dim [+1], batch_size)`.
"""
function compute_mat_grad(input::AbstractMatrix, grad_output::AbstractMatrix, layer::Dense)
    batch_size = size(input, 1)
    if layer.bias !== false && layer.bias !== nothing
        input = hcat(input, ones(eltype(input), batch_size, 1))
    end
    # For each sample: grad_output[i,:] * input[i,:]' -> (out_dim, in_dim)
    # Batched: (batch_size, out_dim, 1) * (batch_size, 1, in_dim) -> (batch_size, out_dim, in_dim)
    go = reshape(grad_output, batch_size, size(grad_output, 2), 1)
    inp = reshape(input, batch_size, 1, size(input, 2))
    grad = go .* inp  # broadcasting: (batch, out_dim, in_dim)
    return permutedims(grad, (2, 3, 1))  # (out_dim, in_dim, batch)
end

"""
    compute_mat_grad(input, grad_output, layer::Conv)

Compute per-sample gradient matrices for a Conv layer (used in EKFAC).

# Arguments
- `input`: Input activations `(W, H, C_in, N)`.
- `grad_output`: Gradient of the output `(W_out, H_out, C_out, N)`.
- `layer`: A `Flux.Conv` layer.

# Returns
A 3D array of shape `(C_out, patch_dim [+1], batch_size)`.
"""
function compute_mat_grad(input::AbstractArray{T,4}, grad_output::AbstractArray{T,4}, layer::Conv) where T
    batch_size = size(input, 4)
    ksize = size(layer.weight)[1:2]
    ks = layer.stride
    pad = layer.pad[1:2]

    patches = extract_patches(input, ksize, ks, pad)

    # Append ones for bias
    if layer.bias !== false && layer.bias !== nothing
        patches = vcat(patches, ones(T, 1, size(patches, 2)))
    end

    patch_dim = size(patches, 1)
    out_c = size(grad_output, 3)
    spatial_size = size(grad_output, 1) * size(grad_output, 2)

    # Reshape patches: (patch_dim, spatial_size, batch)
    patches_3d = reshape(patches, patch_dim, spatial_size, batch_size)

    # Reshape grad_output: (spatial_size, out_c, batch) -> (out_c, spatial_size, batch)
    g_reshaped = permutedims(reshape(grad_output, spatial_size, out_c, batch_size), (2, 1, 3))

    # Per-sample gradient: g_reshaped[:,:,b] * patches_3d[:,:,b]' for each b
    grad = zeros(T, out_c, patch_dim, batch_size)
    for b in 1:batch_size
        grad[:, :, b] = g_reshaped[:, :, b] * patches_3d[:, :, b]'
    end

    return grad
end
