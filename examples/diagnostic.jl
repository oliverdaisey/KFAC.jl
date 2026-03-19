# Diagnostic: compare KFAC natural gradient vs standard gradient per-step

using KFAC, Flux, LinearAlgebra, Statistics, Zygote, Random

Random.seed!(77)

model_base = Chain(Dense(10 => 20, relu), Dense(20 => 5))
x = randn(Float32, 10, 64)
y = Float32.(Flux.onehotbatch(rand(1:5, 64), 1:5))
loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

println("=== Loss trajectory (100 steps, lr=0.01) ===")
println("Starting loss: $(round(loss_fn(model_base, x, y); digits=4))")

model_k = deepcopy(model_base)
opt_k = KFACOptimizer(lr=0.01, damping=0.01, TCov=1, TInv=5, momentum=0.0, kl_clip=10.0)
kfac_losses = Float64[]

model_sgd = deepcopy(model_base)
opt_sgd = Flux.setup(Flux.Optimisers.Descent(0.01), model_sgd)
sgd_losses = Float64[]

model_adam = deepcopy(model_base)
opt_adam = Flux.setup(Adam(0.01), model_adam)
adam_losses = Float64[]

for i in 1:100
    global model_k, model_sgd
    _, model_k = kfac_step!(opt_k, model_k, loss_fn, x, y)
    push!(kfac_losses, loss_fn(model_k, x, y))

    _, g2 = Flux.withgradient(m -> loss_fn(m, x, y), model_sgd)
    Flux.update!(opt_sgd, model_sgd, g2[1])
    push!(sgd_losses, loss_fn(model_sgd, x, y))

    _, g3 = Flux.withgradient(m -> loss_fn(m, x, y), model_adam)
    Flux.update!(opt_adam, model_adam, g3[1])
    push!(adam_losses, loss_fn(model_adam, x, y))
end

println("Step | KFAC     | SGD      | Adam     | Best")
for i in [1, 5, 10, 20, 30, 50, 70, 100]
    kl = round(kfac_losses[i]; digits=4)
    sl = round(sgd_losses[i]; digits=4)
    al = round(adam_losses[i]; digits=4)
    best = ["KFAC", "SGD", "Adam"][argmin([kfac_losses[i], sgd_losses[i], adam_losses[i]])]
    println("  $(lpad(i,3)) | $(lpad(kl,8)) | $(lpad(sl,8)) | $(lpad(al,8)) | $best")
end

println("\nKFAC covariance condition numbers after 100 steps:")
for idx in sort(collect(keys(opt_k.m_aa)))
    aa_cond = cond(opt_k.m_aa[idx])
    gg_cond = cond(opt_k.m_gg[idx])
    println("  Layer $idx: cond(A)=$(round(aa_cond; digits=1)), cond(G)=$(round(gg_cond; digits=1))")
end

# Key question: is the natural gradient direction actually different?
println("\n=== Natural gradient vs standard gradient ===")
loss_val, grads_tree = Zygote.withgradient(m -> loss_fn(m, x, y), model_k)
mg = grads_tree[1]

kfac_layers = KFAC.get_kfac_layers(model_k)
for (idx, layer) in kfac_layers
    r = KFAC.extract_layer_grads(mg, model_k, layer)
    if r !== nothing
        w_g, b_g = r
        p = KFAC.get_matrix_form_grad(w_g, b_g, layer)
        v = KFAC.get_natural_grad(opt_k, idx, p, opt_k.damping, layer)

        # Compare directions
        g_flat = vec(p)
        v_flat = vec(hcat(v[1] isa AbstractMatrix ? reshape(v[1], size(v[1],1), :) : reshape(v[1], :, 1),
                         v[2] !== nothing ? reshape(v[2], :, 1) : zeros(0,1)))

        # For Dense layers, natural grad matrix form
        if layer isa Dense
            v_mat = copy(v[1])
            if v[2] !== nothing
                v_mat = hcat(v_mat, reshape(v[2], :, 1))
            end
            v_flat = vec(v_mat)
        end

        cosine = dot(g_flat, v_flat) / (norm(g_flat) * norm(v_flat) + 1e-10)
        ratio = norm(v_flat) / (norm(g_flat) + 1e-10)
        println("  Layer $idx: cosine(grad, nat_grad)=$(round(cosine; digits=4)), norm_ratio=$(round(ratio; digits=4))")
    end
end
