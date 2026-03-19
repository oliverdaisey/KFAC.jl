# Diagnostic 2: KFAC on a linear softmax model (where Fisher is exact)
# and comparison with different hyperparameter settings.

using KFAC, Flux, LinearAlgebra, Statistics, Zygote, Random

Random.seed!(77)

# ── Test 1: Linear model (KFAC should be near-optimal) ──────────────

println("=" ^ 60)
println("Test 1: Linear softmax model (KFAC ~ Newton's method)")
println("=" ^ 60)

x = randn(Float32, 10, 128)
y = Float32.(Flux.onehotbatch(rand(1:3, 128), 1:3))
loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

# KFAC on linear model
model_k = Chain(Dense(10 => 3))
opt_k = KFACOptimizer(lr=0.1, damping=0.001, TCov=1, TInv=1, momentum=0.0, kl_clip=100.0)
kfac_losses = Float64[]
for i in 1:50
    global model_k
    _, model_k = kfac_step!(opt_k, model_k, loss_fn, x, y)
    push!(kfac_losses, loss_fn(model_k, x, y))
end

# SGD on linear model
model_s = Chain(Dense(10 => 3))
opt_s = Flux.setup(Flux.Optimisers.Descent(0.1), model_s)
sgd_losses = Float64[]
for i in 1:50
    _, g = Flux.withgradient(m -> loss_fn(m, x, y), model_s)
    Flux.update!(opt_s, model_s, g[1])
    push!(sgd_losses, loss_fn(model_s, x, y))
end

# Adam on linear model
model_a = Chain(Dense(10 => 3))
opt_a = Flux.setup(Adam(0.1), model_a)
adam_losses = Float64[]
for i in 1:50
    _, g = Flux.withgradient(m -> loss_fn(m, x, y), model_a)
    Flux.update!(opt_a, model_a, g[1])
    push!(adam_losses, loss_fn(model_a, x, y))
end

println("Step | KFAC     | SGD      | Adam     | Best")
for i in [1, 2, 3, 5, 10, 20, 50]
    kl = round(kfac_losses[i]; digits=4)
    sl = round(sgd_losses[i]; digits=4)
    al = round(adam_losses[i]; digits=4)
    best = ["KFAC", "SGD", "Adam"][argmin([kfac_losses[i], sgd_losses[i], adam_losses[i]])]
    println("  $(lpad(i,3)) | $(lpad(kl,8)) | $(lpad(sl,8)) | $(lpad(al,8)) | $best")
end

# ── Test 2: MLP with tuned hyperparameters ───────────────────────────

println("\n" * "=" ^ 60)
println("Test 2: MLP (tuned hyperparameters, lr=0.1 for KFAC)")
println("=" ^ 60)

Random.seed!(42)
x2 = randn(Float32, 20, 200)
y2 = Float32.(Flux.onehotbatch(rand(1:5, 200), 1:5))

configs = [
    ("KFAC lr=0.1 d=0.03", () -> begin
        m = Chain(Dense(20 => 64, relu), Dense(64 => 32, relu), Dense(32 => 5))
        o = KFACOptimizer(lr=0.1, damping=0.03, TCov=1, TInv=5, momentum=0.9, kl_clip=0.01)
        (m, o, :kfac)
    end),
    ("KFAC lr=0.03 d=0.01", () -> begin
        m = Chain(Dense(20 => 64, relu), Dense(64 => 32, relu), Dense(32 => 5))
        o = KFACOptimizer(lr=0.03, damping=0.01, TCov=1, TInv=5, momentum=0.9, kl_clip=0.01)
        (m, o, :kfac)
    end),
    ("SGD lr=0.1 mom=0.9", () -> begin
        m = Chain(Dense(20 => 64, relu), Dense(64 => 32, relu), Dense(32 => 5))
        o = Flux.setup(Flux.Optimisers.Momentum(0.1, 0.9), m)
        (m, o, :sgd)
    end),
    ("SGD lr=0.03 mom=0.9", () -> begin
        m = Chain(Dense(20 => 64, relu), Dense(64 => 32, relu), Dense(32 => 5))
        o = Flux.setup(Flux.Optimisers.Momentum(0.03, 0.9), m)
        (m, o, :sgd)
    end),
    ("Adam lr=0.01", () -> begin
        m = Chain(Dense(20 => 64, relu), Dense(64 => 32, relu), Dense(32 => 5))
        o = Flux.setup(Adam(0.01), m)
        (m, o, :sgd)
    end),
]

all_losses = Dict{String, Vector{Float64}}()
for (name, setup_fn) in configs
    Random.seed!(42)
    m, o, typ = setup_fn()
    losses = Float64[]
    for i in 1:100
        if typ == :kfac
            _, m = kfac_step!(o, m, loss_fn, x2, y2)
            push!(losses, loss_fn(m, x2, y2))
        else
            _, g = Flux.withgradient(mm -> loss_fn(mm, x2, y2), m)
            Flux.update!(o, m, g[1])
            push!(losses, loss_fn(m, x2, y2))
        end
    end
    all_losses[name] = losses
end

let header = "Step |"
    for (name, _) in configs
        header *= " $(lpad(name[1:min(end,18)], 18)) |"
    end
    println(header)
end

for step in [1, 5, 10, 20, 50, 100]
    line = "$(lpad(step, 4)) |"
    vals = [all_losses[name][step] for (name, _) in configs]
    best_idx = argmin(vals)
    for (j, (name, _)) in enumerate(configs)
        v = round(all_losses[name][step]; digits=4)
        mark = j == best_idx ? "*" : " "
        line *= " $(lpad(v, 17))$mark |"
    end
    println(line)
end

# ── Test 3: Verify the natural gradient on a single step ─────────────

println("\n" * "=" ^ 60)
println("Test 3: Single-step loss reduction comparison")
println("=" ^ 60)

Random.seed!(42)
base_model = Chain(Dense(20 => 64, relu), Dense(64 => 32, relu), Dense(32 => 5))
base_loss = loss_fn(base_model, x2, y2)
println("Base loss: $(round(base_loss; digits=4))")

# One KFAC step (with pre-warmed cov stats)
m_k = deepcopy(base_model)
opt_k2 = KFACOptimizer(lr=0.1, damping=0.03, TCov=1, TInv=1, momentum=0.0, kl_clip=0.01)
_, m_k = kfac_step!(opt_k2, m_k, loss_fn, x2, y2)
println("After 1 KFAC step:  $(round(loss_fn(m_k, x2, y2); digits=4))  (delta = $(round(loss_fn(m_k, x2, y2) - base_loss; digits=4)))")

# One SGD step (same lr)
m_s = deepcopy(base_model)
opt_s2 = Flux.setup(Flux.Optimisers.Descent(0.1), m_s)
_, g = Flux.withgradient(m -> loss_fn(m, x2, y2), m_s)
Flux.update!(opt_s2, m_s, g[1])
println("After 1 SGD step:   $(round(loss_fn(m_s, x2, y2); digits=4))  (delta = $(round(loss_fn(m_s, x2, y2) - base_loss; digits=4)))")

# One Adam step
m_a = deepcopy(base_model)
opt_a2 = Flux.setup(Adam(0.1), m_a)
_, g = Flux.withgradient(m -> loss_fn(m, x2, y2), m_a)
Flux.update!(opt_a2, m_a, g[1])
println("After 1 Adam step:  $(round(loss_fn(m_a, x2, y2); digits=4))  (delta = $(round(loss_fn(m_a, x2, y2) - base_loss; digits=4)))")
