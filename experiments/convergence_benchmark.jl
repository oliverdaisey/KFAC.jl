# Convergence benchmark: KFAC vs EKFAC vs SGD+Momentum vs Adam
#
# Trains a 3-layer MLP (20 -> 64 -> 32 -> 5) on a synthetic 5-class
# classification task (200 samples, 20 features) and compares per-step
# convergence speed across optimisers.
#
# Run with:
#   julia --project=. experiments/convergence_benchmark.jl

using KFAC
using Flux
using Random
using Statistics
using LinearAlgebra
using Printf

Random.seed!(42)

# ── Setup ────────────────────────────────────────────────────────────

n_features = 20
n_classes = 5
n_train = 200
n_epochs = 100

x = randn(Float32, n_features, n_train)
y = Float32.(Flux.onehotbatch(rand(1:n_classes, n_train), 1:n_classes))
loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

make_model() = Chain(
    Dense(n_features => 64, relu),
    Dense(64 => 32, relu),
    Dense(32 => n_classes)
)

println("Dataset: $(n_train) samples, $(n_features) features, $(n_classes) classes")
println("Model:   Dense($(n_features)=>64, relu) -> Dense(64=>32, relu) -> Dense(32=>$(n_classes))")
println("Epochs:  $(n_epochs)\n")

# ── Run each optimiser ───────────────────────────────────────────────

results = Dict{String, Vector{Float64}}()
times   = Dict{String, Float64}()

# KFAC (paper-recommended hyperparameters)
Random.seed!(42)
model = make_model()
opt = KFACOptimizer(lr=0.1, damping=0.03, TCov=1, TInv=5, momentum=0.9, kl_clip=0.01)
losses = Float64[]
t = @elapsed for i in 1:n_epochs
    global model
    _, model = kfac_step!(opt, model, loss_fn, x, y)
    push!(losses, loss_fn(model, x, y))
end
results["KFAC"] = losses
times["KFAC"] = t

# EKFAC
Random.seed!(42)
model = make_model()
opt = EKFACOptimizer(lr=0.1, damping=0.03, TCov=1, TInv=5, TScal=2, momentum=0.9, kl_clip=0.01)
losses = Float64[]
t = @elapsed for i in 1:n_epochs
    global model
    _, model = ekfac_step!(opt, model, loss_fn, x, y)
    push!(losses, loss_fn(model, x, y))
end
results["EKFAC"] = losses
times["EKFAC"] = t

# SGD with momentum
Random.seed!(42)
model = make_model()
opt_s = Flux.setup(Flux.Optimisers.Momentum(0.1, 0.9), model)
losses = Float64[]
t = @elapsed for i in 1:n_epochs
    l, g = Flux.withgradient(m -> loss_fn(m, x, y), model)
    Flux.update!(opt_s, model, g[1])
    push!(losses, loss_fn(model, x, y))
end
results["SGD+Mom"] = losses
times["SGD+Mom"] = t

# Adam
Random.seed!(42)
model = make_model()
opt_a = Flux.setup(Adam(0.01), model)
losses = Float64[]
t = @elapsed for i in 1:n_epochs
    l, g = Flux.withgradient(m -> loss_fn(m, x, y), model)
    Flux.update!(opt_a, model, g[1])
    push!(losses, loss_fn(model, x, y))
end
results["Adam"] = losses
times["Adam"] = t

# ── Training loss trajectory ─────────────────────────────────────────

methods = ["KFAC", "EKFAC", "SGD+Mom", "Adam"]

println("=" ^ 70)
println("TRAINING LOSS TRAJECTORY")
println("=" ^ 70)

@printf("%5s", "Step")
for m in methods
    @printf("  %12s", m)
end
println("  Best")

@printf("%5s", "----")
for _ in methods
    @printf("  %12s", "----------")
end
println("  ----")

for step in [1, 5, 10, 20, 30, 50, 75, 100]
    @printf("%5d", step)
    vals = [results[m][step] for m in methods]
    best_idx = argmin(vals)
    for (j, m) in enumerate(methods)
        v = results[m][step]
        mark = j == best_idx ? "*" : " "
        @printf("  %11.4f%s", v, mark)
    end
    println("  $(methods[best_idx])")
end

# ── Convergence speed ────────────────────────────────────────────────

println("\n" * "=" ^ 70)
println("CONVERGENCE SPEED: steps to reach loss < threshold")
println("=" ^ 70)

@printf("  %-14s", "Threshold")
for m in methods
    @printf("  %8s", m)
end
println()

@printf("  %-14s", "-----------")
for _ in methods
    @printf("  %8s", "------")
end
println()

for threshold in [1.0, 0.5, 0.1, 0.01, 0.001]
    @printf("  loss < %-6.3f", threshold)
    for m in methods
        ep = findfirst(<(threshold), results[m])
        s = ep === nothing ? ">$(n_epochs)" : string(ep)
        @printf("  %8s", s)
    end
    println()
end

# ── Wall-clock time ──────────────────────────────────────────────────

println("\n" * "=" ^ 70)
println("WALL-CLOCK TIME")
println("=" ^ 70)

for m in methods
    @printf("  %-8s  %6.2fs  (%.3fs/step)\n", m, times[m], times[m] / n_epochs)
end

# ── Sanity checks ────────────────────────────────────────────────────

println("\n" * "=" ^ 70)
println("SANITY CHECKS")
println("=" ^ 70)

kfac_final = results["KFAC"][end]
status = kfac_final < 0.001 ? "PASS" : "FAIL"
@printf("  [%s] KFAC reaches near-zero train loss: %.6f\n", status, kfac_final)

kfac_to_01 = findfirst(<(0.1), results["KFAC"])
sgd_to_01  = findfirst(<(0.1), results["SGD+Mom"])
if kfac_to_01 !== nothing && sgd_to_01 !== nothing
    status = kfac_to_01 < sgd_to_01 ? "PASS" : "FAIL"
    @printf("  [%s] KFAC reaches loss<0.1 faster than SGD: %d vs %d steps\n",
            status, kfac_to_01, sgd_to_01)
elseif kfac_to_01 !== nothing
    println("  [PASS] KFAC reaches loss<0.1 at step $kfac_to_01, SGD did not converge")
else
    println("  [FAIL] KFAC did not reach loss<0.1")
end

ekfac_final = results["EKFAC"][end]
status = ekfac_final < 0.001 ? "PASS" : "FAIL"
@printf("  [%s] EKFAC reaches near-zero train loss: %.6f\n", status, ekfac_final)

kfac_50 = results["KFAC"][50]
sgd_50  = results["SGD+Mom"][50]
ratio   = sgd_50 / max(kfac_50, 1e-10)
status  = ratio > 10 ? "PASS" : "WARN"
@printf("  [%s] At step 50, SGD/KFAC loss ratio: %.0fx (KFAC=%.4f, SGD=%.4f)\n",
        status, ratio, kfac_50, sgd_50)

println()
