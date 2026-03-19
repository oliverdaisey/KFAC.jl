# Simple classification example using KFAC and EKFAC optimizers.
#
# This example trains small MLPs on a synthetic linearly-separable dataset
# and compares KFAC, EKFAC, and standard gradient descent.

using KFAC
using Flux
using Random
using Statistics
using LinearAlgebra

Random.seed!(42)

# ── Generate synthetic data ──────────────────────────────────────────

function generate_data(n_samples=500, n_features=10, n_classes=3)
    # Random linear decision boundaries
    W_true = randn(Float32, n_classes, n_features)

    X = randn(Float32, n_features, n_samples)
    logits = W_true * X
    labels = [argmax(logits[:, i]) for i in 1:n_samples]
    Y = Float32.(Flux.onehotbatch(labels, 1:n_classes))

    return X, Y, labels
end

X_train, Y_train, labels_train = generate_data(500)
X_test, Y_test, labels_test = generate_data(200)

# ── Model factory ────────────────────────────────────────────────────

make_model() = Chain(
    Dense(10 => 32, relu),
    Dense(32 => 16, relu),
    Dense(16 => 3)
)

loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

function accuracy(model, X, labels)
    preds = Flux.onecold(model(X))
    return mean(preds .== labels)
end

# ── Train with KFAC ──────────────────────────────────────────────────

println("=" ^ 60)
println("Training with KFAC")
println("=" ^ 60)

model_kfac = make_model()
opt_kfac = KFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=5, momentum=0.9)

kfac_losses = Float64[]
for epoch in 1:50
    global model_kfac
    loss, model_kfac = kfac_step!(opt_kfac, model_kfac, loss_fn, X_train, Y_train)
    push!(kfac_losses, loss)
    if epoch % 10 == 0
        acc = accuracy(model_kfac, X_test, labels_test)
        println("  Epoch $epoch: loss = $(round(loss; digits=4)), test acc = $(round(acc*100; digits=1))%")
    end
end

# ── Train with EKFAC ─────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("Training with EKFAC")
println("=" ^ 60)

model_ekfac = make_model()
opt_ekfac = EKFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=5, TScal=2, momentum=0.9)

ekfac_losses = Float64[]
for epoch in 1:50
    global model_ekfac
    loss, model_ekfac = ekfac_step!(opt_ekfac, model_ekfac, loss_fn, X_train, Y_train)
    push!(ekfac_losses, loss)
    if epoch % 10 == 0
        acc = accuracy(model_ekfac, X_test, labels_test)
        println("  Epoch $epoch: loss = $(round(loss; digits=4)), test acc = $(round(acc*100; digits=1))%")
    end
end

# ── Train with standard SGD (for comparison) ─────────────────────────

println("\n" * "=" ^ 60)
println("Training with SGD (Flux.Adam)")
println("=" ^ 60)

model_sgd = make_model()
opt_sgd = Flux.setup(Adam(0.01), model_sgd)

sgd_losses = Float64[]
for epoch in 1:50
    loss, grads = Flux.withgradient(m -> loss_fn(m, X_train, Y_train), model_sgd)
    Flux.update!(opt_sgd, model_sgd, grads[1])
    push!(sgd_losses, loss)
    if epoch % 10 == 0
        acc = accuracy(model_sgd, X_test, labels_test)
        println("  Epoch $epoch: loss = $(round(loss; digits=4)), test acc = $(round(acc*100; digits=1))%")
    end
end

# ── Summary ──────────────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("Final Results")
println("=" ^ 60)
println("  KFAC  — final loss: $(round(kfac_losses[end]; digits=4)), test acc: $(round(accuracy(model_kfac, X_test, labels_test)*100; digits=1))%")
println("  EKFAC — final loss: $(round(ekfac_losses[end]; digits=4)), test acc: $(round(accuracy(model_ekfac, X_test, labels_test)*100; digits=1))%")
println("  Adam  — final loss: $(round(sgd_losses[end]; digits=4)), test acc: $(round(accuracy(model_sgd, X_test, labels_test)*100; digits=1))%")
