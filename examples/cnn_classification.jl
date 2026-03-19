# CNN classification example using KFAC and EKFAC optimizers.
#
# This example trains a small CNN on synthetic image data and compares
# KFAC, EKFAC, and Adam, demonstrating KFAC's support for Conv layers.

using KFAC
using Flux
using Random
using Statistics
using LinearAlgebra

Random.seed!(42)

# ── Generate synthetic image data ────────────────────────────────────

function generate_image_data(n_samples=200, img_size=8, n_classes=3)
    # Create simple synthetic images where class is determined by
    # spatial patterns (horizontal vs vertical vs diagonal dominance)
    X = zeros(Float32, img_size, img_size, 1, n_samples)
    labels = Int[]

    for i in 1:n_samples
        cls = mod1(i, n_classes)
        push!(labels, cls)
        if cls == 1
            # Horizontal stripes
            for r in 1:img_size
                if mod(r, 2) == 0
                    X[r, :, 1, i] .= 1.0f0
                end
            end
        elseif cls == 2
            # Vertical stripes
            for c in 1:img_size
                if mod(c, 2) == 0
                    X[:, c, 1, i] .= 1.0f0
                end
            end
        else
            # Diagonal pattern
            for r in 1:img_size, c in 1:img_size
                if mod(r + c, 2) == 0
                    X[r, c, 1, i] = 1.0f0
                end
            end
        end
        # Add noise
        X[:, :, 1, i] .+= 0.3f0 .* randn(Float32, img_size, img_size)
    end

    Y = Float32.(Flux.onehotbatch(labels, 1:n_classes))
    return X, Y, labels
end

X_train, Y_train, labels_train = generate_image_data(200)
X_test, Y_test, labels_test = generate_image_data(60)

# ── Model factory ────────────────────────────────────────────────────
# Conv(3x3, 1=>4, relu) -> Conv(3x3, 4=>8, relu) -> Flatten -> Dense -> classes
# Input: 8x8x1 -> 6x6x4 -> 4x4x8 = 128 -> 3

function make_cnn()
    Chain(
        Conv((3, 3), 1 => 4, relu),
        Conv((3, 3), 4 => 8, relu),
        Flux.flatten,
        Dense(128 => 3)
    )
end

loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

function accuracy(model, X, labels)
    preds = Flux.onecold(model(X))
    return mean(preds .== labels)
end

# ── Train with KFAC ──────────────────────────────────────────────────

println("=" ^ 60)
println("Training CNN with KFAC")
println("=" ^ 60)

model_kfac = make_cnn()
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
println("Training CNN with EKFAC")
println("=" ^ 60)

model_ekfac = make_cnn()
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

# ── Train with Adam (for comparison) ─────────────────────────────────

println("\n" * "=" ^ 60)
println("Training CNN with Adam")
println("=" ^ 60)

model_adam = make_cnn()
opt_adam = Flux.setup(Adam(0.01), model_adam)

adam_losses = Float64[]
for epoch in 1:50
    loss, grads = Flux.withgradient(m -> loss_fn(m, X_train, Y_train), model_adam)
    Flux.update!(opt_adam, model_adam, grads[1])
    push!(adam_losses, loss)
    if epoch % 10 == 0
        acc = accuracy(model_adam, X_test, labels_test)
        println("  Epoch $epoch: loss = $(round(loss; digits=4)), test acc = $(round(acc*100; digits=1))%")
    end
end

# ── Summary ──────────────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("Final Results (CNN)")
println("=" ^ 60)
println("  KFAC  — final loss: $(round(kfac_losses[end]; digits=4)), test acc: $(round(accuracy(model_kfac, X_test, labels_test)*100; digits=1))%")
println("  EKFAC — final loss: $(round(ekfac_losses[end]; digits=4)), test acc: $(round(accuracy(model_ekfac, X_test, labels_test)*100; digits=1))%")
println("  Adam  — final loss: $(round(adam_losses[end]; digits=4)), test acc: $(round(accuracy(model_adam, X_test, labels_test)*100; digits=1))%")
