@testset "KFACOptimizer" begin
    @testset "Construction" begin
        opt = KFACOptimizer()
        @test opt.lr == 0.001
        @test opt.momentum == 0.9
        @test opt.stat_decay == 0.95
        @test opt.damping == 0.001
        @test opt.kl_clip == 0.001
        @test opt.weight_decay == 0.0
        @test opt.TCov == 10
        @test opt.TInv == 100
        @test opt.batch_averaged == true
        @test opt.steps == 0

        opt2 = KFACOptimizer(lr=0.01, momentum=0.5, damping=0.03)
        @test opt2.lr == 0.01
        @test opt2.momentum == 0.5
        @test opt2.damping == 0.03

        @test_throws ArgumentError KFACOptimizer(lr=-0.1)
        @test_throws ArgumentError KFACOptimizer(momentum=-0.1)
        @test_throws ArgumentError KFACOptimizer(weight_decay=-0.1)
    end

    @testset "get_kfac_layers" begin
        model = Chain(Dense(4 => 3, relu), Dense(3 => 2))
        layers = KFAC.get_kfac_layers(model)
        @test length(layers) == 2
        @test layers[1][1] == 1
        @test layers[2][1] == 2
        @test layers[1][2] isa Dense
        @test layers[2][2] isa Dense

        # With non-KFAC layers mixed in
        model2 = Chain(Dense(4 => 3, relu), BatchNorm(3), Dense(3 => 2))
        layers2 = KFAC.get_kfac_layers(model2)
        @test length(layers2) == 2  # BatchNorm is not a KFAC layer
    end

    @testset "collect_activations" begin
        model = Chain(Dense(4 => 3, relu), Dense(3 => 2))
        x = randn(Float32, 4, 8)
        acts = KFAC.collect_activations(model, x)
        @test haskey(acts, 1)
        @test haskey(acts, 2)
        @test size(acts[1]) == (8, 4)   # (batch, in_features)
        @test size(acts[2]) == (8, 3)   # (batch, hidden_features)
    end

    @testset "collect_grad_outputs" begin
        model = Chain(Dense(4 => 3, relu), Dense(3 => 2))
        x = randn(Float32, 4, 8)
        output = model(x)
        dy = randn(Float32, size(output)...)
        gouts = KFAC.collect_grad_outputs(model, x, dy)
        @test haskey(gouts, 1)
        @test haskey(gouts, 2)
        @test size(gouts[2]) == (8, 2)   # (batch, out_features)
        @test size(gouts[1]) == (8, 3)   # (batch, hidden_features)
    end

    @testset "get_matrix_form_grad - Dense" begin
        layer = Dense(4 => 3)
        w_grad = randn(3, 4)
        b_grad = randn(3)
        mat = KFAC.get_matrix_form_grad(w_grad, b_grad, layer)
        @test size(mat) == (3, 5)
        @test mat[:, 1:4] == w_grad
        @test mat[:, 5] == b_grad
    end

    @testset "get_matrix_form_grad - Conv" begin
        layer = Conv((3, 3), 2 => 4)
        w_grad = randn(Float32, 3, 3, 2, 4)
        b_grad = randn(Float32, 4)
        mat = KFAC.get_matrix_form_grad(w_grad, b_grad, layer)
        @test size(mat) == (4, 19)  # (out_c, in_c*kw*kh + 1)
    end

    @testset "update_inv!" begin
        opt = KFACOptimizer()
        n_a, n_g = 5, 3
        # Create PSD matrices
        A = randn(n_a, n_a)
        opt.m_aa[1] = A' * A + 0.01I
        G = randn(n_g, n_g)
        opt.m_gg[1] = G' * G + 0.01I

        KFAC.update_inv!(opt, 1)

        @test haskey(opt.Q_a, 1)
        @test haskey(opt.Q_g, 1)
        @test haskey(opt.d_a, 1)
        @test haskey(opt.d_g, 1)
        @test length(opt.d_a[1]) == n_a
        @test length(opt.d_g[1]) == n_g
        @test all(opt.d_a[1] .> 0)
        @test all(opt.d_g[1] .> 0)

        # Verify eigenvectors reconstruct the matrix
        reconstructed_a = opt.Q_a[1] * Diagonal(opt.d_a[1]) * opt.Q_a[1]'
        @test reconstructed_a ≈ Symmetric(opt.m_aa[1]) atol=1e-8
    end

    @testset "get_natural_grad" begin
        opt = KFACOptimizer(damping=0.01)
        layer = Dense(4 => 3)

        # Setup eigendecomposition
        n_a, n_g = 5, 3  # 4+1 for bias, 3 for output
        A = randn(n_a, n_a); A = A'A + 0.1I
        G = randn(n_g, n_g); G = G'G + 0.1I
        opt.m_aa[1] = A; opt.m_gg[1] = G
        KFAC.update_inv!(opt, 1)

        w_grad = randn(3, 4)
        b_grad = randn(3)
        p = KFAC.get_matrix_form_grad(w_grad, b_grad, layer)

        v_w, v_b = KFAC.get_natural_grad(opt, 1, p, opt.damping, layer)
        @test size(v_w) == size(layer.weight)
        @test size(v_b) == size(layer.bias)
    end

    @testset "kl_clip" begin
        updates = [(randn(3, 4), randn(3)), (randn(2, 3), randn(2))]
        grads = [(randn(3, 4), randn(3)), (randn(2, 3), randn(2))]
        clipped, nu = KFAC.kl_clip(updates, grads, 0.01, 0.001)
        @test length(clipped) == 2
        @test nu > 0
        @test nu <= 1.0

        # With tiny kl_clip, nu should be small
        _, nu_small = KFAC.kl_clip(updates, grads, 1.0, 1e-10)
        @test nu_small < 1.0
    end

    @testset "kfac_step! - Dense only model" begin
        Random.seed!(123)
        model = Chain(Dense(4 => 8, relu), Dense(8 => 3))
        opt = KFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=1, momentum=0.0)

        x = randn(Float32, 4, 16)
        y = Flux.onehotbatch(rand(1:3, 16), 1:3)
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        loss1, model = kfac_step!(opt, model, loss_fn, x, Float32.(y))
        @test isfinite(loss1)
        @test opt.steps == 1

        # After several steps, loss should decrease
        losses = [loss1]
        for i in 1:20
            l, model = kfac_step!(opt, model, loss_fn, x, Float32.(y))
            push!(losses, l)
        end
        @test losses[end] < losses[1]
    end

    @testset "kfac_step! - step counter increments" begin
        model = Chain(Dense(2 => 3, relu), Dense(3 => 2))
        opt = KFACOptimizer(TCov=1, TInv=1)

        x = randn(Float32, 2, 4)
        y = Flux.onehotbatch(rand(1:2, 4), 1:2)
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        for i in 1:5
            _, model = kfac_step!(opt, model, loss_fn, x, Float32.(y))
        end
        @test opt.steps == 5
    end

    @testset "kfac_step! - covariance matrices populated" begin
        model = Chain(Dense(3 => 4, relu), Dense(4 => 2))
        opt = KFACOptimizer(TCov=1, TInv=1)

        x = randn(Float32, 3, 8)
        y = Flux.onehotbatch(rand(1:2, 8), 1:2)
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        _, model = kfac_step!(opt, model, loss_fn, x, Float32.(y))

        @test haskey(opt.m_aa, 1)
        @test haskey(opt.m_aa, 2)
        @test haskey(opt.m_gg, 1)
        @test haskey(opt.m_gg, 2)
        @test haskey(opt.Q_a, 1)
        @test haskey(opt.Q_g, 1)
    end
end
