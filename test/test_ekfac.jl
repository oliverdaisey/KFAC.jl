@testset "EKFACOptimizer" begin
    @testset "Construction" begin
        opt = EKFACOptimizer()
        @test opt.lr == 0.001
        @test opt.TScal == 10
        @test opt.steps == 0

        opt2 = EKFACOptimizer(lr=0.003, TScal=5, damping=0.03)
        @test opt2.lr == 0.003
        @test opt2.TScal == 5
        @test opt2.damping == 0.03

        @test_throws ArgumentError EKFACOptimizer(lr=-0.1)
        @test_throws ArgumentError EKFACOptimizer(momentum=-0.1)
        @test_throws ArgumentError EKFACOptimizer(weight_decay=-0.1)
    end

    @testset "update_inv! - EKFAC variant" begin
        opt = EKFACOptimizer()
        n_a, n_g = 5, 3
        A = randn(n_a, n_a); A = A'A + 0.01I
        G = randn(n_g, n_g); G = G'G + 0.01I
        opt.m_aa[1] = A
        opt.m_gg[1] = G

        KFAC.update_inv!(opt, 1)

        @test haskey(opt.S_l, 1)
        @test size(opt.S_l[1]) == (n_g, n_a)
        @test all(opt.S_l[1] .> 0)

        # S_l should equal d_g * d_a' initially
        @test opt.S_l[1] ≈ opt.d_g[1] * opt.d_a[1]' atol=1e-10
    end

    @testset "ekfac_step! - Dense only model" begin
        Random.seed!(456)
        model = Chain(Dense(4 => 8, relu), Dense(8 => 3))
        opt = EKFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=1, TScal=1, momentum=0.0)

        x = randn(Float32, 4, 16)
        y = Flux.onehotbatch(rand(1:3, 16), 1:3)
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        loss1, model = ekfac_step!(opt, model, loss_fn, x, Float32.(y))
        @test isfinite(loss1)
        @test opt.steps == 1

        losses = [loss1]
        for i in 1:20
            l, model = ekfac_step!(opt, model, loss_fn, x, Float32.(y))
            push!(losses, l)
        end
        @test losses[end] < losses[1]
    end

    @testset "ekfac_step! - scaling factors populated" begin
        model = Chain(Dense(3 => 4, relu), Dense(4 => 2))
        opt = EKFACOptimizer(TCov=1, TInv=1, TScal=1)

        x = randn(Float32, 3, 8)
        y = Flux.onehotbatch(rand(1:2, 8), 1:2)
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        # Step 0: covariance + inv computed, S_l initialised from eigenvalues
        _, model = ekfac_step!(opt, model, loss_fn, x, Float32.(y))
        @test haskey(opt.S_l, 1)
        @test haskey(opt.S_l, 2)

        # Step 1: TScal triggers scaling update
        _, model = ekfac_step!(opt, model, loss_fn, x, Float32.(y))
        @test opt.steps == 2
    end
end
