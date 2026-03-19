@testset "Integration Tests" begin
    @testset "KFAC trains a simple classification task" begin
        # Create a linearly separable dataset
        Random.seed!(789)
        n_samples = 100
        X = randn(Float32, 4, n_samples)
        # True labels based on a simple linear rule
        true_w = randn(Float32, 4)
        labels = [dot(true_w, X[:, i]) > 0 ? 1 : 2 for i in 1:n_samples]
        Y = Float32.(Flux.onehotbatch(labels, 1:2))

        model = Chain(Dense(4 => 8, relu), Dense(8 => 2))
        opt = KFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=1, momentum=0.0)
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        initial_loss = loss_fn(model, X, Y)
        for i in 1:50
            _, model = kfac_step!(opt, model, loss_fn, X, Y)
        end
        final_loss = loss_fn(model, X, Y)

        @test final_loss < initial_loss
        @test final_loss < 0.5  # should be able to fit this simple task

        # Check accuracy
        preds = Flux.onecold(model(X))
        acc = mean(preds .== labels)
        @test acc > 0.7
    end

    @testset "EKFAC trains a simple classification task" begin
        Random.seed!(101)
        n_samples = 100
        X = randn(Float32, 4, n_samples)
        true_w = randn(Float32, 4)
        labels = [dot(true_w, X[:, i]) > 0 ? 1 : 2 for i in 1:n_samples]
        Y = Float32.(Flux.onehotbatch(labels, 1:2))

        model = Chain(Dense(4 => 8, relu), Dense(8 => 2))
        opt = EKFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=1, TScal=1, momentum=0.0)
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        initial_loss = loss_fn(model, X, Y)
        for i in 1:50
            _, model = ekfac_step!(opt, model, loss_fn, X, Y)
        end
        final_loss = loss_fn(model, X, Y)

        @test final_loss < initial_loss
    end

    @testset "KFAC with momentum" begin
        Random.seed!(202)
        model = Chain(Dense(3 => 5, relu), Dense(5 => 2))
        opt = KFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=1, momentum=0.9)

        x = randn(Float32, 3, 32)
        y = Float32.(Flux.onehotbatch(rand(1:2, 32), 1:2))
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        loss1, model = kfac_step!(opt, model, loss_fn, x, y)
        @test isfinite(loss1)

        # Momentum buffers should be populated
        @test !isempty(opt.momentum_bufs)

        for i in 1:10
            _, model = kfac_step!(opt, model, loss_fn, x, y)
        end
        @test opt.steps == 11
    end

    @testset "KFAC with weight decay" begin
        Random.seed!(303)
        model = Chain(Dense(3 => 5, relu), Dense(5 => 2))
        opt = KFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=1,
                            weight_decay=0.01, momentum=0.0)

        x = randn(Float32, 3, 16)
        y = Float32.(Flux.onehotbatch(rand(1:2, 16), 1:2))
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        # Weight decay doesn't kick in until step 20*TCov
        for i in 1:25
            _, model = kfac_step!(opt, model, loss_fn, x, y)
        end
        @test opt.steps == 25
        @test isfinite(loss_fn(model, x, y))
    end

    @testset "KFAC with no-bias layers" begin
        Random.seed!(404)
        model = Chain(Dense(4 => 3, relu; bias=false), Dense(3 => 2; bias=false))
        opt = KFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=1, momentum=0.0)

        x = randn(Float32, 4, 8)
        y = Float32.(Flux.onehotbatch(rand(1:2, 8), 1:2))
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        loss1, model = kfac_step!(opt, model, loss_fn, x, y)
        @test isfinite(loss1)

        for i in 1:10
            _, model = kfac_step!(opt, model, loss_fn, x, y)
        end
        @test loss_fn(model, x, y) < loss1
    end

    @testset "Model parameters actually change" begin
        Random.seed!(505)
        model = Chain(Dense(3 => 4, relu), Dense(4 => 2))
        opt = KFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=1, momentum=0.0)

        x = randn(Float32, 3, 8)
        y = Float32.(Flux.onehotbatch(rand(1:2, 8), 1:2))
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        w_before = copy(model.layers[1].weight)
        _, model = kfac_step!(opt, model, loss_fn, x, y)
        w_after = model.layers[1].weight
        @test w_before != w_after
    end

    @testset "TCov and TInv scheduling" begin
        model = Chain(Dense(2 => 3, relu), Dense(3 => 2))
        opt = KFACOptimizer(lr=0.01, damping=0.1, TCov=5, TInv=10, momentum=0.0)

        x = randn(Float32, 2, 4)
        y = Float32.(Flux.onehotbatch(rand(1:2, 4), 1:2))
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        # Step 0: TCov and TInv should both trigger
        _, model = kfac_step!(opt, model, loss_fn, x, y)
        @test haskey(opt.m_aa, 1)  # covariance computed at step 0

        # Steps 1-4: no covariance update
        for i in 1:4
            _, model = kfac_step!(opt, model, loss_fn, x, y)
        end
        @test opt.steps == 5
    end

    @testset "Multiple runs produce finite results" begin
        for seed in [1, 42, 100, 999]
            Random.seed!(seed)
            model = Chain(Dense(5 => 4, relu), Dense(4 => 3))
            opt = KFACOptimizer(lr=0.01, damping=0.1, TCov=1, TInv=1, momentum=0.0)

            x = randn(Float32, 5, 16)
            y = Float32.(Flux.onehotbatch(rand(1:3, 16), 1:3))
            loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

            for i in 1:10
                l, model = kfac_step!(opt, model, loss_fn, x, y)
                @test isfinite(l)
            end
        end
    end
end
