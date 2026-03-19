@testset "Utility Functions" begin
    @testset "update_running_stat!" begin
        # Test basic EMA behaviour
        running = ones(3, 3)
        new_stat = 2.0 * ones(3, 3)
        decay = 0.9

        update_running_stat!(new_stat, running, decay)
        expected = 0.9 * ones(3,3) + 0.1 * 2.0 * ones(3,3)  # = 1.1
        @test running ≈ expected atol=1e-10

        # Repeated updates should converge towards the new stat
        for _ in 1:1000
            update_running_stat!(new_stat, running, decay)
        end
        @test running ≈ new_stat atol=1e-4

        # Test with decay = 0 (instant update)
        running = ones(2, 2)
        new_stat = 5.0 * ones(2, 2)
        # decay close to 0 → converges quickly
        for _ in 1:100
            update_running_stat!(new_stat, running, 0.01)
        end
        @test running ≈ new_stat atol=1e-2
    end

    @testset "extract_patches" begin
        # Simple 1-channel, 4x4 image with known values
        x = reshape(Float64.(1:16), 4, 4, 1, 1)

        # 2x2 kernel, stride 1, no padding
        patches = extract_patches(x, (2, 2), (1, 1), (0, 0))
        @test size(patches) == (4, 9)  # 4 = 1*2*2, 9 = 3*3*1

        # 2x2 kernel, stride 2, no padding
        patches = extract_patches(x, (2, 2), (2, 2), (0, 0))
        @test size(patches) == (4, 4)  # 4 patches

        # With padding
        patches = extract_patches(x, (3, 3), (1, 1), (1, 1))
        @test size(patches) == (9, 16)  # 9 = 1*3*3, 16 = 4*4*1

        # Multi-channel
        x_mc = randn(8, 8, 3, 2)
        patches = extract_patches(x_mc, (3, 3), (1, 1), (0, 0))
        @test size(patches, 1) == 3 * 3 * 3  # 27
        @test size(patches, 2) == 6 * 6 * 2  # 72

        # Verify patch content
        x_simple = reshape(Float64.(1:4), 2, 2, 1, 1)
        patches = extract_patches(x_simple, (2, 2), (1, 1), (0, 0))
        @test size(patches) == (4, 1)
        @test patches[:, 1] == vec(x_simple)
    end

    @testset "compute_cov_a - Dense" begin
        layer = Dense(4 => 3)

        # Input as (batch, features) matrix
        a = randn(8, 4)  # 8 samples, 4 features

        cov = compute_cov_a(a, layer)
        @test size(cov) == (5, 5)  # 4 + 1 for bias
        @test issymmetric(round.(cov; digits=10))

        # Without bias
        layer_nobias = Dense(4 => 3; bias=false)
        cov_nb = compute_cov_a(a, layer_nobias)
        @test size(cov_nb) == (4, 4)
        @test issymmetric(round.(cov_nb; digits=10))

        # Verify it's a valid covariance (PSD)
        eigenvalues = eigvals(Symmetric(cov))
        @test all(eigenvalues .>= -1e-10)
    end

    @testset "compute_cov_a - Conv" begin
        layer = Conv((3, 3), 2 => 4)

        a = randn(Float32, 8, 8, 2, 4)  # W=8, H=8, C=2, N=4
        cov = compute_cov_a(a, layer)
        # Expected: (2*3*3 + 1) = 19, since bias
        @test size(cov) == (19, 19)

        layer_nb = Conv((3, 3), 2 => 4; bias=false)
        cov_nb = compute_cov_a(a, layer_nb)
        @test size(cov_nb) == (18, 18)
    end

    @testset "compute_cov_g - Dense" begin
        layer = Dense(4 => 3)

        g = randn(8, 3)  # 8 samples, 3 output features

        # batch_averaged = true
        cov_ba = compute_cov_g(g, layer; batch_averaged=true)
        @test size(cov_ba) == (3, 3)

        # batch_averaged = false
        cov_nba = compute_cov_g(g, layer; batch_averaged=false)
        @test size(cov_nba) == (3, 3)

        # Both should be PSD
        @test all(eigvals(Symmetric(cov_ba)) .>= -1e-10)
        @test all(eigvals(Symmetric(cov_nba)) .>= -1e-10)
    end

    @testset "compute_cov_g - Conv" begin
        layer = Conv((3, 3), 2 => 4)

        g = randn(Float32, 6, 6, 4, 2)  # W=6, H=6, C_out=4, N=2
        cov = compute_cov_g(g, layer; batch_averaged=true)
        @test size(cov) == (4, 4)
    end

    @testset "compute_mat_grad - Dense" begin
        layer = Dense(4 => 3)
        input = randn(8, 4)
        grad_output = randn(8, 3)

        grad = compute_mat_grad(input, grad_output, layer)
        # Shape: (out_dim, in_dim+1, batch)
        @test size(grad) == (3, 5, 8)

        layer_nb = Dense(4 => 3; bias=false)
        grad_nb = compute_mat_grad(input, grad_output, layer_nb)
        @test size(grad_nb) == (3, 4, 8)
    end

    @testset "compute_mat_grad - Conv" begin
        layer = Conv((3, 3), 2 => 4)
        input = randn(Float32, 8, 8, 2, 2)
        grad_output = randn(Float32, 6, 6, 4, 2)

        grad = compute_mat_grad(input, grad_output, layer)
        # (out_c, in_c*kw*kh + 1, batch)
        @test size(grad) == (4, 19, 2)
    end
end
