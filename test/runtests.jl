using KFAC
using Test
using Flux
using LinearAlgebra
using Statistics
using Random

Random.seed!(42)

@testset "KFAC.jl" begin
    include("test_utils.jl")
    include("test_kfac.jl")
    include("test_ekfac.jl")
    include("test_integration.jl")
end
