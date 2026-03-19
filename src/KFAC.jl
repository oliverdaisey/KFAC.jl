module KFAC

using Flux
using Zygote
using LinearAlgebra
using Functors
using Statistics
using ChainRulesCore

include("utils.jl")
include("kfac_optimizer.jl")
include("ekfac_optimizer.jl")

export KFACOptimizer, EKFACOptimizer
export kfac_step!, ekfac_step!
export compute_cov_a, compute_cov_g, update_running_stat!
export extract_patches, compute_mat_grad

end
