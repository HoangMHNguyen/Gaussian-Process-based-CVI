module BayesianUtilities

using LinearAlgebra, Distributions, Zygote, ForwardDiff, SpecialFunctions, StatsFuns
import Base.*, Base.+, Base.-, LinearAlgebra.\

include("utilities.jl")
include("info_measure.jl")
include("exp_family.jl")
include("expectations.jl")
include("inference_rules.jl")
include("optimizer.jl")
include("cvi.jl")


end
