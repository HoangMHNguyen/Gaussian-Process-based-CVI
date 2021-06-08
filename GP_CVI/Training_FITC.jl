
include("kernel function.jl")
include("FITC.jl")
include("Optimization_FITC.jl")

### Training ###
""" In training phase, we need to:
   - Optimize hyper-parameters and inducing points
   - Get the posterior of inducing points. We will use this info to predict new values

**Remember: X always has the shape DxN, even when X is 1-D (in such case, X will be a row vector 1xN.)
"""
function train_FITC(Xf::Matrix, Xu_init::Matrix, f_observed::Vector, σ_init::Real, l_init::Real, η::Real, num_itr::Int)
    #Perform optimization to get optimal inducing points and parameters
    Xu_op, σ_op, l_op, llh = optim_FITC(Xf, Xu_init, f_observed, σ_init, l_init, η, num_itr); 

    #Now get the necessary info for prediction
    μ_u, Σ_u = FITC_post_indu(Xf, Xu_op, f_observed, σ_op, l_op);

    return Xu_op, σ_op, l_op, μ_u, Σ_u, llh 
end 

function train_FITC(Xf::Matrix, Xu_init::Matrix, f_observed::Vector, σ_init::Real, l_init::Vector, η::Real, num_itr::Int)
    #Perform optimization to get optimal inducing points and parameters
    Xu_op, σ_op, l_op, llh = optim_FITC(Xf, Xu_init, f_observed, σ_init, l_init, η, num_itr); 

    #Now get the necessary info for prediction
    μ_u, Σ_u = FITC_post_indu(Xf, Xu_op, f_observed, σ_op, l_op);

    return Xu_op, σ_op, l_op, μ_u, Σ_u, llh 
end 