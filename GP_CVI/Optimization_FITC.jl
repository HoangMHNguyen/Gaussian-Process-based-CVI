include("kernel function.jl")
include("FITC.jl")

"""
This file contains functions for:
    - Computing the negative log-likelihood function 
    - Optimizing parameters including: inducing points and hyper-parameters of kernel function
"""

##### Compute the negative log-likelihood ###########
function neg_llh(Xf::Matrix,Xu::Matrix,f::Vector,σ::Real,l::Real)
    """
    This function computes the negative log-likelihood value.
    Xf, f: our observations (size DxN and Nx1, respectively)   
    Xu: inducing points (size DxU, U is the number of inducing points)
    σ and l: parameters of kernel functions (scalars)
    """
    Kff = autocov_mat(Xf,σ,l); #auto-covariance matrix of training points
    Kuu = autocov_mat(Xu,σ,l); #auto-covariance matrix of inducing points

    eye_ff = 1.0* Matrix(I, size(Kff,1),size(Kff,1));
    eye_uu = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
    Kff = Kff + 1e-5*eye_ff; #add noise for stable computation
    Kuu = Kuu + 1e-5*eye_uu; #add noise for stable computation

    Kuf = cross_cov_mat(Xu,Xf,σ,l); #cross-covariance matrix of inducing and training points
    Kfu = cross_cov_mat(Xf,Xu,σ,l);
    
    Λff = Diagonal(Kff - Kfu*Kuu^(-1)*Kuf);
    
    #compute the negative log-evidence
    neglog_lh = 1/2*log(1e-6 + det(Kuu + Kuf*Λff^(-1)*Kfu)*det(Λff)/det(Kuu)) + 1/2*f'*(Λff^(-1) - Λff^(-1)*Kfu*(Kuu + Kuf*Λff^(-1)*Kfu)^(-1)*Kuf*Λff^(-1))*f;
    
    return neglog_lh
end

function neg_llh(Xf::Matrix,Xu::Matrix,f::Vector,σ::Real,l::Vector)
    """
    This function computes the negative log-likelihood value.
    Xf, f: our observations (size DxN, Nx1 respectively)   
    Xu: inducing points (size DxU)
    σ and l: parameters of kernel functions. σ is scalar; l is a column vector with D elements
    """
    Kff = autocov_mat(Xf,σ,l); #auto-covariance matrix of training points
    Kuu = autocov_mat(Xu,σ,l); #auto-covariance matrix of inducing points

    eye_ff = 1.0* Matrix(I, size(Kff,1),size(Kff,1));
    eye_uu = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
    Kff = Kff + 1e-5*eye_ff; #add noise for stable computation
    Kuu = Kuu + 1e-5*eye_uu; #add noise for stable computation
    
    Kuf = cross_cov_mat(Xu,Xf,σ,l); #cross-covariance matrix of inducing and training points
    Kfu = cross_cov_mat(Xf,Xu,σ,l);
    
    Λff = Diagonal(Kff - Kfu*Kuu^(-1)*Kuf);
    
    #compute the negative log-evidence
    neglog_lh = 1/2*log(1e-6 + det(Kuu + Kuf*Λff^(-1)*Kfu)*det(Λff)/det(Kuu)) + 1/2*f'*(Λff^(-1) - Λff^(-1)*Kfu*(Kuu + Kuf*Λff^(-1)*Kfu)^(-1)*Kuf*Λff^(-1))*f;
    
    return neglog_lh
end

##### Optimizing function ######
function optim_FITC(Xf::Matrix,Xu_val::Matrix,f_observed::Vector,σ_val::Real,l_val::Real,η::Real,num_itr::Int)
    """
    This function is for optimizing inducing points Xu and hyper-parameters σ and l
    Xf, f_observed: our observations (size DxN and Nx1, respectively)
    Xu_val: initial inducing points (size DxU)
    σ_val, l_val: parameters of kernel function (scalars)
    η: step size for gradient descent
    num_itr: number of iterations for gradient descent
    """
    for iter = 1:num_itr
        old_Xu_val = Xu_val; #store old values of inducing points Xu
        old_σ_val = σ_val; #store old values of σ
        old_l_val = l_val; #store old values of l
        
        # get gradient 
        grads = gradient(Xu_val,σ_val,l_val) do Xu, σ, l
            neg_llh(Xf,Xu,f_observed,σ,l)
        end
        
        #Update inducing points and parameters
        Xu_val = Xu_val - η*grads[1];
        σ_val = σ_val - η*grads[2];
        l_val = l_val - η*grads[3];
    end
    
    return Xu_val, σ_val, l_val, neg_llh(Xf,Xu_val,f_observed,σ_val,l_val)
end


function optim_FITC(Xf::Matrix,Xu_val::Matrix,f_observed::Vector,σ_val::Real,l_val::Vector,η::Real,num_itr::Int)
    """
    This function is for optimizing inducing points Xu and hyper-parameters σ and l
    Xf, f_observed: our observations (size DxN and Nx1, respectively)
    Xu_val: initial inducing points (size DxU)
    σ_val, l_val: parameters of kernel function. σ_val is scalar; l_val is a column vector with D elements
    η: step size for gradient descent
    num_itr: number of iterations for gradient descent
    """
    for iter = 1:num_itr
        old_Xu_val = Xu_val; #store old values of inducing points Xu
        old_σ_val = σ_val; #store old values of σ
        old_l_val = l_val; #store old values of l
        
        # get gradient 
        grads = gradient(Xu_val,σ_val,l_val) do Xu, σ, l
            neg_llh(Xf,Xu,f_observed,σ,l)
        end
        
        #Update inducing points and parameters
        Xu_val = Xu_val - η*grads[1];
        σ_val = σ_val - η*grads[2];
        l_val = l_val - η*grads[3];
    end
    
    return Xu_val, σ_val, l_val, neg_llh(Xf,Xu_val,f_observed,σ_val,l_val)
end
