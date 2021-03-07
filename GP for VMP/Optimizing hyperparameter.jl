##### Optimization
include("Kernels.jl")
function log_mll(K,y,A1,A2,σ_f,l)
    """
    K: matrix of norms ||x-x'||^2
    y: observations ((NxT)x1)
    A1, A2: correlation matrices between output dimensions
    σ_f: vector of std of kernels
    l: vector of lengthscale of kernels
    """
    K1 = σ_f[1]^2 *exp.(-K./(2*l[1]^2));
    K2 = σ_f[2]^2 *exp.(-K./(2*l[2]^2));
    I_n =1* Matrix(I, size(K,2),size(K,2)); #identity matrix
    K1 = K1 + 1e-6*I_n;
    K2 = K2 + 1e-6*I_n;
    Kff = cov_MGP(A1, A2, K1, K2); # done covariance matrix for f after N observations

    I_nd = 1* Matrix(I, size(Kff,2),size(Kff,2));
    Kff = Kff + 1e-5*I_nd;

    log_mll = 1/2*y'*inv(Kff)*y + 1/2*log(det(Kff)+1e-6); #negative log-likelyhood, we want to minimize this
    
    return log_mll
end

#optimize using Zygote 
function optim_params(η,num_itr,x_observed,y_observed,A1,A2,σ_f_val,l_val)
    """
    x_observed: input of training data points (DxN)
    y_observed: output of training data points (TxN)
    A1, A2: covariance matrices between output dimensions
    σ_f_val, l_val: vectors of hyperparameters that we want to optimize 
    """
    N = size(x_observed,2);
    T = size(y_observed,1);
    # create matrix of norms ||x-x'||^2
    K = zeros(N,N);
    for i=1:size(x_observed,2)
        for j=1:size(x_observed,2)
            K[i,j] = norm(x_observed[:,i] - x_observed[:,j],2)^2; 
        end
    end
    y_observed_resha = vcat(y_observed'...)

    for iter = 1:num_itr
        old_σ_f_val = σ_f_val; #old value of σ_f
        old_l_val = l_val; #old value of l
        grads = gradient(σ_f_val,l_val) do σ_f, l
            log_mll(K,y_observed_resha, A1, A2, σ_f,l)
        end
        σ_f_val = σ_f_val .- η*grads[1];
        l_val = l_val .- η*grads[2];
        if norm(σ_f_val-old_σ_f_val,1) < 1e-18 && norm(l_val-old_l_val,1) < 1e-18
            break;
        end
    end
    return σ_f_val, l_val, log_mll(K,y_observed_resha, A1, A2, σ_f_val,l_val)
end
