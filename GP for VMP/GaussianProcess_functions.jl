include("Kernels.jl")
function GP(x_observed, y_observed, x_new, A1, A2, σ_f, l)
    """
    x_observed: input of training data (size DxN)
    y_observed: output of training data (size TxN)
    x_new: new input (size Dx1)
    A1, A2: covariance matrices in Linear model of coregionalization(size TxT)
    σ_f: vector of scale factors for 2 GPs u1 and u2
    l: length-scale vectors
    """
    N = size(x_observed,2);
    T = size(y_observed,1);
    K_η_observed = Kff_observation(x_observed,A1,A2,σ_f,l); # covariance matrix of N input data points
    K_η_new = Kff_observation(x_new,A1,A2,σ_f,l); # covariance matrix for new data point
    K_η_cross = K_cross(x_observed,x_new,A1,A2,σ_f,l); # cross-covariance between training and new data points 

    y_observed = reshape(y_observed,N*T);
    eye_matrix = 1.0* Matrix(I, size(K_η_observed,1),size(K_η_observed,1)); #identity matrix
    predictive_mean = K_η_cross'*inv(K_η_observed + 1e-6*eye_matrix)*y_observed;
    predictive_covmatrix = K_η_new - K_η_cross'*inv(K_η_observed + 1e-6*eye_matrix)*K_η_cross;

    return predictive_mean, predictive_covmatrix
end
