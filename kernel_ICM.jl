#######################################
# This is the basic file containing functions
# that compute auto-covariance and cross-covariance matrices for multivariate GP.
# Note that this file only takes account of SE kernel. 

# We also assume that there are 2 latent function u.
#######################################

######## Kernel function #######
function SE(x1::Real, x2::Real, σ::Real, l::Real)
    """
    all parameters are scalars
    The function returns a scalar value showing correlation between x1 and x2 (k12)
    """
    return σ^2 * exp(-(x1-x2)^2/(2*l^2))
end

function SE(x1::Vector, x2::Vector, σ::Real, l::Real)
    """
    x now is a column vector
    σ and l are real numbers
    The function returns a scalar value showing correlation between x1 and x2 (k12)
    """ 
    return σ^2 * exp(-norm(x1-x2,2)^2/(2*l^2))
end

function SE(x1::Vector, x2::Vector, σ::Real, l::Vector)
    """
    x is a column vector
    l is a vector of length scales
    σ is real
    The function returns a scalar value showing correlation between x1 and x2 (k12)
    """
    return σ^2 * exp((-1/2 * transpose(x1 - x2)*Diagonal(l.^2)^(-1)*(x1 - x2))[1])
end

#### auto-covariance matrix for multivariateGP ###
function autocov_mat(X::Matrix, A::Matrix, σ::Real, l::Real)
    """
    The function computes the auto-covaraince matrix K(X,X)
    X is a input matrix, size DxN (D: dimension, N: number of observations)
    A1 and A2 are the 2 coregionalization matrices, size TxT
    The matrix K will have the size of NxN
    """
    K = sum(norm.(X,2).^2,dims=1)' .- 2*X'X .+ sum(norm.(X,2).^2,dims=1); 
    K =  σ^2*exp.(-K/(2*l^2)); #here, size of K1 is NxN

    return kron(A, K)
end

function autocov_mat(X::Matrix, A::Matrix, σ::Real, l::Vector)
    """
    The function computes the auto-covaraince matrix K(X,X)
    X is a input matrix, size DxN (D: dimension, N: number of observations)
    The matrix K will have the size of NxN
    """
    λ = Diagonal(l.^2); #diagonal matrix whose elements are the squared length-scale on each dimension of input space
    K = sum(transpose(norm.(X,2).^2) * λ^(-1), dims=2) .- 2*transpose(X)*λ^(-1)*X .+ transpose(sum(transpose(norm.(X,2).^2) * λ^(-1), dims=2));
    K = σ^2 * exp.(-1/2*K);
    return kron(A,K)
end

function autocov_mat(X::Vector, A::Matrix, σ::Real, l::Real)
    """
    The function computes the auto-covaraince matrix K(X,X)
    X is a input vector, size Dx1 (D: dimension)
    K is scalar
    """
    K = SE(X,X,σ,l);
    return kron(A,K)
end

function autocov_mat(X::Vector, A::Matrix, σ::Real, l::Vector)
    """
    The function computes the auto-covaraince matrix K(X,X)
    X is a input vector, size Dx1 (D: dimension)
    K is a scalar
    """
    K = SE(X,X,σ,l);
    return kron(A,K)
end


########## cross-covariance matrix
function cross_cov_mat(X1::Matrix, X2::Matrix, A::Matrix, σ::Real, l::Real)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    Both X1 and X2 are two matrices, size (DxN1) and (DxN2) respectively
    The matrix K will have the size of N1xN2
    """
    K_cross = sum(norm.(X1,2).^2,dims=1)' .- 2*X1'X2 .+ sum(norm.(X2,2).^2,dims=1);
    K_cross =  σ^2*exp.(-K_cross/(2*l^2));

    return kron(A,K_cross)
end

function cross_cov_mat(X1::Matrix, X2::Matrix, A::Matrix, σ::Real, l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    Both X1 and X2 are two matrices, size (DxN1) and (DxN2) respectively
    The matrix K will have the size of N1xN2
    """

    λ = Diagonal(l.^2);
    K_cross = sum(transpose(norm.(X1,2).^2) * λ^(-1), dims=2) .- 2*X1'*λ^(-1)*X2 .+ sum(transpose(norm.(X2,2).^2) * λ^(-1), dims=2)';
    K_cross = σ^2 * exp.(-1/2*K_cross);

    return kron(A,K_cross)
end

function cross_cov_mat(X1::Vector, X2::Matrix, A::Matrix, σ::Real, l::Real)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a vector (Dx1) 
    X2 is a matrix (DxN2)
    K now is a row vector, size of 1xN2
    """
    K_cross = sum(norm.(X1,2).^2,dims=1)' .- 2*X1'*X2 .+ sum(norm.(X2,2).^2,dims=1);
    K_cross = σ^2*exp.(-K_cross/(2*l^2));

    return kron(A,K_cross) 
end

function cross_cov_mat(X1::Vector, X2::Matrix, A::Matrix, σ::Real, l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a vector (Dx1) 
    X2 is a matrix (DxN2)
    K will be a row vector, size 1xN2
    """
    λ = Diagonal(l.^2);
    K_cross = sum(transpose(norm.(X1,2).^2) * λ^(-1), dims=2) .- 2*X1'*λ^(-1)*X2 .+ sum(transpose(norm.(X2,2).^2) * λ^(-1), dims=2)';
    K_cross =  σ^2 * exp.(-1/2*K_cross);

    return kron(A,K_cross) 
end

function cross_cov_mat(X1::Matrix, X2::Vector, A::Matrix, σ::Real,l::Real)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a matrix DxN1
    X2 is a vector (Dx1)
    K now is a column vector, size of N1x1
    """
    K_cross = sum(norm.(X1,2).^2,dims=1)' .- 2*X1'*X2 .+ sum(norm.(X2,2).^2,dims=1);
    K_cross =  σ^2*exp.(-K_cross/(2*l^2));

    return kron(A,K_cross) 
end

function cross_cov_mat(X1::Matrix, X2::Vector, A::Matrix, σ::Real, l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a matrix DxN1
    X2 is a vector (Dx1)
    K now is a column vector, size of N1x1
    """
    λ = Diagonal(l.^2);
    K_cross = sum(transpose(norm.(X1,2).^2) * λ^(-1), dims=2) .- 2*X1'*λ^(-1)*X2 .+ sum(transpose(norm.(X2,2).^2) * λ^(-1), dims=2)'
    K_cross =  σ^2 * exp.(-1/2*K_cross);

    return kron(A,K_cross)
end

function cross_cov_mat(X1::Vector, X2::Vector, A::Matrix, σ::Real, l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a matrix DxN1
    X2 is a vector (Dx1)
    K now is a column vector, size of N1x1
    """
    λ = Diagonal(l.^2);
    K_cross = sum(transpose(norm.(X1,2).^2) * λ^(-1), dims=2) .- 2*X1'*λ^(-1)*X2 .+ sum(transpose(norm.(X2,2).^2) * λ^(-1), dims=2)'
    K_cross =  σ^2 * exp.(-1/2*K_cross);

    return kron(A,K_cross)
end



