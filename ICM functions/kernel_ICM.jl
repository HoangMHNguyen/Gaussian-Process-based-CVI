#######################################
# This is the basic file containing functions
# that compute auto-covariance and cross-covariance matrices for ICM.
# Note that this file only takes account of SE kernel. 

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
    x's are column vectors
    σ and l are scalars
    The function returns a scalar value showing correlation between x1 and x2 (k12)
    """ 
    return σ^2 * exp(-norm(x1-x2,2)^2/(2*l^2))
end

function SE(x1::Vector, x2::Vector, σ::Real, l::Vector)
    """
    x's are column vectors
    l is a vector of length scales
    σ is scalar
    The function returns a scalar value showing correlation between x1 and x2 (k12)
    """
    return σ^2 * exp((-1/2 * transpose(x1 - x2)*Diagonal(l.^2)^(-1)*(x1 - x2)))
end

#### auto-covariance matrix for ICM ###
function autocov_mat(X::Matrix, A::Matrix, σ::Real, l::Real)
    """
    The function computes the auto-covariance matrix K(X,X)
    X is the input matrix, size dxN (d: dimensionality of the input, N: number of observations)
    A is the coregionalization matrix, size DxD, where D is the dimensionality of the output
    The function returns a matrix that has the size of DN x DN
    """
    k = sum(norm.(X,2).^2,dims=1)' .- 2*X'X .+ sum(norm.(X,2).^2,dims=1); 
    k =  σ^2*exp.(-k/(2*l^2)); # size of k is NxN

    return kron(A, k)
end

function autocov_mat(X::Matrix, A::Matrix, σ::Real, l::Vector)
    """
    The function computes the auto-covariance matrix K(X,X)
    X is the input matrix, size dxN (d: dimensionality of the input, N: number of observations)
    A is the coregionalization matrix, size DxD, where D is the dimensionality of the output
    The function returns a matrix that has the size of DN x DN
    """
    λ = Diagonal(l.^2); #diagonal matrix whose elements are the squared length-scale on each dimension of input space
    k = sum(transpose(norm.(X,2).^2) * λ^(-1), dims=2) .- 2*transpose(X)*λ^(-1)*X .+ transpose(sum(transpose(norm.(X,2).^2) * λ^(-1), dims=2));
    k = σ^2 * exp.(-1/2*k);
    return kron(A,k)
end

function autocov_mat(X::Vector, A::Matrix, σ::Real, l::Real)
    """
    The function computes the auto-covariance matrix K(X,X)
    X is a input vector
    A is the coregionalization matrix, size DxD, where D is the dimensionality of the output
    The function returns a matrix that has the size of D x D
    """
    k = SE(X,X,σ,l);
    return A*k
end

function autocov_mat(X::Vector, A::Matrix, σ::Real, l::Vector)
    """
    The function computes the auto-covariance matrix K(X,X)
    X is the input vector
    A is the coregionalization matrix, size DxD, where D is the dimensionality of the output
    The function returns a matrix that has the size of D x D
    """
    k = SE(X,X,σ,l);
    return A*k
end


########## cross-covariance matrix
function cross_cov_mat(X1::Matrix, X2::Matrix, A::Matrix, σ::Real, l::Real)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    Both X1 and X2 are two matrices, size (dxN1) and (dxN2) respectively
    A is the coregionalization matrix, size DxD
    The matrix K will have the size of DN1 x DN2
    """
    k_cross = sum(norm.(X1,2).^2,dims=1)' .- 2*X1'X2 .+ sum(norm.(X2,2).^2,dims=1);
    k_cross =  σ^2*exp.(-k_cross/(2*l^2));

    return kron(A,k_cross)
end

function cross_cov_mat(X1::Matrix, X2::Matrix, A::Matrix, σ::Real, l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    Both X1 and X2 are two matrices, size (dxN1) and (dxN2) respectively
    A is the coregionalization matrix, size DxD
    The matrix K will have the size of DN1 x DN2
    """
    λ = Diagonal(l.^2);
    k_cross = sum(transpose(norm.(X1,2).^2) * λ^(-1), dims=2) .- 2*X1'*λ^(-1)*X2 .+ sum(transpose(norm.(X2,2).^2) * λ^(-1), dims=2)';
    k_cross = σ^2 * exp.(-1/2*k_cross);

    return kron(A,k_cross)
end

function cross_cov_mat(X1::Vector, X2::Matrix, A::Matrix, σ::Real, l::Real)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a vector (dx1) 
    X2 is a matrix (dxN2)
    The matrix K will have the size of D x DN2
    """
    k_cross = sum(norm.(X1,2).^2,dims=1)' .- 2*X1'*X2 .+ sum(norm.(X2,2).^2,dims=1);
    k_cross = σ^2*exp.(-k_cross/(2*l^2));

    return kron(A,k_cross) 
end

function cross_cov_mat(X1::Vector, X2::Matrix, A::Matrix, σ::Real, l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a vector (dx1) 
    X2 is a matrix (dxN2)
    The matrix K will have the size of D x DN2
    """
    λ = Diagonal(l.^2);
    k_cross = sum(transpose(norm.(X1,2).^2) * λ^(-1), dims=2) .- 2*X1'*λ^(-1)*X2 .+ sum(transpose(norm.(X2,2).^2) * λ^(-1), dims=2)';
    k_cross =  σ^2 * exp.(-1/2*k_cross);

    return kron(A,k_cross) 
end

function cross_cov_mat(X1::Matrix, X2::Vector, A::Matrix, σ::Real,l::Real)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a vector (dxN1) 
    X2 is a matrix (dx1)
    The matrix K will have the size of DN1 x D
    """
    k_cross = sum(norm.(X1,2).^2,dims=1)' .- 2*X1'*X2 .+ sum(norm.(X2,2).^2,dims=1);
    k_cross =  σ^2*exp.(-k_cross/(2*l^2));

    return kron(A,k_cross) 
end

function cross_cov_mat(X1::Matrix, X2::Vector, A::Matrix, σ::Real, l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a vector (dxN1) 
    X2 is a matrix (dx1)
    The matrix K will have the size of DN1 x D
    """
    λ = Diagonal(l.^2);
    k_cross = sum(transpose(norm.(X1,2).^2) * λ^(-1), dims=2) .- 2*X1'*λ^(-1)*X2 .+ sum(transpose(norm.(X2,2).^2) * λ^(-1), dims=2)';
    k_cross =  σ^2 * exp.(-1/2*k_cross);

    return kron(A,k_cross)
end

function cross_cov_mat(X1::Vector, X2::Vector, A::Matrix, σ::Real, l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a vecor dx1
    X2 is a vector dx1
    The matrix K will have the size of D x D
    """
    k_cross = SE(X1,X2,σ,l);

    return A*k_cross
end



