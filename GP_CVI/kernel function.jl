#######################################
# This is the basic file containing functions
# for computing auto-covariance and cross-covariance matrices. 
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
    return σ^2 * exp((-1/2 * (x1 - x2)'*Diagonal(l.^2)^(-1)*(x1 - x2))[1])
end

############# auto-covariance matrix
function autocov_mat(X::Matrix, σ::Real, l::Real)
    """
    The function computes the auto-covaraince matrix K(X,X)
    X is a input matrix, size DxN (D: dimension, N: number of observations)
    The matrix K will have the size of NxN
    """
    #N = size(X,2);
    #K = zeros(N,N);
    #for i=1:N
    #    for j=1:N
    #        K[i,j] = SE(X[:,i],X[:,j],σ,l);
    #    end
    #end
    K = sum(norm.(X,2).^2,dims=1)' .- 2*X'X .+ sum(norm.(X,2).^2,dims=1);
    K =  σ^2*exp.(-K/(2*l^2)); 
    return K 
end

function autocov_mat(X::Matrix, σ::Real, l::Vector)
    """
    The function computes the auto-covaraince matrix K(X,X)
    X is a input matrix, size DxN (D: dimension, N: number of observations)
    The matrix K will have the size of NxN
    """
    #N = size(X,2);
    #K = zeros(N,N);
    #for i=1:N
    #    for j=1:N
    #        K[i,j] = SE(X[:,i],X[:,j],σ,l);
    #    end
    #end
    λ = Diagonal(l.^2); #diagonal matrix whose elements are squared length-scale on each dimension
    K = sum((norm.(X,2).^2)' * λ^(-1), dims=2) .- 2*X'*λ^(-1)*X .+ sum((norm.(X,2).^2)' * λ^(-1), dims=2)';
    K = σ^2 * exp.(-1/2*K);
    return K 
end

function autocov_mat(X::Vector, σ::Real, l::Real)
    """
    The function computes the auto-covaraince matrix K(X,X)
    X is a input vector, size Dx1 (D: dimension)
    K is scalar
    """
    K = SE(X,X,σ,l);
    return K 
end

function autocov_mat(X::Vector, σ::Real, l::Vector)
    """
    The function computes the auto-covaraince matrix K(X,X)
    X is a input vector, size Dx1 (D: dimension)
    K is a scalar
    """
    K = SE(X,X,σ,l);
    return K 
end

########## cross-covariance matrix
function cross_cov_mat(X1::Matrix, X2::Matrix,σ::Real,l::Real)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    Both X1 and X2 are two matrices, size (DxN1) and (DxN2) respectively
    The matrix K will have the size of N1xN2
    """
    #N1 = size(X1,2);
    #N2 = size(X2,2);
    #K_cross = zeros(N1,N2);
    #for i=1:N1
    #    for j=1:N2
    #        K_cross[i,j] = SE(X1[:,i],X2[:,j],σ,l)
    #    end
    #end
    K_cross = sum(norm.(X1,2).^2,dims=1)' .- 2*X1'X2 .+ sum(norm.(X2,2).^2,dims=1);
    K_cross =  σ^2*exp.(-K_cross/(2*l^2));
    return K_cross
end

function cross_cov_mat(X1::Matrix, X2::Matrix,σ::Real,l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    Both X1 and X2 are two matrices, size (DxN1) and (DxN2) respectively
    The matrix K will have the size of N1xN2
    """
    #N1 = size(X1,2);
    #N2 = size(X2,2);
    #K_cross = zeros(N1,N2);
    #for i=1:N1
        #for j=1:N2
            #K_cross[i,j] = SE(X1[:,i],X2[:,j],σ,l)
        #end
    #end

    λ = Diagonal(l.^2);
    K_cross = sum((norm.(X1,2).^2)' * λ^(-1), dims=2) .- 2*X1'*λ^(-1)*X2 .+ sum((norm.(X2,2).^2)' * λ^(-1), dims=2)';
    K_cross = σ^2 * exp.(-1/2*K_cross);
    return K_cross
end

function cross_cov_mat(X1::Vector, X2::Matrix,σ::Real,l::Real)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a vector 1-D 
    X2 is a matrix (DxN2)
    K now is a row vector, size of 1xN2
    """
    N2 = size(X2,2);
    K_cross = zeros(1,N2);
    for i=1:N2
        K_cross[i] = SE(X1,X2[:,i],σ,l)
    end
    return K_cross
end

function cross_cov_mat(X1::Vector, X2::Matrix,σ::Real,l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    Both X1 and X2 are two matrices, size (DxN1) and (DxN2) respectively
    K will be a row vector, size 1xN2
    """
    N2 = size(X2,2);
    K_cross = zeros(1,N2);
    for i=1:N2
        K_cross[i] = SE(X1,X2[:,i],σ,l)
    end
    return K_cross
end

function cross_cov_mat(X1::Matrix, X2::Vector,σ::Real,l::Real)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a matrix DxN1
    X2 is a vector (Dx1)
    K now is a column vector, size of N1x1
    """
    N1 = size(X1,2);
    K_cross = zeros(N1,1);
    for i=1:N1
        K_cross[i] = SE(X1[:,i],X2,σ,l)
    end
    return K_cross
end

function cross_cov_mat(X1::Matrix, X2::Vector,σ::Real,l::Vector)
    """
    This function computes the cross-covariance matrices K(X1,X2)
    X1 is a matrix DxN1
    X2 is a vector (Dx1)
    K now is a column vector, size of N1x1
    """
    N1 = size(X1,2);
    K_cross = zeros(N1,1);
    for i=1:N1
        K_cross[i] = SE(X1[:,i],X2,σ,l)
    end
    return K_cross
end
