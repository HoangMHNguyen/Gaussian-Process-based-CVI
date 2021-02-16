"""
All functions are used for 2-D output vector f
"""

function SE(x1, x2, σ_f, l)
    """
    Squared Exponential kernel k(x,x')
    x1 and x2 are 2 input points
    σ_l: factor that governs the width of the uncertainty of factor
    l: length-scale
    """
    return σ_f^2 * exp(-norm(x1-x2,2)^2/(2*l^2))

end

"""
k1 = SE(x1,x2,σ_l1,l1)
k2 = SE(x1,x2,σ_l2,l2)
"""
function cov_MGP(A1, A2, K1, K2)
    return kron(A1, K1) + kron(A2, K2)
end

function Kff_observation(X, A1, A2, σ_f, l)
    """
    This function computes the Covariance matrix Kff after observing N data points
    X: DxN input matrix, containing N observation; D is the dimension of 1 input point
    A1, A2: correlation matrix
    σ_f: vector containing the scale factors for 2 GPs u1 and u2
    l: vector containing the length scale for 2 GPs u1 and u2
    The result is a symmetric matrix
    Only use this function to compute Kyy, K**, whose inputs are the same

    """
    K1 = zeros(size(X,2),size(X,2));
    K2 = zeros(size(X,2),size(X,2));
    for i=1:size(X,2)
        for j=1:size(X,2)
            K1[i,j] = SE(X[:,i],X[:,j],σ_f[1],l[1]); # kernel for 1st GP
            K2[i,j] = SE(X[:,i],X[:,j],σ_f[2],l[2]); # kernel for 2nd GP
        end
    end
    # now we have K1, K2 with size NxN
    K_mgp = cov_MGP(A1, A2, K1, K2); # the covariance matrix for the multivariate GP of latent function vector f
    return K_mgp
end


function K_cross(X1,X2,A1,A2,σ_f,l)
    """
    This function computes cross-covariance matrix of observations Y and test value Y*
    X1 is observations
    X2 is test input points
    The remaining parameters are defined as above    
    """
    K1 = zeros(size(X1,2),size(X2,2));
    K2 = zeros(size(X1,2),size(X2,2));
    for i=1:size(X1,2)
        for j=1:size(X2,2)
            K1[i,j] = SE(X1[:,i],X2[:,j],σ_f[1],l[1]);
            K2[i,j] = SE(X1[:,i],X2[:,j],σ_f[2],l[2]);
        end
    end
    K_y_ynew = cov_MGP(A1, A2, K1, K2);
    return K_y_ynew
end

