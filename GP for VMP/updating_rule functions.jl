# Update-rule for natural parameters, only used for VMP-univariate Gaussian example 

# function to compute η of message mμ
function narparams_mu(p, x)
    """
    p: gamma distribution 
    x: 1 observation
    """
    mean_p = p.α * p.θ;
    η = Vector{Float64}(undef,2); # initialize the natural parameter vector

    η[1] = - mean_p/2;
    η[2] = mean_p * x;

    return η
end

#function to compute η of message mβ
function  narparams_beta(p,x)
    """
    p: univariate Gaussian distribution 
    x: 1 observation
    """
    mean_p = p.μ;
    var_p = p.σ^2
    η = Vector{Float64}(undef,2); # initialize the natural parameter vector

    η[1] = 1/2;
    η[2] = -1/2*x^2 + x*mean_p - 1/2*(var_p + mean_p^2);
    
    return η
end