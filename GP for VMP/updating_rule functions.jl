# Update-rule for natural parameters, only used for VMP-univariate Gaussian example 

# function to compute η of message mμ
function narparams_mu(p, x)
    """
    p: gamma distribution 
    x: array of observations
    """
    mean_p = p.α * p.θ;
    η = Vector{Float64}(undef,2); # initialize the natural parameter vector

    η[1] = mean_p * sum(x);
    η[2] = - (length(x) * mean_p)/2;

    return η
end

#function to compute η of message mβ
function  naparams_beta(p,x)
    """
    p: univariate Gaussian distribution 
    x: array of observation
    """
    mean_p = p.μ;
    var_p = p.σ^2
    η = Vector{Float64}(undef,2); # initialize the natural parameter vector

    η[1] = -sum(x.^2) + 2*mean_p*sum(x) - length(x)*(var_p + mean_p^2);
    η[2] = length(x)/2;

    return η
end

#### update posterior
