using Pkg
Pkg.activate("./workspace/")
Pkg.instantiate()

using LinearAlgebra, Distributions, Random, Statistics, SpecialFunctions
using Optim, Zygote
using Plots

pyplot()


include("Kernels.jl")
include("updating_rule functions.jl")
include("GaussianProcess_functions.jl")
include("Optimizing hyperparameter.jl")

# Generate observations
N = 15; #number of observations
μ_true = 3.0; #ground-truth of mean
β_true = 2.0; #ground-truth of precision

x_observed = sqrt(1/β_true)*randn(N) .+ μ_true; # collect N observations: x is drawn from Gaussian(μ_true,β_true⁻¹)
# the following information is need for updating messages
sum_x = sum(x_observed);
sum_x_squared = sum(x_observed.^2);

#### Define priors for μ and β
p_μ = Normal(0,100);
p_β = Gamma(0.01,0.1);
 
η_pμ = [p_μ.μ/(p_μ.σ^2), -1/(2*p_μ.σ^2)]; # natural parameters of μ prior
η_pβ = [-1/p_β.θ, p_β.α-1]; # natural parameters of β prior

###### Initialize posterior for μ and β
q_μ = Normal(0,50); 
q_β = Gamma(0.1, 0.1);

η_qμ = [q_μ.μ/(q_μ.σ^2), -1/(2*q_μ.σ^2)]; # natural parameters of μ posterior
η_qβ = [-1/q_β.θ, q_β.α-1]; # natural parameters of β posterior

"""
Create Gaussian processes for natural parameters
η_qμ = sum(η_prior + η_mμ)  => mean and variance for q(μ)
η_qβ = sum(η_prior + η_β)  => α and β for q(β)
After we obtain these posteriors, perform FE minimization 
Repeat the process again
"""

#### Run VMP to get observations.
input_η_mβ = []; #store inputs for mβ
η_mβ_observed = []; #store natural parameters of mβ 

input_η_mμ = []; #store inputs for mμ
η_mμ_observed = []; #store natural parameters of mμ


FE = []; #store FE values
n_iter = 50;

for i=1:n_iter
    ################  suppose we update q_β first #####################
    #η_mβ = [-sum_x_squared + 2*q_μ.μ*sum_x - N*(q_μ.σ^2 + q_μ.μ^2), N/2]; # natural parameters of m_β , collect μ and σ
    η_mβ = naparams_beta(q_μ,x_observed);
    append!(input_η_mβ,[q_μ.μ, q_μ.σ]); # collect μ and σ
    append!(η_mβ_observed, η_mβ); # collect corresponding observations of mβ

    # update natural parameters for β
    η_qβ = η_pβ + η_mβ;

    #convert from natural parameters to α and θ
    α_updated = η_qβ[2] + 1;
    θ_updated = -1/η_qβ[1]; 

    # update qβ
    q_β = Gamma(α_updated,θ_updated);

    ################# now we update q_μ #####################
    #η_mμ = [q_β.α*q_β.θ*sum_x, -N*q_β.α*q_β.θ/2]; 
    η_mμ = narparams_mu(q_β,x_observed);
    append!(input_η_mμ,[q_β.α, q_β.θ]); # collect α and θ
    append!(η_mμ_observed, η_mμ); # collect corresponding observations of mμ

    # update natural parameters for μ
    η_qμ = η_pμ + η_mμ;

    μ_updated = -η_qμ[1]/(2*η_qμ[2]);
    σ_updated = sqrt(-1/(2*η_qμ[2]));
    # update qμ
    q_μ = Normal(μ_updated,σ_updated);


    # and calculate the FE
    # FE = L_qμ + L_qβ - sum (loglikelihood)
    L_qμ = (η_qμ-η_pμ)' * [q_μ.μ, (q_μ.σ^2 + q_μ.μ^2)] + 1/2*(-log(1/p_μ.σ^2)+(1/p_μ.σ^2)*p_μ.μ^2 + log(1/q_μ.σ^2)-(1/q_μ.σ^2)*q_μ.μ^2);
    L_qβ = (η_qβ-η_pβ)' * [q_β.α * q_β.θ, log(q_β.α * q_β.θ)-q_β.α * q_β.θ^2/(2*(q_β.α * q_β.θ)^2)] - p_β.α*log(1/p_β.θ) + log(gamma(p_β.α)) +q_β.α*log(1/q_β.θ) - log(gamma(q_β.α));
    sum_llh = sum((q_β.α*q_β.θ*q_μ.μ)*x_observed .- 0.5*q_β.α*q_β.θ*x_observed.^2 .+ 0.5*(log(q_β.α*q_β.θ)-q_β.α*q_β.θ^2/(2*(q_β.α*q_β.θ)^2)-q_β.α*q_β.θ*(q_μ.σ^2+q_μ.μ^2)-log(2*π)));
    
    if i>1
        if L_qμ + L_qβ - sum_llh - FE[i-1] == 0
            println(i)
            break;
        end 
    end
    append!(FE,L_qμ + L_qβ - sum_llh) ;

end 

# reshape to 2xN, where N is the number of observations, and we finally get observations
input_η_mβ = reshape(input_η_mβ,2, Int(length(input_η_mβ)/2)); 
η_mβ_observed = reshape(η_mβ_observed,2,Int(length(η_mβ_observed)/2));   

input_η_mμ = reshape(input_η_mμ,2,Int(length(input_η_mμ)/2));  
η_mμ_observed = reshape(η_mμ_observed,2,Int(length(η_mμ_observed)/2)); 

##############Now, we build Gps for natural parameters of mβ and mμ
############# With observations, we are totally able to optimize hyperparameter of GPs
### First, let work on GP for mμ
T = 2; # dimension
#Define parameters for the kernel
σf_mμ = [1, 1];
l_mμ = [1e-3, 2e-3];
# Define covariance matrix A1 and A2
eye_matrix =1* Matrix(I, T,T); #identity matrix
A1_mμ, A2_mμ = eye_matrix, eye_matrix; # the choice of A can be found in https://linkinghub.elsevier.com/retrieve/pii/S0950705117306123

#now, optimize hyper-parameters 
σf_mμ_opt, l_mμ_opt,llh_μ = optim_params(1e-15,100,input_η_mμ,η_mμ_observed,A1_mμ,A2_mμ,σf_mμ,l_mμ) #optimized GP for η_mμ

### This is the GP for mβ
σf_mβ = [1, 1];
l_mβ = [1e-5, 1e-5];

# Define covariance matrix A1 and A2
A1_mβ, A2_mβ = eye_matrix, eye_matrix; 

# now, optimize hyper-parameters
σf_mβ_opt, l_mβ_opt,llh_β = optim_params(1e-15,50,input_η_mβ,η_mβ_observed,A1_mβ,A2_mβ,σf_mβ,l_mβ) #optimized GP for η_mβ



#################################################################################################################
#### Test with training data
#### After optimizing all parameters for Gps, now we apply Gps to VMP
#### Define priors for μ and β
p_μ = Normal(0,100);
p_β = Gamma(0.01,0.1);
 
η_pμ = [p_μ.μ/(p_μ.σ^2), -1/(2*p_μ.σ^2)]; # natural parameters of μ prior
η_pβ = [-1/p_β.θ, p_β.α-1]; # natural parameters of β prior

###### Initialize posterior for μ and β
q_μ = Normal(0,50); 
q_β = Gamma(0.01, 0.1);

#### Preparation for gaussian process
input_η_mβ_gp = []; #store inputs for mβ
η_mβ_observed_gp = []; #store natural parameters of mβ 

input_η_mμ_gp = []; #store inputs for mμ
η_mμ_observed_gp = []; #store natural parameters of mμ


FE_gp = []; #store FE values
n_iter = length(FE);

for i=1:n_iter
    # This is just an example (i.e. we obtained m_μ and m_β). Later when we finish the GPs, we can replace Gps here

    ################  suppose we update q_β first #####################
    #η_mβ = [-sum_x_squared + 2*q_μ.μ*sum_x - N*(q_μ.σ^2 + q_μ.μ^2), N/2]; # natural parameters of m_β , collect μ and σ
    mean_β,cova_β = GP(input_η_mβ,η_mβ_observed,[q_μ.μ, q_μ.σ],A1_mβ,A2_mβ,σf_mβ_opt,l_mβ_opt);
    η_mβ = convert(Array{Float64,1},mean_β);
    append!(input_η_mβ_gp,[q_μ.μ, q_μ.σ]); # collect μ and σ
    append!(η_mβ_observed_gp, η_mβ); # collect corresponding observations of mβ

    # update natural parameters for β
    η_qβ = η_pβ + η_mβ;

    #convert from natural parameters to α and θ
    α_updated = η_qβ[2] + 1;
    θ_updated = -1/η_qβ[1]; 

    # update qβ
    q_β = Gamma(α_updated,θ_updated);

    ################# now we update q_μ #####################
    #η_mμ = [q_β.α*q_β.θ*sum_x, -N*q_β.α*q_β.θ/2]; 
    mean_μ,cova_μ = GP(input_η_mμ,η_mμ_observed,[q_β.α, q_β.θ],A1_mμ,A2_mμ,σf_mμ_opt,l_mμ_opt);
    η_mμ = convert(Array{Float64,1},mean_μ);

    append!(input_η_mμ_gp,[q_β.α, q_β.θ]); # collect α and θ
    append!(η_mμ_observed_gp, η_mμ); # collect corresponding observations of mμ

    # update natural parameters for μ
    η_qμ = η_pμ + η_mμ;

    μ_updated = -η_qμ[1]/(2*η_qμ[2]);
    σ_updated = sqrt(-1/(2*η_qμ[2]));
    # update qμ
    q_μ = Normal(μ_updated,σ_updated);


    # and calculate the FE
    # FE = L_qμ + L_qβ - sum (loglikelihood)
    L_qμ = (η_qμ-η_pμ)' * [q_μ.μ, (q_μ.σ^2 + q_μ.μ^2)] + 1/2*(-log(1/p_μ.σ^2)+(1/p_μ.σ^2)*p_μ.μ^2 + log(1/q_μ.σ^2)-(1/q_μ.σ^2)*q_μ.μ^2);
    L_qβ = (η_qβ-η_pβ)' * [q_β.α * q_β.θ, log(q_β.α * q_β.θ)-q_β.α * q_β.θ^2/(2*(q_β.α * q_β.θ)^2)] - p_β.α*log(1/p_β.θ) + log(gamma(p_β.α)) +q_β.α*log(1/q_β.θ) - log(gamma(q_β.α));
    sum_llh = sum((q_β.α*q_β.θ*q_μ.μ)*x_observed .- 0.5*q_β.α*q_β.θ*x_observed.^2 .+ 0.5*(log(q_β.α*q_β.θ)-q_β.α*q_β.θ^2/(2*(q_β.α*q_β.θ)^2)-q_β.α*q_β.θ*(q_μ.σ^2+q_μ.μ^2)-log(2*π)));
    
    if i>1
        if L_qμ + L_qβ - sum_llh - FE[i-1] == 0
            println(i)
            break;
        end 
    end
    append!(FE_gp,L_qμ + L_qβ - sum_llh) ;

end 

## Plot groundtruth-FE and GP-FE
plot(collect(1:n_iter),[FE FE_gp],label = ["gt_FE" "GP_FE"])
#savefig("F:\\Tue\\Thesis\\codes\\GP for VMP\\FE_comparison.png")

###### test value of log_mll (not important) #####
"""
N_1 = size(input_η_mμ,2);
T = size(η_mμ_observed,1);
K = zeros(N_1,N_1);
for i=1:N_1
    for j=1:N_1
        K[i,j] = norm(input_η_mμ[:,i] - input_η_mμ[:,j],2)^2; 
    end
end
η_mμ_observed_resha = reshape(η_mμ_observed,N_1*T)
log_mll(K,η_mμ_observed_resha,A1_mμ,A2_mμ,σf_mμ,l_mμ)
"""