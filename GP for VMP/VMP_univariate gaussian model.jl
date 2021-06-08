using Pkg
Pkg.activate("F:/Tue/Thesis/codes/Git_code/Thesis_Hoang/Thesis_workspace/")
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

# Define prior for μ and β
p_μ = Normal(0,100.0); # mean and std, respectively
p_β = Gamma(0.01, 100.0); # shape and scale parameters, respectively

η_pμ = [-1/(2*p_μ.σ^2), p_μ.μ/p_μ.σ^2]; # natural parameters of p(μ)
η_pβ = [p_β.α-1, -1/p_β.θ]; # natural parameters of p(β)

# Initialize approximate posteriors
q_μ = Normal(0,50.0);
q_β = Gamma(0.01,10.0);

# Start to run VMP to obtain datapoint 
input_ηβ = []; #store inputs for mβ
output_ηβ = []; #store natural parameters of mβ 

input_ημ = []; #store inputs for mμ
output_ημ = []; #store natural parameters of mμ

FE = []; #store FE values

n_iter = 50;
for i=1:n_iter
    newinfo_μ = zeros(2); # new information for μ, which is the sum of all natural parameters η of m_μ
    newinfo_β = zeros(2); # new information for β, which is the sum of all natural parameters η of m_β

    # update q(β) first
    for obser=1:length(x_observed)
        η_β = narparams_beta(q_μ,x_observed[obser]);
        newinfo_β += η_β;
        #store inputs and outputs for constructing GPs later
        append!(input_ηβ, [q_μ.μ, q_μ.σ, x_observed[obser]]);
        append!(output_ηβ, η_β);
    end
    # update natural parameter vector of q(β)
    η_qβ = η_pβ + newinfo_β;
    # Extract statistical information (α and θ) of q(β) from η_qβ
    α_new = η_qβ[1] + 1;
    θ_new = -1/η_qβ[2];
    # Update q(β)
    q_β = Gamma(α_new,θ_new);

    ###### Update q(μ) 
    for obser=1:length(x_observed)
        η_μ = narparams_mu(q_β,x_observed[obser]);
        newinfo_μ += η_μ;
        #store inputs and outputs for constructing GPs later
        append!(input_ημ, [q_β.α, q_β.θ, x_observed[obser]]);
        append!(output_ημ, η_μ);
    end
    # update natural parameter vector of q(μ)
    η_qμ = η_pμ + newinfo_μ;
    # Extract statistical information (μ and σ) of q(μ) from η_qμ
    μ_new = - η_qμ[2]/(2*η_qμ[1]);
    σ_new =  sqrt(-1/(2*η_qμ[1]));
    #Update q(μ)
    q_μ = Normal(μ_new,σ_new);

    # calculate FE
    L_qμ = (η_qμ-η_pμ)' * [q_μ.σ^2 + q_μ.μ^2, q_μ.μ] + 1/2*(-log(1/p_μ.σ^2)+(1/p_μ.σ^2)*p_μ.μ^2 + log(1/q_μ.σ^2)-(1/q_μ.σ^2)*q_μ.μ^2);
    L_qβ = (η_qβ-η_pβ)' * [log(q_β.α * q_β.θ)-q_β.α * q_β.θ^2/(2*(q_β.α * q_β.θ)^2), q_β.α * q_β.θ] - p_β.α*log(1/p_β.θ) + log(gamma(p_β.α)) +q_β.α*log(1/q_β.θ) - log(gamma(q_β.α));
    sum_llh = sum((q_β.α*q_β.θ*q_μ.μ)*x_observed .- 0.5*q_β.α*q_β.θ*x_observed.^2 .+ 0.5*(log(q_β.α*q_β.θ)-q_β.α*q_β.θ^2/(2*(q_β.α*q_β.θ)^2)-q_β.α*q_β.θ*(q_μ.σ^2+q_μ.μ^2)-log(2*π)));
    
    if i>1
        if FE[i-1] - (L_qμ + L_qβ - sum_llh)  < 1e-4
            println(i)
            break;
        end 
    end
    append!(FE,L_qμ + L_qβ - sum_llh) ;
end

######### Now time for GPs ########
input_ηβ = reshape(input_ηβ,3, Int(length(input_ηβ)/3)); 
output_ηβ = reshape(output_ηβ,2,Int(length(output_ηβ)/2));   

input_ημ = reshape(input_ημ,3,Int(length(input_ημ)/3));  
output_ημ = reshape(output_ημ,2,Int(length(output_ημ)/2)); 
##############Now, we build Gps for natural parameters of mβ and mμ

######### Build GP for η_μ and η_β ######
N_gp = size(output_ημ,2); # used for both processes
T = size(output_ημ,1); # used for both processes 
eye_matrix =1* Matrix(I, T,T); #identity matrix
###### GP for η_μ
σf_μ = [1, 1];
l_μ = [2, 3];
# Define covariance matrix A1 and A2
A1_μ, A2_μ = eye_matrix, eye_matrix; # the choice of A can be found in https://linkinghub.elsevier.com/retrieve/pii/S0950705117306123

σf_μ_opt, l_μ_opt,llh_μ = optim_params(1e-4,100,input_ημ,output_ημ,A1_μ,A2_μ,σf_μ,l_μ) #optimized GP for η_mμ

###### GP for η_β
σf_β = [1, 1];
l_β = [2, 1];
# Define covariance matrix A1 and A2
A1_β, A2_β = eye_matrix, eye_matrix; 

σf_β_opt, l_β_opt,llh_β = optim_params(1e-4,100,input_ηβ,output_ηβ,A1_β,A2_β,σf_β,l_β) #optimized GP for η_mβ

##### Finally, we have got the GPs for η_μ and η_β. Now use them in VMP
#####################################################################
#Now we use GPs to approximate the natural parameter vectors of mβ and mμ. The training data set 
#will be used as test data. The purpose of this experiment is to check if the GPs works properly.
#At the end, we plot 2 curves of FE: 1 for the ground truth and another for GP scenario. If both curves go down, then we're fine!!!

# Define prior for μ and β
p_μ = Normal(0,100.0); # mean and std, respectively
p_β = Gamma(0.01, 100.0); # shape and scale parameters, respectively

η_pμ = [-1/(2*p_μ.σ^2), p_μ.μ/p_μ.σ^2]; # natural parameters of p(μ)
η_pβ = [p_β.α-1, -1/p_β.θ]; # natural parameters of p(β)

# Initialize approximate posteriors
q_μ = Normal(0,50.0);
q_β = Gamma(0.01,10.0);


new_input_ηβ = []; #store new inputs of mβ
new_output_ηβ = []; #store new natural parameters of mβ 
covariance_β = []; # store covariance matrices of mβ

new_input_ημ = []; #store new inputs for mμ
new_output_ημ = []; #store new natural parameters of mμ
covariance_μ = []; # store covariance matrices of mμ

FE_gp = [];
n_iter = 50;
for i=1:n_iter
    newinfo_μ = zeros(2); # new information for μ, which is the sum of all natural parameters η of m_μ
    newinfo_β = zeros(2); # new information for β, which is the sum of all natural parameters η of m_β

    # update q(β) first
    for obser=1:length(x_observed)
        η_β,cov_β = GP(input_ηβ,output_ηβ, [q_μ.μ, q_μ.σ,x_observed[obser]], A1_β, A2_β,σf_β_opt,l_β_opt); #GP for η_β
        newinfo_β += η_β;
        append!(new_input_ηβ, [q_μ.μ, q_μ.σ, x_observed[obser]]);
        append!(new_output_ηβ, η_β);
        append!(covariance_β,cov_β);
    end
    # update natural parameter vector of q(β)
    η_qβ = η_pβ + newinfo_β;
    # Extract statistical information (α and θ) of q(β) from η_qβ
    α_new = η_qβ[1] + 1;
    θ_new = -1/η_qβ[2];
    # Update q(β)
    q_β = Gamma(α_new,θ_new);

    ###### Update q(μ) 
    for obser=1:length(x_observed)
        η_μ,cov_μ = GP(input_ημ,output_ημ, [q_β.α, q_β.θ,x_observed[obser]], A1_μ, A2_μ,σf_μ_opt,l_μ_opt); #GP for η_μ
        newinfo_μ += η_μ;
        append!(new_input_ημ, [q_β.α, q_β.θ, x_observed[obser]]);
        append!(new_output_ημ, η_μ);
        append!(covariance_μ,cov_μ);
    end
    # update natural parameter vector of q(μ)
    η_qμ = η_pμ + newinfo_μ;
    # Extract statistical information (μ and σ) of q(μ) from η_qμ
    μ_new = - η_qμ[2]/(2*η_qμ[1]);
    σ_new =  sqrt(-1/(2*η_qμ[1]));
    #Update q(μ)
    q_μ = Normal(μ_new,σ_new);

    # calculate FE
    L_qμ = (η_qμ-η_pμ)' * [q_μ.σ^2 + q_μ.μ^2, q_μ.μ] + 1/2*(-log(1/p_μ.σ^2)+(1/p_μ.σ^2)*p_μ.μ^2 + log(1/q_μ.σ^2)-(1/q_μ.σ^2)*q_μ.μ^2);
    L_qβ = (η_qβ-η_pβ)' * [log(q_β.α * q_β.θ)-q_β.α * q_β.θ^2/(2*(q_β.α * q_β.θ)^2), q_β.α * q_β.θ] - p_β.α*log(1/p_β.θ) + log(gamma(p_β.α)) +q_β.α*log(1/q_β.θ) - log(gamma(q_β.α));
    sum_llh = sum((q_β.α*q_β.θ*q_μ.μ)*x_observed .- 0.5*q_β.α*q_β.θ*x_observed.^2 .+ 0.5*(log(q_β.α*q_β.θ)-q_β.α*q_β.θ^2/(2*(q_β.α*q_β.θ)^2)-q_β.α*q_β.θ*(q_μ.σ^2+q_μ.μ^2)-log(2*π)));
    
    if i>1
        if FE_gp[i-1] - (L_qμ + L_qβ - sum_llh)  < 1e-4
            println(i)
            break;
        end 
    end
    append!(FE_gp,L_qμ + L_qβ - sum_llh) ;
end

plot(collect(1:length(FE)),FE,label="True FE")
plot!(collect(1:length(FE_gp)),FE_gp,label="GP_FE")
"""
#### Visualize some outputs of GPs 
new_input_ηβ = reshape(new_input_ηβ,3, Int(length(new_input_ηβ)/3)); 
new_output_ηβ = reshape(new_output_ηβ,2,Int(length(new_output_ηβ)/2));   
covariance_β = reshape(covariance_β,4,Int(length(covariance_β)/4))

new_input_ημ = reshape(new_input_ημ,3,Int(length(new_input_ημ)/3));  
new_output_ημ = reshape(new_output_ημ,2,Int(length(new_output_ημ)/2)); 
covariance_μ = reshape(covariance_μ,4,Int(length(covariance_μ)/4))

visual_mean_β = [new_output_ηβ[1,i] for i=16:15:size(new_output_ηβ,2)];
visual_var_β = [covariance_β[1,i] for i=16:15:size(covariance_β,2)];

visual_mean_μ = [new_output_ημ[1,i] for i=16:15:size(new_output_ημ,2)];
visual_var_μ = [covariance_μ[1,i] for i=16:15:size(covariance_μ,2)];

scatter(visual_mean_β,yerror=visual_var_β)
scatter(visual_mean_μ,yerror=visual_var_μ)
