""" Multivariate Gaussian Distribution
In this example we will demonstrate the Linear model of coregionalization, a symmetric Multiple output GP.
The output is a 2-D vector, i.e. our outputs contain f1 and f2. The input is scalar (1-D vector)
Both f1 and f2 are linear combinations of 2 latent function u1 and u2, drawn from 2 Gaussian Process Gp1(0,k1) and Gp2(0,k2) respectively.
So:
T = 2; (number of outputs)
Q = 2; (number of Gaussian processes for latent functions)

source: https://linkinghub.elsevier.com/retrieve/pii/S0950705117306123
Note: there is a notice for visualization, please read it before you plot graphs :) 
"""
using Pkg
Pkg.activate("./Thesis_workspace/")
Pkg.instantiate()

using Plots
using Random, Distributions, Statistics
using LinearAlgebra
using Optim, Zygote

pyplot()

include("Function.jl")

Random.seed!(128) ;

T = 2; #output dimensions
Q = 2; # number of latent gaussian process
N = 50; # number of observations

x = collect(0:0.01:5); # 1-D input

# outputs f1 and f2 
f1 = 2*sin.(x);
f2 = cos.(x); 

# After optimizing parameters, you can run the code again from here to the beginning of optimizationg section.
# hyperparameters at the beginning 
σ_f_val = [0.3, 0.4];# vector of variances for 2 kernels
l_val = [0.1, 0.01]; # vector of length-scales for 2 kernels 
σ_n_val = [0.5, 1]; #vector of std of noises

# these are hyperparameters after optimizating
#σ_f_val = σ_f_optim;# vector of variances for 2 kernels
#l_val = l_optim; # vector of length-scales for 2 kernels 
#σ_n_val = σ_n_optim ; #vector of std of noises

Σ_s = Diagonal(σ_n_val); # covariance matrix of noises

ϵ1 = σ_n_val[1] * randn(length(x)); # nosie for the 1st process
ϵ2 = σ_n_val[2] * randn(length(x)); # noise for the 2nd process 

# noisy observation
y1 = f1 + ϵ1;
y2 = f2 + ϵ2;

# Define correlation matrices between outputs
# In this example, we use identity matrix, which indicates that outputs are uncorrelated
eye_matrix =1* Matrix(I, T,T); #identity matrix
A1_val, A2_val = eye_matrix, eye_matrix; # the choice of A can be found in the source 

# observations
pos = sort(randperm(length(x))[1:N]);
x_observed = x[pos];
y1_observed = y1[pos];
y2_observed = y2[pos];
y_observed = [y1_observed y2_observed]; # observation vector

# Covariance matrix after observing some data points
Kff_observed = Kff_observation(x_observed',A1_val,A2_val,σ_f_val,l_val); # covariance matrix of f at N input data points

# the noise matrix
I_n =1* Matrix(I, N,N); #identity matrix
Σ_m = kron(Σ_s, I_n); #Kronecker product

# covariance matrix for y_observed
Kyy = Kff_observed + Σ_m;

# predict value
x_test = [2.11]; # test point
Ky_new = Kff_observation(x_test', A1_val, A2_val, σ_f_val,l_val); # K**
Kyy_new = K_cross(x_observed', x_test, A1_val, A2_val, σ_f_val, l_val); #K*

y_observed_resha = reshape(y_observed,N*T);

# Mean and covariance matrix of predictive distribution (Multivariate Gaussian distribution)
predictive_mean = Kyy_new'*inv(Kyy)*y_observed_resha; 
predictive_covmatrix = Ky_new - Kyy_new'*inv(Kyy)*Kyy_new;

##### Optimization

# create matrix of norms ||x-x'||^2
K = zeros(N,N);
for i=1:size(x_observed',2)
    for j=1:size(x_observed',2)
        K[i,j] = norm(x_observed'[:,i] - x_observed'[:,j],2)^2; # kernel for 1st GP
    end
end



##### Optimization
function log_mll(K,y,A1,A2,σ_f,l,σ_n)
    """
    K: matrix of norms ||x-x'||^2
    y: observations ((NxT)x1)
    A1, A2: correlation matrices between output dimensions
    σ_f: vector of std of kernels
    l: vector of lengthscale of kernels
    σ_n: noise vectors
    """
    K1 = σ_f[1]^2 *exp.(-K./(2*l[1]^2));
    K2 = σ_f[2]^2 *exp.(-K./(2*l[2]^2));
    I_n =1* Matrix(I, size(K,2),size(K,2)); #identity matrix
    K1 = K1 + 1e-6*I_n;
    K2 = K2 + 1e-6*I_n;
    Kff = cov_MGP(A1, A2, K1, K2); # done covariance matrix for f after N observations

    Σ_s = Diagonal(σ_n);
    Σ_m = kron(Σ_s, I_n);
    Kyy = Kff + Σ_m;
    I_nd = 1* Matrix(I, size(Kyy,2),size(Kyy,2));
    Kyy = Kyy + 1e-5*I_nd;

    log_mll = 1/2*y'*inv(Kyy)*y + 1/2*log(det(Kyy)+1e-6); #negative log-likelyhood, we want to minimize this
    
    return log_mll
end

#optimize using Zygote 
η = 1e-4; # learning rate
M = 1000; # number of iteration
for iter = 1:M
    old_σ_f_val = σ_f_val; #old value of σ_f
    old_l_val = l_val; #old value of l
    old_σ_n_val = σ_n_val; #old value of σ_n
    grads = gradient(σ_f_val,l_val,σ_n_val) do σ_f, l, σ_n
        log_mll(K,y_observed_resha, A1_val, A2_val, σ_f,l,σ_n)
    end
    σ_f_val = σ_f_val .- η*grads[1];
    l_val = l_val .- η*grads[2];
    σ_n_val = σ_n_val .- η*grads[3];
    if norm(σ_f_val-old_σ_f_val,1) < 1e-5 && norm(l_val-old_l_val,1) < 1e-5 && norm(σ_n_val-old_σ_n_val,1) < 1e-5 
        break;
    end
end
log_mll(K,y_observed_resha,A1_val,A2_val,σ_f_val,l_val,σ_n_val)

σ_f_optim = σ_f_val;
l_optim = l_val;
σ_n_optim = σ_n_val;

"""  If you want to do the visualization, do the followings:
 After done optimization, go back to the hyperparameter setting, change the paramters to optimal values (by removing '#' symbol)
 Then run the code again, but remember: don't run the optimization section, only run the part before the optimization.
 After that, you can run the visualization part.

 Reason: In the visualization section, we use Kyy to compute Kff_posterior. Unless we define Kyy again with optimal parameters in 
 the visualization section, the value of Kyy is still corresponding to the initial values of hyper-parameters, which will cause "sqrt error"
 when we compute Kff_posterior. By running the code again, we ensure that every covariance matrix is computed with the same optimal hp-parameters. 

"""

####################  Visualization  ###############################
### Prior for f
Kff_prior = Kff_observation(x',A1_val, A2_val, σ_f_val, l_val); 
Kff_prior_1 = Kff_prior[1:length(x),1:length(x)]; # prior for f1
Kff_prior_2 = Kff_prior[length(x)+1:end,length(x)+1:end]; # prior for f2

eye_matrix =1* Matrix(I, length(x),length(x)) #identity matrix
# Plot f1
K1_factorized = cholesky(Kff_prior_1 + 1e-6*eye_matrix); # factorize K
f1_prior = (K1_factorized.U)' * randn(length(x),5);

f1_mean_prior = x.*0;
σ1_prior = sqrt.(diag(Kff_prior_1))

plot(x,f1_mean_prior,xlims = (0,5),ylims = (-5,5),grid=false,legend=false,fillrange=f1_mean_prior-2*σ1_prior, fillalpha=.3, c = :orange)
plot!(x,f1_mean_prior,xlims = (0,5),ylims = (-5,5),grid=false,legend=false,fillrange=f1_mean_prior+2*σ1_prior, fillalpha=.3, c = :orange)

plot!(x,f1_prior)
xlabel!("Input x")
ylabel!("Output f1")
#savefig("F:\\Tue\\Thesis\\codes\\GP example\\multivariate GP\\Example1_f1_prior.png")

# Plot f2
K2_factorized = cholesky(Kff_prior_2 + 1e-6*eye_matrix); # factorize K
f2_prior = (K2_factorized.U)' * randn(length(x),5);

f2_mean_prior = x.*0;
σ2_prior = sqrt.(diag(Kff_prior_2))

plot(x,f2_mean_prior,xlims = (0,5),ylims = (-5,5),grid=false,legend=false,fillrange=f2_mean_prior-2*σ2_prior, fillalpha=.3, c = :orange)
plot!(x,f2_mean_prior,xlims = (0,5),ylims = (-5,5),grid=false,legend=false,fillrange=f2_mean_prior+2*σ2_prior, fillalpha=.3, c = :orange)

plot!(x,f2_prior)
xlabel!("Input x")
ylabel!("Output f2")
#savefig("F:\\Tue\\Thesis\\codes\\GP example\\multivariate GP\\Example1_f2_prior.png")

##### After some observations
Kfy = K_cross(x', x_observed', A1_val, A2_val, σ_f_val, l_val);

f_mean_posterior = Kfy*inv(Kyy)*y_observed_resha;  # mean of posterior dist. over f when observing datapoints
Kff_posterior = Kff_prior - Kfy*inv(Kyy)*Kfy'; # covariance matrix of posterior dist. over f when observing datapoints

# posterior of f1
Kff_posterior_1 = Kff_posterior[1:length(x),1:length(x)]; # prior for f1
f1_mean_posterior = f_mean_posterior[1:length(x)]
σ1_posterior = sqrt.(diag(Kff_posterior_1));

plot(x,f1_mean_posterior,xlims = (0,5),ylims = (-5,5),grid=false,label = "Mean",ribbon=2*σ1_posterior, fillalpha=.3, c = :orange)
scatter!(x_observed,y1_observed)
xlabel!("Input x")
ylabel!("Output f1")

plot!(x,f1, label="True function",legend = true)
#savefig("F:\\Tue\\Thesis\\codes\\GP example\\multivariate GP\\Example1_f1_posterior.png")

# posterior of f2
Kff_posterior_2 = Kff_posterior[length(x)+1:end,length(x)+1:end]; # prior for f2
f2_mean_posterior = f_mean_posterior[length(x)+1:end]
σ2_posterior = sqrt.(diag(Kff_posterior_2));

plot(x,f2_mean_posterior,xlims = (0,5),ylims = (-5,5),grid=false,label = "Mean",ribbon=2*σ2_posterior, fillalpha=.3, c = :orange)
scatter!(x_observed,y2_observed)
xlabel!("Input x")
ylabel!("Output f2")

plot!(x,f2, label="True function",legend = true)
#savefig("F:\\Tue\\Thesis\\codes\\GP example\\multivariate GP\\Example1_f2_posterior.png")






