using Pkg
Pkg.instantiate()

using Random, Distributions, Statistics
using LinearAlgebra
using Plots

include("Function.jl")

pyplot()

### define noise ###
σ_n = 0.5 # standard deviation of noise
N = Normal(0,σ_n) # Normal noise with zero mean and variance σ^2

########### prepare data
x = collect(0:0.01:5) # input
f = 2*sin.(x)  # latent function
ϵ = rand(N,length(x)) #noise
y = f + ϵ 

#scatter(x,y,xlims = (-0.001,100.0001),ylims = (-8,10)) #visualize our observations
########### change hyperparameter here
σ_f = 1  # factor that scales the correlations
l = 0.5 # length-scale, governs the correlation between values of f
K_ff = SE(x, x, σ_f, l)  #our Kernel matrix for the prior distribution over f

##################################################
#our observations or training data
x_observed = [x[5], x[7], x[16], x[20], x[39], x[50], x[61], x[123], x[200], x[299], x[400], x[431]];
y_observed = [y[5], y[7], y[16], y[20], y[39], y[50], y[61], y[123], y[200], y[299], y[400], y[431]];

scatter(x_observed,y_observed,xlims = (0,5),ylims = (-3,3))
#savefig("F:\\Tue\\Thesis\\codes\\GP example\\observations.png")
##################################################

#######################################################################
########  the prior distribution over f  ####
eye_matrix =1* Matrix(I, length(x),length(x)) #identity matrix
K_factorized = cholesky(K_ff - minimum(eigvals(K_ff))*eye_matrix) # factorize K
f_prior = (K_up.U)' * randn(length(x),5)

f_mean_prior = x.*0
σ_prior = sqrt.(diag(K_ff))

# plot the prior
plot(x,f_mean_prior,xlims = (0,5),ylims = (-3,3),grid=false,legend=false,fillrange=f_mean_prior-2*σ_prior, fillalpha=.3, c = :orange)
plot!(x,f_mean_prior,xlims = (0,5),ylims = (-3,3),grid=false,legend=false,fillrange=f_mean_prior+2*σ_prior, fillalpha=.3, c = :orange)

plot!(x,f_prior)
xlabel!("Input x")
ylabel!("Output f")
#savefig("F:\\Tue\\Thesis\\codes\\GP example\\prior_new.png")
#####################################################################

#####################################################################
#### Posterior with some observations

eye_matrix_yy =1* Matrix(I, length(x_observed),length(x_observed)) #identity matrix 
K_yy = SE(x_observed, x_observed, σ_f, l) .+ σ_n^2*eye_matrix_yy #Covariance function of observations y
K_fy = SE(x,x_observed,σ_f,l) # covariance function of y and f
K_yf = SE(x_observed,x,σ_f,l) # covariance function of f and y

f_mean_posterior = K_fy*inv(K_yy)*y_observed  # mean of posterior dist. over f when observing datapoints
K_ff_posterior = K_ff - K_fy*inv(K_yy)*K_yf # covariance matrix of posterior dist. over f when observing datapoints
 
σ_posterior = sqrt.(diag(K_ff_posterior))

plot(x,f_mean_posterior,xlims = (0,5),ylims = (-3,3),grid=false,legend=false,fillrange=f_mean_posterior-2*σ_posterior, fillalpha=.3, c = :orange)
plot!(x,f_mean_posterior,xlims = (0,5),ylims = (-3,3),grid=false,legend=false,fillrange=f_mean_posterior+2*σ_posterior, fillalpha=.3, c = :orange)

scatter!(x_observed,y_observed)
xlabel!("Input x")
ylabel!("Output f")

plot!(x,f)
#savefig("F:\\Tue\\Thesis\\codes\\GP example\\posterior_f_sig1_l0.1.png")
#################################################################################

#################################################################################
##### Predictive distribution ####
x_new = [2.11]; #suppose this is our new input
K_yy_new = SE(x_new,x_new,σ_f,l); # covariance matrix Cov(y*,y*)
K_y_new = SE(x_new,x_observed,σ_f,l); # cross-Covariance matrix Cov(y*,y)

mean_y_new = K_y_new*inv(K_yy)*y_observed;  # mean of the predictive distribution
Cov_y_new = K_yy_new - K_y_new*inv(K_yy)*K_y_new'; # covariance of predictive distribution

Predictive = MvNormal(mean_y_new,Cov_y_new) # Predictive distribution
y_predict = rand(Predictive,length(x_new))
scatter!(x_new,y_predict[:],c = :red)
#savefig("F:\\Tue\\Thesis\\codes\\GP example\\predict_x2.11.png")

