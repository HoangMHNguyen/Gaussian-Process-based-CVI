# Example with GaussianProcesses.jl
# multivariate case
using Pkg
Pkg.instantiate()

using GaussianProcesses, Stheno
using Plots
using Random, Distributions, Statistics
using LinearAlgebra
using Optim, Zygote

pyplot()

Random.seed!(123);
# Model 1: Using SE kernel 
σ_n = 0.5; # std of noise
# Training data
d, n = 4, 60; # Dimension and number of observations
x = 2π * rand(d, n);  # 4-D input (or predictors)
f = sin.(x[1,:]).*sin.(x[2,:]) .- (x[3,:].^2 .* cos.(x[4,:])); # latent function 
y =  f + σ_n*rand(n); # observation

# Define parameters 
l = 2; # length-scale
σ_f = 1.5; # std of latent function f

# Define mean function and covariance matrix
mean_prior = MeanZero();  # zero mean function
kernel_SE = SE(log(l),log(σ_f)); # SE Iso kernel with l=0.5, sigma = 1, note that 
                                 # the function uses log scale

# Fit the GP
gp = GP(x,y,mean_prior,kernel_SE,log(σ_n)); # log(sigma_noise) = log(1.5)

# Optimization
#optimize!(gp) # L-BFGS
optimize!(gp,NelderMead())

 
#Predict
x_new = [2.35 1 -0.1 -3.14] #new input
μ, σ = predict_y(gp,x_new',full_cov=true) # predict new observation y*

#Take the value of marginal likelihood
mll_m1 = gp.mll #marginal likelihood of model 1 with SE kernel function

##################################################
# Model 2: Using Matern 3/2 kernel
kernel_Matern = Matern(3/2, log(l), log(σ_f)); # Matern 3/2 kernel, same length scale and std 
# Fit the GP
gp_2 = GP(x,y,mean_prior,kernel_Matern,log(σ_n)); # log(sigma_noise) = log(1.5)

# Optimization
#optimize!(gp_2,GradientDescent()) # L-BFGS
optimize!(gp_2,NelderMead())
 
#Predict
x_new = [2.35 1 -0.1 -3.14] #new input
μ_2, σ_2 = predict_y(gp_2,x_new',full_cov=true) # predict new observation y*

#Take the value of marginal likelihood
mll_m2 = gp_2.mll #marginal likelihood of model 1 with SE kernel function

###################################################
# Model 3: Using Matern 5/2 kernel
kernel_Matern52 = Matern(5/2, log(l), log(σ_f)); # Matern 3/2 kernel, same length scale and std 
# Fit the GP
gp_3 = GP(x,y,mean_prior,kernel_Matern52,log(σ_n)); # log(sigma_noise) = log(1.5)

# Optimization
#optimize!(gp_3) # L-BFGS
optimize!(gp_3,NelderMead())
 
#Predict
x_new = [2.35 1 -0.1 -3.14] #new input
μ_2, σ_2 = predict_y(gp_3,x_new',full_cov=true) # predict new observation y*

#Take the value of marginal likelihood
mll_m3 = gp_3.mll #marginal likelihood of model 1 with SE kernel function
###################################################################################

# Model 4: Using Periodic kernel
p = 3; # periodic parameter 
kernel_Per = Periodic(log(l), log(σ_f),log(p)); 
# Fit the GP
gp_4 = GP(x,y,mean_prior,kernel_Per,log(σ_n + 2)); # log(sigma_noise) = log(1.5)

# Optimization
optimize!(gp_4); # L-BFGS
#optimize!(gp_4,NelderMead());
 
#Predict
x_new = [2.35 1 -0.1 -3.14] #new input
μ_2, σ_2 = predict_y(gp_4,x_new',full_cov=true) # predict new observation y*

#Take the value of marginal likelihood
mll_m2 = gp_4.mll #marginal likelihood of model 1 with SE kernel function

########################################################################################
# Model 5: Using RQ kernel 
α = 3; # periodic parameter 
kernel_RQ = RQ(log(l), log(σ_f),log(α)); 
# Fit the GP
gp_5 = GP(x,y,mean_prior,kernel_RQ,log(σ_n)); # log(sigma_noise) = log(1.5)

# Optimization
#optimize!(gp_5); # L-BFGS
optimize!(gp_5,NelderMead());
 
#Predict
x_new = [2.35 1 -0.1 -3.14] #new input
μ_2, σ_2 = predict_y(gp_5,x_new',full_cov=true) # predict new observation y*

#Take the value of marginal likelihood
mll_m2 = gp_5.mll #marginal likelihood of model 1 with SE kernel function

##############################################################
# Model 6: combine RQ and Periodic
α = 3; # periodic parameter 
kernel_RQ = RQ(log(l), log(σ_f),log(α)); 
p = 3; # periodic parameter 
kernel_Per = Periodic(log(l), log(σ_f),log(p)); 

new_kernel = kernel_RQ + kernel_Per;

gp_6 = GP(x,y,mean_prior,new_kernel,log(σ_n + 1.5)); 

# Optimization
#optimize!(gp_6); # L-BFGS
optimize!(gp_6,NelderMead());
 
#Predict
x_new = [2.35 1 -0.1 -3.14] #new input
μ_2, σ_2 = predict_y(gp_6,x_new',full_cov=true) # predict new observation y*

#Take the value of marginal likelihood
mll_m2 = gp_6.mll #marginal likelihood of model 1 with SE kernel function

samples = rand(gp_prior_2,x,5);

Predictive = MvNormal(mean_y_new,Cov_y_new)