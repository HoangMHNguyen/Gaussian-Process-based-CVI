using Pkg
Pkg.instantiate()

Random.seed!(128) ;
### define noise ###
σ_n = 0.5 # standard deviation of noise
N = Normal(0,σ_n) # Normal noise with zero mean and variance σ^2

########### prepare data #########
x = collect(0:0.01:5); # input
f = 2*sin.(x);   # latent function
ϵ = rand(N,length(x)); #noise
y = f + ϵ ; # things we'll observe

## hyperparameters for kernel function
σ_f = 1;  # prior standard deviation of the latent function
l = 0.5; # prior length-scale, governs the correlation between values of f

#our observations or training data
pos = sort(unique(rand(1:501,50)));
x_observed = x[pos];
y_observed = y[pos];

########  the prior distribution over f  ####
f_mean_prior = MeanZero(); # zero mean
kerl = SE(log(l),log(σ_f)); # kernel function, here we use squared exponential
gp_prior = GPE(; mean = f_mean_prior, kernel = kerl) # gaussian process without observation (prior)

# Plot some samples from the prior of f
samples = rand(gp_prior,x,5);
plot(x,samples,xlims = (0,5),ylims = (-3,3))
plot!(x,zeros(length(x)),xlims = (0,5),ylims = (-3,3),grid=false,label="gp prior mean",ribbon = 2*σ_f, fillalpha=.3, c = :orange)

####### After observing data points 
gp = GP(x_observed,y_observed,f_mean_prior,kerl,log(σ_n)); #Gaussian process with observations
plot(gp; title = "GP with given hyperparameters")

############# optimize hyperparameter by maximizing marginal log likelihood ########
# optimize hyperparameters using command "optimize!"
optimize!(gp,Newton()); # use L-BFGS solver
plot(gp; title = "GP after Optimization",label="approx function")
plot!(x[1:length(x_observed)],f[1:length(x_observed)],label = "True function")


#Predict a values
μ_new, σ_new = predict_y(gp,[2.11],full_cov=true)
#we can use μ_new as the predictive value

# get the value of marginal log likelihood for model with SE kernel
mll_m1 = gp.mll

##########################################################################################################
##### model 2: RQ kernel
α = 2;
kernel_2 = RQ(log(l),log(σ_f),log(α));
gp_prior_2 = GPE(; mean = f_mean_prior, kernel = kernel_2) # gaussian process without observation (prior)

# Plot some samples from the prior of f
samples = rand(gp_prior_2,x,5);
plot(x,samples,xlims = (0,5),ylims = (-3,3))
plot!(x,zeros(length(x)),xlims = (0,5),ylims = (-3,3),grid=false,label="gp prior mean",ribbon = 2*σ_f, fillalpha=.3, c = :orange)

####### After observing data points 
gp_2 = GP(x_observed,y_observed,f_mean_prior,kernel_2,log(σ_n)); #Gaussian process with observations
plot(gp_2; title = "GP with given hyperparameters")

############# optimize hyperparameter by maximizing marginal log likelihood ########
# optimize hyperparameters using command "optimize!"
optimize!(gp_2,GradientDescent()) #
plot(gp_2; title = "GP after Optimization",label="approx function")
plot!(x[1:length(x_observed)],f[1:length(x_observed)],label = "True function")


#Predict a values
μ_new, σ_new = predict_y(gp_2,[2.11],full_cov=true)

# get the value of marginal log likelihood for model with SE kernel
mll_m2 = gp_2.mll

############################################################################################################
##### model 3: Periodic kernel
p = 2; # periodic parameter
kernel_3 = Periodic(log(l),log(σ_f),log(p));
gp_prior_3 = GPE(; mean = f_mean_prior, kernel = kernel_3) # gaussian process without observation (prior)

# Plot some samples from the prior of f
samples = rand(gp_prior_3,x,5);
plot(x,samples,xlims = (0,5),ylims = (-3,3))
plot!(x,zeros(length(x)),xlims = (0,5),ylims = (-3,3),grid=false,label="gp prior mean",ribbon = 2*σ_f, fillalpha=.3, c = :orange)

####### After observing data points 
gp_3 = GP(x_observed,y_observed,f_mean_prior,kernel_3,log(σ_n)) #Gaussian process with observations
plot(gp_3; title = "GP with given hyperparameters")

############# optimize hyperparameter by maximizing marginal log likelihood ########
# optimize hyperparameters using command "optimize!"
optimize!(gp_3,NelderMead()) # use L-BFGS solver
plot(gp_3; title = "GP after Optimization",label="approx function")
plot!(x[1:pos[end]],f[1:pos[end]],label = "True function")


#Predict a values
μ_new, σ_new = predict_y(gp_3,[2.11],full_cov=true)

# get the value of marginal log likelihood for model with SE kernel
mll_m3 = gp_3.mll

###########################################################################
##### model 4: Poly kernel
c = 5; #additive constant
deg = 5; # degree of polynomial 
kernel_4 = Poly(log(c),log(σ_f),deg);
gp_prior_4 = GPE(; mean = f_mean_prior, kernel = kernel_4) # gaussian process without observation (prior)

# Plot some samples from the prior of f
# cannot plot samples from priors when using Polynomial kernel
#samples = rand(gp_prior_4,x,5);
#plot(x,samples,xlims = (0,5),ylims = (-3,3))
#plot!(x,zeros(length(x)),xlims = (0,5),ylims = (-3,3),grid=false,label="gp prior mean",ribbon = 2*σ_f, fillalpha=.3, c = :orange)

####### After observing data points 
gp_4 = GP(x_observed,y_observed,f_mean_prior,kernel_4,log(σ_n)) #Gaussian process with observations
plot(gp_4; title = "GP with given hyperparameters")

############# optimize hyperparameter by maximizing marginal log likelihood ########
# optimize hyperparameters using command "optimize!"
optimize!(gp_4,GradientDescent()) # use L-BFGS solver
plot(gp_4; title = "GP after Optimization",label="approx function")
plot!(x[1:pos[end]],f[1:pos[end]],label = "True function")


#Predict a values
μ_new, σ_new = predict_y(gp_4,[2.11],full_cov=true)

# get the value of marginal log likelihood for model with SE kernel
mll_m3 = gp_4.mll
