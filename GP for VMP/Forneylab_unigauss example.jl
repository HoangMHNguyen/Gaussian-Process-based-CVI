using Pkg
Pkg.activate("./workspace/")
Pkg.instantiate()

using ForneyLab
using LinearAlgebra
using Optim, Zygote
using Plots

pyplot()

# Generate observations
N = 15; #number of observations
μ_true = 3.0; #ground-truth of mean
β_true = 2.0; #ground-truth of precision

x_observed = sqrt(1/β_true)*randn(N) .+ μ_true; # collect N observations: x is drawn from Gaussian(μ_true,β_true⁻¹)

# Run VMP for univariate Gaussian 

# model specification
g = FactorGraph();
#prior for mean and precision
@RV μ ~ GaussianMeanVariance(0,100);
@RV β ~ Gamma(0.001,0.001);

# observation model (output)
x = Vector{Variable}(undef,N);
for i=1:N
    @RV x[i] ~ GaussianMeanPrecision(μ,β)
    placeholder(x[i], :x, index = i )
end
;
#factorize posterior distribution
q = PosteriorFactorization(μ,β, ids = [:M, :W]);

#visualize subgraphs
#ForneyLab.draw(q.posterior_factors[:M])

# build the variational update algorithms for each posterior factor
algorithm_update = messagePassingAlgorithm(free_energy=true);
# generate source code for the algorithms
source_code = algorithmSourceCode(algorithm_update, free_energy=true);

##### Execution
# load algorithm
eval(Meta.parse(source_code));

# initialize marginals dictionary (assign priors to posteriors)
data = Dict(:x => x_observed); #marginal for observation
marginals = Dict(:μ => vague(GaussianMeanVariance),
                :β => vague(Gamma)) ; #marginal dictionary for posteriors
n_its = 2*N; #number of iteration
F = Vector{Float64}(undef, n_its); #initialize vector for storing free free energy

μ_est = Vector{Float64}(undef, n_its); # vector that stores values of μ
β_est = Vector{Float64}(undef, n_its); # vector that stores values of β

for i=1:n_its
    stepM!(data, marginals)
    stepW!(data, marginals)

    #store free energy
    F[i] = freeEnergy(data, marginals)
end
;

# Plot free energy
plot(1:n_its, F, color="black", marker="o")

μ_approx = mean(marginals[:μ]);
β_approx = marginals[:β].params[:a] / marginals[:β].params[:b] 