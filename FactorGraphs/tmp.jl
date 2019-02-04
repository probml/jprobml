# Bayes rule for a linear Gaussian model
# X -> Y, observe Y, infer X
# Y ~ N(A x + b, R)
# X ~ N(mu, Q)

# We consider the example from MLAPP sec 4.4.2.2
# where A=I, b=0, mu=[0.5 0.5], Q=0.1I, R=0.1*[2 1; 1 1]
#https://github.com/probml/pmtk3/blob/master/demos/gaussInferParamsMean2d.m

using ForneyLab
import LinearAlgebra:Diagonal
using Distributions

function make_model()
    g = FactorGraph()

    hidden_dim = 2
    obs_dim = 2
    Q = 0.1*eye(hidden_dim)
    mu_x = [0.5, 0.5]
    A = eye(2)
    b = [0.0, 0.0]
    R = 0.1 .* [2 1; 1 1]
    y_mean = A*mu_x + b
    y_dist = Distributions.MvNormal(y_mean, R)
    n_data = 10
    y_data_all = rand(y_dist, n_data) # n_data x obs_dim
    y_data = vec(mean(y_data_all, dims=2))

    @RV x ~ GaussianMeanVariance(mu_x, Q)
    mu_y = A*x + b
    @RV y ~ GaussianMeanVariance(mu_y, R)
    placeholder(y, :y, dims=(obs_dim,)) # add clamping

    return x, y, y_data
end

x, y, y_data = make_model()
algo = Meta.parse(sumProductAlgorithm(x))
eval(algo) # compile the step! function
data = Dict(:y     => y_data)
marginals = step!(data);
post = marginals[:x]
post_gauss = Distributions.MvNormal(mean(post), cov(post))
prior_gauss = Distributions.MvNormal([0.5, 0.5], 0.1*eye(2))

using Plots; pyplot()
xrange = -1.5:0.1:1.5
yrange = xrange
contour(xrange, yrange, (x,y)->Distributions.pdf(prior_gauss,[x,y]), reuse=false)
title!("Prior")

contour(xrange, yrange, (x,y)->Distributions.pdf(post_gauss,[x,y]), reuse=false)
title!("Posterior")

#=

import ForneyLab
const FL = ForneyLab
import ForneyLab.@RV
import LinearAlgebra

function make_model()
    g = FL.FactorGraph()

    Q = 0.1*FL.eye(2)
    mu_x = [0.5, 0.5]
    A = FL.eye(2)
    b = [0.0, 0.0]
    R = 0.1 .* [2 1; 1 1]
    y_data = A*mu_x + b + randn(2) # Should sample using noise ~ R

    @RV x ~ FL.GaussianMeanVariance(mu_x, Q)
    #@RV obs_noise ~ GaussianMeanVariance(zeros(nobs), R)
    #@RV y = A*x + b + obs_noise
    mu_y = A*x + b
    @RV y ~ FL.GaussianMeanVariance(mu_y, R)
    FL.placeholder(y, :y, dims=(nobs,)) # add clamping

    return x, y, y_data
end

x, y, y_data = make_model()

algo = Meta.parse(FL.sumProductAlgorithm(x))
eval(algo) # compile the step! function
data = Dict(:y     => y_data)
marginals = step!(data);
post = marginals[:x]
=#
