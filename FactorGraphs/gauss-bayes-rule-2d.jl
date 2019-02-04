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

function make_data(params, n_data)
    y_mean = params.true_x
    y_dist = Distributions.MvNormal(y_mean, params.R)
    y_data_all = rand(y_dist, n_data) # n_data x obs_dim
    y_data = vec(mean(y_data_all, dims=2))
    return y_data, y_data_all
end

function make_factor_graph(params, n_data)
    g = FactorGraph()
    @RV x ~ GaussianMeanVariance(params.mu_x, params.Q)
    mu_y = x
    Robs = (1.0/n_data)*params.R # since observing average of data
    @RV y ~ GaussianMeanVariance(mu_y, Robs)
    placeholder(y, :y, dims=(2,))
    fg = (x = x, y = y)
    return fg
end

function make_fg_inference(fg)
    algo = Meta.parse(sumProductAlgorithm(fg.x))
    eval(algo) # compile the step! function
    function infer(y_data)
        data = Dict(fg.y.id => y_data)
        marginals = step!(data);
        post = marginals[:x]
        post_gauss = Distributions.MvNormal(mean(post), cov(post))
        return post_gauss
    end
    return infer
end

params =
    (true_x = [0.5, 0.5],
     mu_x = [0.0, 0.0],
     Q = 0.1 * eye(2),
     R = 0.1 .* [2 1; 1 1]
     )

prior_gauss = Distributions.MvNormal(params.mu_x, params.Q)
n_data = 10
Random.seed!(1)
y_data, y_data_all = make_data(params, n_data)
fg = make_factor_graph(params, n_data)
infer = make_fg_inference(fg)
post_gauss = infer(y_data)

using Plots; pyplot()
xrange = -1.5:0.01:1.5
yrange = xrange
scatter(y_data_all[1,:], y_data_all[2,:], label="obs", reuse=false)
scatter!([params.true_x[1]], [params.true_x[2]], label="truth")
xlims!(minimum(xrange), maximum(xrange))
ylims!(minimum(xrange), maximum(xrange))
title!("Data")
savefig("Figures/gauss-bayes-rule-2d-data.png")

contour(xrange, yrange, (x,y)->Distributions.pdf(prior_gauss,[x,y]),
    reuse=false, title="prior")
contour(xrange, yrange, (x,y)->Distributions.pdf(post_gauss,[x,y]),
    reuse=false, title = "Posterior")

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
