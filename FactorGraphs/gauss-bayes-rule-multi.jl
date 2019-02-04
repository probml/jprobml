# Bayes rule for a linear Gaussian model
# X -> Y, observe Y, infer X
# Y ~ N(A x + b, R)
# X ~ N(0, Q)

#https://github.com/biaslab/ForneyLab.jl/issues/17#issuecomment-460290360

using ForneyLab, LinearAlgebra

function make_model()
g = FactorGraph()

nhidden = 4
nobs = 2
A = [1 0 0 0; 0 1 0 0]
b = zeros(Float64, nobs)
Q = eye(nhidden)
R = 0.1*eye(nobs)
y_data = randn(nobs)

@RV x ~ GaussianMeanVariance(zeros(nhidden), Q)
@RV obs_noise ~ GaussianMeanVariance(zeros(nobs), R)
@RV y = A*x + b + obs_noise
placeholder(y, :y, dims=(nobs,)) # add clamping
end

algo = Meta.parse(sumProductAlgorithm(x))
eval(algo) # Load algorithm
data = Dict(:y     => y_data)
marginals = step!(data);
