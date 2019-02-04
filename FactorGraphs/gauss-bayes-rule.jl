# Bayes rule for a linear Gaussian model
# X -> Y, observe Y, infer X
# Y ~ N(a x + b, R)
# X ~ N(0, Q)

using ForneyLab, LinearAlgebra


###################### Scalar example
g = FactorGraph()
A = 0.2
b = 0.0
Q = 1
R = 0.
y_data = 1.2

@RV x ~ GaussianMeanVariance(0.0, Q)
@RV n_t ~ GaussianMeanVariance(0.0, R)
@RV y = A*x + b + n_t
placeholder(y, :y)

println("generating inference code")
algo = Meta.parse(sumProductAlgorithm(x))
println("Compiling")
eval(algo) # Load algorithm

data = Dict(:y     => y_data)

println("running inference")
marginals = step!(data);

####################
# Multivariate example
#https://github.com/biaslab/ForneyLab.jl/issues/17


g = FactorGraph()

nhidden = 4
nobs = 2
A = [1 0 0 0;
     0 1 0 0]
b = zeros(Float64, nobs)
Q = eye(nhidden)
R = 0.1*eye(nobs)
y_data = randn(nobs)

@RV x ~ GaussianMeanVariance(zeros(nhidden), Q)
@RV obs_noise ~ GaussianMeanVariance(zeros(nobs), R)
@RV y = A*x + b + obs_noise
placeholder(y, :y, dims=(nobs,)) # add clamping

algo = Meta.parse(sumProductAlgorithm(x))
eval(algo) # Load algorithm
data = Dict(:y     => y_data)
marginals = step!(data);
post = marginals[:x]
