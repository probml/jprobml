# Bayes rule for a linear Gaussian model
# X -> Y, observe Y, infer X
# Y ~ N(a x + b, R)
# X ~ N(0, Q)
using ForneyLab
g = FactorGraph()

nhidden = 1 #2
nobs = 1
A = 0.2 # [1 0]
Q = 1 # eye(nhidden)
R = 0.1 # 0.1*eye(nobs)
y_data = 1.2

@RV x ~ GaussianMeanVariance(zeros(nhidden), Q)
@RV obs_noise ~ GaussianMeanVariance(zeros(nobs), R)
@RV y = x + obs_noise
#@RV y ~ GaussianMeanVariance(x, R)
placeholder(y, :y) # add clamping

println("generating inference code")
algo = Meta.parse(sumProductAlgorithm(x))
println("Compiling")
eval(algo) # Load algorithm

data = Dict(:y     => y_data)

println("running inference")
marginals = step!(data);
