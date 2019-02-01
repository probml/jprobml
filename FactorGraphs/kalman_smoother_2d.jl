
# RTS smoothing for a linear Gaussian system
# based on https://github.com/probml/pmtk3/blob/master/demos/kalmanTrackingDemo.m

import Distributions
const Dist = Distributions

function diag_to_full(v)
    d = length(v)
    A = zeros(d,d)
    A[diagind(A)] = v
    return A
end

# True model
using PDMats
Ftrue = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1.0];
Htrue = [1.0 0 0 0; 0 1 0 0];
nobs, nhidden = size(Htrue)
#Qtrue = PDiagMat(fill(0.001, nhidden))
Qtrue = diag_to_full(fill(0.001, nhidden))
#Rtrue = PDiagMat(fill(1.0, nobs))
Rtrue = diag_to_full(fill(1.0, nobs))
init_mu = [8, 10, 1, 0.0];
#init_V = PDiagMat(fill(1, nhidden))
init_V = diag_to_full(fill(1, nhidden))

using LinearAlgebra
function lin_gauss_ssm_sample(T, F, H, Q, R, init_mu, init_V)
    # z(t+1) = F * z(t-1) + N(0,Q) so F is D*D
    # y(y) = H * z(t) + N(0, R) so H is O*D
    nobs, nhidden = size(H)
    zs = Array{Float64}(undef, nhidden, T)
    ys = Array{Float64}(undef, nobs, T)
    prior_z = Dist.MvNormal(init_mu, init_V)
    process_noise_dist = Dist.MvNormal(Q)
    obs_noise_dist = Dist.MvNormal(R)
    zs[:,1] = rand(prior_z)
    ys[:,1] = H*zs[:,1] + rand(obs_noise_dist)
    for t=2:T
        zs[:,t] = F*zs[:,t-1] + rand(process_noise_dist)
        ys[:,t] = H*zs[:,t] + rand(obs_noise_dist)
    end
    return zs, ys
end

# Generate data
Random.seed!(1)
T = n_samples = 20
(zs, ys) = lin_gauss_ssm_sample(10, Ftrue, Htrue, Qtrue, Rtrue, init_mu, init_V)

##############################
using ForneyLab

g = FactorGraph()

# Prior statistics
m_x_0 = placeholder(:m_x_0)
v_x_0 = placeholder(:v_x_0)

# State prior
@RV x_0 ~ GaussianMeanVariance(m_x_0, v_x_0)

# Transition and observation model
x = Vector{Variable}(undef, n_samples)
y = Vector{Variable}(undef, n_samples)


#process_noise_dist_forney = Variable()
#GaussianMeanVariance(process_noise_dist_forney, zeros(nhidden), Qtrue) # assign
x_t_prev = x_0
for t = 1:n_samples
    global x_t_prev
    @RV process_noise_t ~ GaussianMeanVariance(zeros(nhidden), Qtrue)
    #@RV process_noise_t ~ process_noise_dist_forney
    @RV x[t]= Ftrue * x_t_prev + process_noise_t;
    x[t].id = Symbol("x_", t) #:x_t;
    @RV obs_noise_t ~ GaussianMeanVariance(zeros(nobs), Rtrue)
    @RV y[t] = x[t] + obs_noise_t
    placeholder(y[t], :y, index=t)
    x_t_prev = x[t]
end

println("generating inference code")
algo = Meta.parse(sumProductAlgorithm(x))
println("Compiling")
eval(algo) # Load algorithm

# Prepare data dictionary and prior statistics
data = Dict(:y     => ys,
            :m_x_0 => init_mu,
            :v_x_0 => init_V)

# Execute algorithm
println("running forwards backwards")
marginals = step!(data);
