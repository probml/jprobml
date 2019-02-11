
# https://github.com/probml/pmtk3/blob/bb64a174c4ded561cfdba26d835e1aaa7221c368/demos/kalmanTrackingDemo.m


using  LinearAlgebra, Test
import Distributions
include("kalman_qe.jl")

function make_params()
    F = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1.0];
    H = [1.0 0 0 0; 0 1 0 0];
    nobs, nhidden = size(H)
    Q = Matrix(I, nhidden, nhidden) .* 0.001
    R = Matrix(I, nobs, nobs) .* 0.1 # 1.0
    mu0 = [8, 10, 1, 0.0];
    V0 = Matrix(I, nhidden, nhidden) .* 1.0
    params = (mu0 = mu0, V0 = V0, F = F, H = H, Q = Q, R = R)
    return params
end


# Returns xs: H*T, ys: O*T
function kalman_sample(kn::Kalman, T::Int)
    nobs, nhidden = size(kn.G)
    xs = Array{Float64}(undef, nhidden, T)
    ys = Array{Float64}(undef, nobs, T)
    mu0 = kn.cur_x_hat
    V0 = kn.cur_sigma
    prior_z = Distributions.MvNormal(mu0, V0)
    process_noise_dist = Distributions.MvNormal(kn.Q)
    obs_noise_dist = Distributions.MvNormal(kn.R)
    xs[:,1] = rand(prior_z)
    ys[:,1] = kn.G*zs[:,1] + rand(obs_noise_dist)
    for t=2:T
        xs[:,t] = kn.A*xs[:,t-1] + rand(process_noise_dist)
        ys[:,t] = kn.G*xs[:,t] + rand(obs_noise_dist)
    end
    return xs, ys
end


function kalman_filter(kn::Kalman, y::AbstractMatrix)
    obs_size, T = size(y)
    n = size(kn.G, 2)
    @assert n == kn.n
    x_filtered = Matrix{Float64}(undef, n, T)
    sigma_filtered = Array{Float64}(undef, n, n, T)
    sigma_forecast = Array{Float64}(undef, n, n, T)
    logL = 0
    for t in 1:T
        logL = logL + log_likelihood(kn, y[:, t])
        prior_to_filtered!(kn, y[:, t])
        x_filtered[:, t], sigma_filtered[:, :, t] = kn.cur_x_hat, kn.cur_sigma
        filtered_to_forecast!(kn)
        sigma_forecast[:, :, t] = kn.cur_sigma
    end
    return x_filtered, logL, sigma_filtered, sigma_forecast
end

function kalman_smoother(kn::Kalman, y::AbstractMatrix)
    T = size(y, 2)
    x_filtered, logL, sigma_filtered, sigma_forecast = kalman_filter(kn, y)
    x_smoothed = copy(x_filtered)
    sigma_smoothed = copy(sigma_filtered)
    for t in (T-1):-1:1
        x_smoothed[:, t], sigma_smoothed[:, :, t] =
            go_backward(kn, x_filtered[:, t], sigma_filtered[:, :, t],
                        sigma_forecast[:, :, t], x_smoothed[:, t+1],
                        sigma_smoothed[:, :, t+1])
    end
    return x_smoothed, logL, sigma_smoothed
end

include("utils.jl")
function do_plot(zs, ys, m, V)
    # m is H*T, V is H*H*T, where H=4 hidden states
    plt = scatter(ys[1,:], ys[2,:], label="observed", reuse=false)
    plt = scatter!(zs[1,:], zs[2,:], label="true", marker=:star)
    xlims!(minimum(ys[1,:])-1, maximum(ys[1,:])+1)
    ylims!(minimum(ys[2,:])-1, maximum(ys[2,:])+1)
    display(plt)
    m2 = m[1:2,:]
    V2 = V[1:2, 1:2, :]
    T = size(m2, 2)
    for t=1:T
        plt = plot_gauss2d(m2[:,t], V2[:,:,t])
    end
    display(plt)
end


Random.seed!(2)
T = 10
params = make_params()
F = params.F; H = params.H; Q = params.Q; R = params.R; mu0 = params.mu0; V0 = params.V0;
kf = Kalman(F, H, Q, R)
(zs, ys) = kalman_sample(kf, T) # H*T, O*T
println("inference")
set_state!(kf, mu0, V0)
mF, loglik, VF = kalman_filter(kf, ys)
set_state!(kf, mu0, V0)
mS, loglik, VS = kalman_smoother(kf, ys)
#m, V = kalman_smoother(params, ys)

println("plotting")
using Plots; pyplot()
closeall()
do_plot(zs, ys, mF, VF); title!("Filtering")
do_plot(zs, ys, mS, VS); title!("Smoothing")
