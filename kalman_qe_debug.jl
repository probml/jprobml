# https://github.com/QuantEcon/QuantEcon.jl

using LinearAlgebra

#=
import QuantEcon
function kalman_smoother_qe(params, ys)
    F = params.F; H = params.H; Q = params.Q; R = params.R; mu0 = params.mu0; V0 = params.V0;
    A = F; G = H;
    kf = QuantEcon.Kalman(A, G, Q, R)
    QuantEcon.set_state!(kf, mu0, V0)
    loglik = QuantEcon.compute_loglikelihood(kf, ys) # ok
    mx, loglik, Vx = QuantEcon.smooth(kf, ys)
    return mx, Vx
end
=#

using QuantEcon
function kalman_smoother_qe(params, ys)
    F = params.F; H = params.H; Q = params.Q; R = params.R; mu0 = params.mu0; V0 = params.V0;
    A = F; G = H;
    kf = Kalman(A, G, Q, R)
    set_state!(kf, mu0, V0)
    loglik = compute_loglikelihood(kf, ys)
    mx, loglik, Vx = smooth(kf, ys)
    return mx, Vx
end

function make_params()
    F = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1.0];
    H = [1.0 0 0 0; 0 1 0 0];
    nobs, nhidden = size(H)
    Q = Matrix(I, nhidden, nhidden) .* 0.001
    R = Matrix(I, nobs, nobs) .* 1.0
    mu0 = [8, 10, 1, 0.0];
    V0 = Matrix(I, nhidden, nhidden) .* 1.0
    params = (mu0 = mu0, V0 = V0, F = F, H = H, Q = Q, R = R)
    return params
end

params = make_params()
T = 10
nhidden = 4
nobs = 2
zs = randn(nhidden, T)
ys = randn(nobs, T)
mx, Vx = kalman_smoother_qe(params, ys)
