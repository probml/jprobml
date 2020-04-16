# Based on
#https://github.com/sisl/algforopt-notebooks/blob/master/second-order.ipynb

using LinearAlgebra

include("first-order.jl")

function _line_search(f, x, d)
	d = normalize(d)
	objective = α -> f(x + α*d)
	v, α = f(x), 1e-6
	while f(x + α*d) < v
		v = f(x + α*d)
		α += 1e-6
	end
	return x + α*d
end

#abstract type DescentMethod end
mutable struct DFP <: DescentMethod
	Q::Matrix{Real}
end
DFP() = DFP(Matrix{Real}(undef, 0, 0))
function init!(M::DFP, f, ∇f, x)
	M.Q = Matrix(1.0I, length(x), length(x))
	return M
end
function step!(M::DFP, f, ∇f, x)
	Q, g = M.Q, ∇f(x)
	x′ = _line_search(f, x, -Q*g)
	g′ = ∇f(x′)
	δ = x′ - x
    γ = g′ - g
    Q[:] = Q - Q*γ*γ'*Q/(γ'*Q*γ) + δ*δ'/(δ'*γ)
    return x′
end

mutable struct BFGS <: DescentMethod
	Q::Matrix{Real}
end
BFGS() = BFGS(Matrix{Real}(undef, 0, 0))
function init!(M::BFGS, f, ∇f, x)
	M.Q = Matrix(1.0I, length(x), length(x))
	return M
end
function step!(M::BFGS, f, ∇f, x)
	Q, g = M.Q, ∇f(x)
	x′ = _line_search(f, x, -Q*g)
	g′ = ∇f(x′)
	δ = x′ - x
    γ = g′ - g
    Q[:] = Q - (δ*γ'*Q + Q*γ*δ')/(δ'*γ) + (1 + (γ'*Q*γ)/(δ'*γ))[1]*(δ*δ')/(δ'*γ)
    return x′
end

mutable struct LBFGS <: DescentMethod
	m::Int # memory length
	δs::Array{Vector{Real},1}
	γs::Array{Vector{Real},1}
	qs::Array{Vector{Real},1}
end
LBFGS(m) = LBFGS(m, Array{Vector{Real},1}(), Array{Vector{Real},1}(), Array{Vector{Real},1}())
function init!(M::LBFGS, f, ∇f, x)
	M.δs = []
	M.γs = []
    M.qs = []
	return M
end
function step!(M::LBFGS, f, ∇f, x)
    δs, γs, qs, g = M.δs, M.γs, M.qs, ∇f(x)
    m = length(δs)
    if m > 0
        q = g
        for i in m : -1 : 1
            qs[i] = copy(q)
            q -= (δs[i]⋅q)/(γs[i]⋅δs[i])*γs[i]
        end
        z = (γs[m] .* δs[m] .* q) / (γs[m]⋅γs[m])
        for i in 1 : m
            z += δs[i]*(δs[i]⋅qs[i] - γs[i]⋅z)/(γs[i]⋅δs[i])
        end
        x′ = _line_search(f, x, -z)
    else
        x′ = _line_search(f, x, -g)
    end
    g′ = ∇f(x′)
    push!(δs, x′ - x);
	push!(γs, g′ - g)
    push!(qs, zeros(length(x)))
    while length(δs) > M.m
        popfirst!(δs); popfirst!(γs); popfirst!(qs)
    end
    return x′
end
