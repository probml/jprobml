# Modified from
#https://github.com/sisl/algforopt-notebooks/blob/master/first-order.ipynb

# See also
# https://github.com/FluxML/Flux.jl/blob/master/src/optimise/optimisers.jl

#include("support_code.jl");
#using Plots
#using Vec
include("utils.jl")

# A4O p36
function bracket_minimum(f, x=0; s=1e-2, k=2.0)
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    end
    while true
        c, yc = b + s, f(b + s)
        if yc > yb
            return a < c ? (a, c) : (c, a)
        end
        a, ya, b, yb = b, yb, c, yc
        s *= k
    end
end

# A4O p92
# 1d optimization based on Newton's method
function secant_method(df, x1, x2, ϵ)
    df1 = df(x1)
    delta = Inf
    while abs(delta) > ϵ
        df2 = df(x2)
        delta = (x2 - x1)/(df2 - df1)*df2
        x1, x2, df1 = x2, x2 - delta, df2
    end
    x2
end

# A4O p54
function line_search_secant(f, dfn, x, d)
    objective = α -> f(x + α*d)
    a, b = bracket_minimum(objective)
    #α = minimize(objective, a, b)
	f1d = a -> begin
		xnew = x + a*d
		g2 = dfn(xnew)
		v1 = dot(g2, d) / norm(d)
		return v1
	end
	α = secant_method(f1d, a, b, 0.0001)
	#println("ls secant chose $α")
    return α
end



# A40 p56
using LinearAlgebra
function line_search_backtracking(f, ∇f, x, d; α=1.0, p=0.5, β=1e-4, max_iter=50)
	y, g = f(x), ∇f(x)
	iter = 1
	while (iter <= max_iter) && (f(x + α*d) > y + β*α*(g⋅d))
		α *= p
		#println("line search iter $iter, $α")
		iter += 1
	end
	#println("backtracking ls chose $α")
	return α
end

import LineSearches
#https://julianlsolvers.github.io/LineSearches.jl/latest/examples/generated/customoptimizer.html
function line_search_backtracking_optim(f, g, x, d)
	linesearch_fn = LineSearches.BackTracking(order=3)
	#linesearch_fn = LineSearches.HagerZhang()
	s = d
	ϕ(α) = f(x .+ α.*s)
    function dϕ(α)
        gvec = g(x .+ α.*s)
        return dot(gvec, s)
    end
    function ϕdϕ(α)
		phi =  f(x .+ α.*s)
		gvec = g(x .+ α.*s)
        dphi = dot(gvec, s)
        return (phi, dphi)
    end
	fx = f(x)
	gvec = g(x)
	dϕ_0 = dot(s, gvec)
	α, fx = linesearch_fn(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
	#println("optim ls chose $α")
	return α
end


#################


abstract type DescentMethod end

struct GradientDescent <: DescentMethod
		α::Real # learning rate
end
function init!(M::GradientDescent, f, ∇f, x)
	return M
end
function step!(M::GradientDescent, f, ∇f, x)
	α, g = M.α, ∇f(x)
	return x - α*g
end


struct GradientDescentLineSearch <: DescentMethod
end

function init!(M::GradientDescentLineSearch, f, ∇f, x)
	return M
end
function step!(M::GradientDescentLineSearch, f, ∇f, x)
	d =  -normalize(∇f(x))
	α = line_search_backtracking(f,  ∇f, x, d)
	return x + α*d
end

struct GradientDescentLineSearchExact <: DescentMethod
end

function init!(M::GradientDescentLineSearchExact, f, ∇f, x)
	return M
end
function step!(M::GradientDescentLineSearchExact, f, ∇f, x)
	d =  -normalize(∇f(x))
	α = line_search_secant(f,  ∇f, x, d)
	return x + α*d
end

mutable struct Momentum <: DescentMethod
	α::Real # learning rate
	β::Real # momentum decay
	v::Vector{Real} # momentum
end
# default constructor
Momentum(α) = Momentum(α, 0.9, Vector{Real}())

function init!(M::Momentum, f, ∇f, x)
	M.v = zeros(length(x))
	return M
end
function step!(M::Momentum, f, ∇f, x)
	α, β, v, g = M.α, M.β, M.v, ∇f(x)
	v[:] = β*v - α*g
	return x + v
end

mutable struct NesterovMomentum <: DescentMethod
		α::Real # learning rate
		β::Real # momentum decay
		v::Vector{Real} # momentum
end
# default constructor
NesterovMomentum(α) = NesterovMomentum(α, 0.9, Vector{Real}())

function init!(M::NesterovMomentum, f, ∇f, x)
	M.v = zeros(length(x))
	return M
end
function step!(M::NesterovMomentum, f, ∇f, x)
	α, β, v = M.α, M.β, M.v
	v[:] = β*v - α*∇f(x + β*v)
	return x + v
end


mutable struct Adagrad <: DescentMethod
	α::Real # learning rate
	ϵ::Real # small value
	s::Vector{Real} # sum of squared gradient
end
 # default constructor
Adagrad(α) = Adagrad(α, 1e-8, Vector{Real}())

function init!(M::Adagrad, f, ∇f, x)
	M.s = zeros(length(x))
	return M
end

function step!(M::Adagrad, f, ∇f, x)
	α, ϵ, s, g = M.α, M.ϵ, M.s, ∇f(x)
	s[:] += g.*g
	return x - α*g ./ (sqrt.(s) .+ ϵ)
end

mutable struct RMSProp <: DescentMethod
	α::Real # learning rate
	γ::Real # decay
	ϵ::Real # small value
	s::Vector{Real} # sum of squared gradient
end
#default constructor
RMSProp(α) = RMSProp(α, 0.9, 1e-8, Vector{Real}())

function init!(M::RMSProp, f, ∇f, x)
	M.s = zeros(length(x))
	return M
end
function step!(M::RMSProp, f, ∇f, x)
	α, γ, ϵ, s, g = M.α, M.γ, M.ϵ, M.s, ∇f(x)
	s[:] = γ*s + (1-γ)*(g.*g)
	return x - α*g ./ (sqrt.(s) .+ ϵ)
end

mutable struct Adadelta <: DescentMethod
	γs::Real # gradient decay
	γx::Real # update decay
	ϵ::Real # small value
	s::Vector{Real} # sum of squared gradients
	u::Vector{Real} # sum of squared updates
end
# default constructor
Adadelta() = Adadelta(0.9, 0.9, 1e-8, Vector{Real}(), Vector{Real}())

function init!(M::Adadelta, f, ∇f, x)
	M.s = zeros(length(x))
	M.u = zeros(length(x))
	return M
end
function step!(M::Adadelta, f, ∇f, x)
	γs, γx, ϵ, s, u, g = M.γs, M.γx, M.ϵ, M.s, M.u, ∇f(x)
	s[:] = γs*s + (1-γs)*g.*g
	Δx = - (sqrt.(u) .+ ϵ) ./ (sqrt.(s) .+ ϵ) .* g
	u[:] = γx*u + (1-γx)*Δx.*Δx
	return x + Δx
end

mutable struct Adam <: DescentMethod
	α::Real # learning rate
	γv::Real # decay
	γs::Real # decay
	ϵ::Real # small value
	k::Int # step counter
	v::Vector{Real} # 1st moment estimate
	s::Vector{Real} # 2nd moment estimate
end
# default constructor
Adam(α) = Adam(α, 0.9, 0.999, 1e-8, 0, Vector{Real}(), Vector{Real}())

function init!(M::Adam, f, ∇f, x)
	M.k = 0
	M.v = zeros(length(x))
	M.s = zeros(length(x))
	return M
end
function step!(M::Adam, f, ∇f, x)
	α, γv, γs, ϵ, k = M.α, M.γv, M.γs, M.ϵ, M.k
	s, v, g = M.s, M.v, ∇f(x)
	v[:] = γv*v + (1-γv)*g
	s[:] = γs*s + (1-γs)*g.*g
	M.k = k += 1
	v_hat = v ./ (1 - γv^k)
	s_hat = s ./ (1 - γs^k)
	return x - α*v_hat ./ (sqrt.(s_hat) .+ ϵ)
end

mutable struct HyperGradientDescent <: DescentMethod
	α0::Real # initial learning rate
	μ::Real # learning rate of the learning rate
	α::Real # current learning rate
	g_prev::Vector{Real} # previous gradient
end
# default constructor
HyperGradientDescent(α0, μ) = HyperGradientDescent(α0, μ, NaN, Vector{Real}())

function init!(M::HyperGradientDescent, f, ∇f, x)
	M.α = M.α0
	M.g_prev = zeros(length(x))
	return M
end
function step!(M::HyperGradientDescent, f, ∇f, x)
	α, μ, g, g_prev = M.α, M.μ, ∇f(x), M.g_prev
	α = α + μ*(g⋅g_prev)
	M.g_prev, M.α = g, α
	return x - α*g
end


#####

function run_descent_method(M::DescentMethod, f, ∇f,  x0;
		max_iter = 100,
		callback = (M, f, ∇f, x, iter) -> norm(∇f(x), Inf) < 1e-4)
    x = x0
	done = false
    init!(M, f, ∇f, x)
    for iter = 1:max_iter
		x = step!(M, f, ∇f, x);
		done = callback(M, f, ∇f, x, iter)
        if done; break; end
    end
	return x
end

####################
