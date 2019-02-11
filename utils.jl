
using Test, LinearAlgebra, Flux

#https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/20
function lin2cart(shape, indices)
    #return Tuple.(CartesianIndices(shape)[indices])
    return [Base._ind2sub(Tuple(shape), i) for i in indices]
end

function cart2lin(shape, indices)
    #return  LinearIndices(shape)[CartesianIndex.(indices)]
    return [Base._sub2ind(Tuple(shape), i...) for i in indices]
end

function normalize_probdist(probdist)
    s = sum(probdist)
    return probdist ./ s
end

# Source:
# https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/kalman/gaussian_contours.jl
function bivariate_normal(X::Matrix, Y::Matrix, σ_x::Real=1.0, σ_y::Real=1.0,
                         μ_x::Real=0.0, μ_y::Real=0.0, σ_xy::Real=0.0)
   Xμ = X .- μ_x
   Yμ = Y .- μ_y
   ρ = σ_xy/(σ_x*σ_y)
   z = Xμ.^2/σ_x^2 + Yμ.^2/σ_y^2 - 2*ρ.*Xμ.*Yμ/(σ_x*σ_y)
   denom = 2π*σ_x*σ_y*sqrt(1-ρ^2)
   return exp.(-z/(2*(1-ρ^2))) ./ denom
end


# https://github.com/probml/pmtk3/blob/master/matlabTools/graphics/gaussPlot2d.m
function plot_gauss2d(m, C)
    U = eigvecs(C)
    D = eigvals(C)
    N = 100
    t = range(0, stop=2*pi, length=N)
    xy = zeros(Float64, 2, N)
    xy[1,:] = cos.(t)
    xy[2,:] = sin.(t)
    #k = sqrt(6) # approx sqrt(chi2inv(0.95, 2)) = 2.45
    k = 1.0
    w = (k * U * Diagonal(sqrt.(D))) * xy # 2*N
    #Plots.scatter!([m[1]], [m[2]], marker=:star, label="")
    handle = Plots.plot!(w[1,:] .+ m[1], w[2,:] .+ m[2], label="")
    return handle
end

function plot_gauss2d_test()
    m = [0.0, 0.0]
    #C = randn(2,2); C = C*C';
    C = [1.0 0.0; 0.0 3.0];
    @test isposdef(C)
    Plots.scatter([m[1]], [m[2]], marker=:star)
    plot_2dgauss(m, C)
end

#Compute scalar-valued function fn(x) and its derivative using Flux's
#revere mode AD.
function fun_and_grad(fn, x)
  # Based on https://github.com/tpapp/LogDensityProblems.jl/blob/master/src/AD_Flux.jl
  y, back = Flux.Tracker.forward(fn, x)
  yval = Flux.Tracker.data(y)
  g = first(Flux.Tracker.data.(back(1)))
  return yval, g
end

function fun_and_grad_test()
	f(x) = x[1]^2 + 100*x[2]^2
	g(x) = [2*x[1], 200*x[2]]
	x = [1,1]
	y1 = f(x)
	g1 = g(x)
	y2, g2 = fun_and_grad(f, x)
	@test isapprox(y1, y2)
	@test isapprox(g1, g2)
end



#http://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/
# The minimum value of 0.0 is at (a,a^2).
function rosenbrock(x; a=1.0, b=100.0)
  return (a - x[1])^2 + b * (x[2] - x[1]^2)^2
end

function rosenbrock_grad!(g, x; a=1.0, b=100.0)
	g[1] = -2.0 * (a-x[1]) -4*b*(x[2] - x[1]^2) * x[1]
	g[2] = 2.0 * b * (x[2] - x[1]^2)
end

function rosenbrock_grad(x; a=1.0, b=100.0)
  g = Vector{Float64}(undef, 2)
  rosenbrock_grad!(g, x, a=a, b=b)
  return g
end

function rosenbrock_hessian(x; a=1.0, b=100.0)
	H = zeros(2,2)
	H[1,1] = 2 + 8*b * x[2];
	H[1,2] = -4 * b * x[1];
	H[2,1] = H[1,2];
	H[2,2] = 2 * b;
	return H
end

function rosenbrock_condition()
	H = rosenbrock_hessian([1,1])
	println("condition number = $(cond(H))")
	evals = eigvals(H)
	@assert isapprox(cond(H), evals[2]/evals[1])
end

function rosenbrock_grad_test()
  x = randn(2)
  y, g = fun_and_grad(rosenbrock, x)
  @test isapprox(y, rosenbrock(x))
  @test isapprox(g, rosenbrock_grad(x))
  g2 = Tracker.gradient(rosenbrock, x)[1]
  @test isapprox(g, g2)
end
