# Apply various optimization algorithsm to the 2d rosenbrock function.
# See this webpage for a cool interactive demo of various methods
# where you can click on an arbitrary point in space to start an optimization
# run from there, and then visualize the trajectory.
#https://bl.ocks.org/EmilienDupont/f97a3902f4f3a98f350500a3a00371db

#https://commons.wikimedia.org/wiki/File:Banana-SteepDesc.gif

import Flux
import Optim
using Test

include("AlgoForOpt/first-order.jl")

#http://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/
# The minimum value of 0.0 is at (a,a^2).
function rosenbrock(x; a=1.0, b=100.0)
  return (a - x[1])^2 + b * (x[2] - x[1]^2)^2
end

function rosenbrock_grad!(G, x; a=1.0, b=100.0)
	G[1] = -2.0 * (a-x[1]) -4*b*(x[2] - x[1]^2) * x[1]
	G[2] = 2.0 * b * (x[2] - x[1]^2)
end

function rosenbrock_grad(x; a=1.0, b=100.0)
  G = Vector{Float64}(undef, 2)
  rosenbrock_grad!(G, x, a=a, b=b)
  return G
end

#Compute scalar-valued function fn(x) and its derivative using Flux's
#revere mode AD.
function grad_and_val(fn, x)
  # Based on https://github.com/tpapp/LogDensityProblems.jl/blob/master/src/AD_Flux.jl
  y, back = Flux.Tracker.forward(fn, x)
  yval = Flux.Tracker.data(y)
  g = first(Flux.Tracker.data.(back(1)))
  return g, yval
end

function grad_and_val_test()
  x = randn(2)
  g, y = grad_and_val(rosenbrock, x)
  @test isapprox(y, rosenbrock(x))
  @test isapprox(g, rosenbrock_grad(x))
  g2 = Tracker.gradient(rosenbrock, x)[1]
  @test isapprox(g, g2)
end


#####################


# To generate figs 5-1 we use Rosenbrock a=1,b=5
f(x) = rosenbrock(x, a=1, b=5)
∇f(x) = rosenbrock_grad(x, a=1, b=5)
ftuple = (x,y) -> f([x,y])
xdomain = range(-2, stop=2, length=100)
ydomain = range(-2, stop=2, length=100)
x0 = [-1.0, -1.0]
levels=[1,2,3,5,10,20,50,100]

#=
# To generate figs 5-5 to 5-7 we use this pseudo-rosenbrock function
# (note the 4x[2])
f = x -> (1-x[1])^2 + 100*(4x[2] - x[1]^2)^2
∇f = x -> [2*(200x[1]^3 - 800x[1]*x[2] + x[1] - 1), -800*(x[1]^2 - 4x[2])]
#f = rosenbrock
#∇f = rosenbrock_grad
ftuple = (x,y) -> f([x,y])
xdomain = range(-3, stop=2, length=100)
ydomain = range(-0.5, stop=2, length=100)
x0 = [-2,1.5]
levels=[2,10,50,200,500]
=#

#########################

N = 50
expts = Tuple{DescentMethod, String}[]
# values chosen for rosenbrock(a=1,b=5)
push!(expts, (GradientDescentLineSearch(), "gradient descent + linesearch"))
push!(expts, (GradientDescent(0.01), "gradient descent (0.01)"))
push!(expts, (GradientDescent(0.05), "gradient descent (0.05)"))
push!(expts, (GradientDescent(0.055), "gradient descent (0.055)"))
push!(expts, (Momentum(0.005, 0.9, zeros(2)), "momentum (0.005, 0.9)"))
push!(expts, (NesterovMomentum(0.005, 0.9, zeros(2)), "nesterov (0.005, 0.9)"))
#=
# values chosen for pseudo-rosenbrock
push!(expts, (GradientDescent(0.0003), "gradient descent (0.0003)"))
push!(expts, (GradientDescent(0.003), "gradient descent (0.003)"))
push!(expts, (GradientDescentLineSearch(), "gradient descent + linesearch"))
#push!(expts, (Momentum(0.0003, 0.9, zeros(2)), "momentum"))
#push!(expts, (NesterovMomentum(0.0002, 0.92, zeros(2)), "Nesterov momentum"))
#push!(expts, (HyperGradientDescent(0.0004, 8e-13, NaN, zeros(2)), "hypermomentum"))
#push!(expts, (HyperNesterovMomentum(0.00023, 1e-12, 0.93, zeros(2), NaN, zeros(2)), "hyper-Nesterov"))
=#
using Plots
pyplot()
for (M, name) in expts
	plt = contour(xdomain, ydomain, (x,y)->f([x,y]), levels=levels, reuse=false)
	scatter!([x0[1]], [x0[2]], marker=:circle, markercolor=:black, markersize=5)
	scatter!([1.0], [1.0], marker=:star, markercolor=:red, markersize=5)

	pts = [x0]
	ftrace = [f(x0)]
	function cb(M, fn, grad, x, iter)
		y = fn(x)
		g = grad(x)
		gnorm = norm(g)
		#println("$name, iter=$iter, f=$y, gnorm=$gnorm")
		push!(pts, x)
		push!(ftrace, y)
		return (abs(gnorm) < 1e-2)
	end
	xfinal = run_descent_method(M, f, ∇f, x0; max_iter=N, callback=cb)
	#pts = run_descent_method(M, f, ∇f, x0, N)
	xtrace = hcat(pts...) # 2xN
	plot!(xtrace[1,:], xtrace[2,:], label=name)
	display(plt)

	plt = plot(ftrace, label=name)
	display(plt)
end



#=
function get_opt_history(opt_result)
	xtrace = Optim.x_trace(opt_result)
	N = length(xtrace)
	D = length(xtrace[1])
	xs = hcat(xtrace...) #DxN
	ftrace = Optim.f_trace(opt_result)
	return xs, ftrace
end

using Plots
pyplot()
function plot_history(f, xhist)
	xrange =range(minimum(xhist[1,:]), stop=maximum(xhist[1,:]), length=100)
	yrange =range(minimum(xhist[2,:]), stop=maximum(xhist[2,:]), length=100)
	plt = contour(xrange, yrange, (x,y)->f([x,y]),
		reuse=false, title="rosen")
	scatter!(xhist[1,:], xhist[2,:], label="hist")
	scatter!([xhist[1,1]], [xhist[2,1]], label="start")
	display(plt)
end

function grad_descent_rosen_demo()
	Random.seed!(1)
	x0 = randn(2)
	solver = Optim.LBFGS(;linesearch = Optim.BackTracking())
	opts = Optim.Options(iterations = 100, store_trace=true, extended_trace=true,
		show_trace=true)
	result = Optim.optimize(rosenbrock, rosenbrock_grad, x0, solver, opts; inplace=false)
	xopt = result.minimizer
	xhist, fhist = get_opt_history(result)
	plot_history(rosenbrock, xhist)
end

=#

#x1, iter = gradient_descent(rosenbrock, rosenbrock_grad, x0, lr=0.1)


#=
# https://github.com/JuliaNLSolvers/Optim.jl/issues/452
x=rand(20,3); y=x*[1;2;3]+randn(20);
function cb(opt_state)
  #iteration, value, g_norm
  #metadata: Dict{Any,Any}
  # For BFGS, if extended_trace=true, we store this metadata:
   #Any["Current step size", "g(x)", "~inv(H)", "x"]
   #dump(opt_state)
   #println(keys(opt_state.metadata))
   println(opt_state.metadata["x"])
   # false means don't stop the optimization procedure
   # if you want the callback to stop the procedure under some conditions, return true
   return false
end
using PyPlot
#opts = Optim.Options(extended_trace=true, store_trace=false, callback=cb)
opts = Optim.Options(extended_trace=true, store_trace=true)
w0 = [0.0;0.0;0.0]
res = Optim.optimize(w->sum(z->z^2, x*w-y), w0, Optim.BFGS(), opts)
xs = Optim.x_trace(res)
=#
