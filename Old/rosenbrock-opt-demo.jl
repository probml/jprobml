# Apply various optimization algorithsm to 2d objectives.

# For the rosenbrock function.
# See this webpage for a cool interactive demo of various methods
#https://bl.ocks.org/EmilienDupont/f97a3902f4f3a98f350500a3a00371db

# For the quadratic, we follow the parameters from
#http://people.duke.edu/~ccc14/sta-663-2018/notebooks/S09G_Gradient_Descent_Optimization.html


#import Flux
import Optim
#using Test
using LinearAlgebra

include("utils.jl")
include("first-order.jl")
include("second-order.jl")


#####################

f(x) = rosenbrock(x, a=1, b=5)
∇f(x) = rosenbrock_grad(x, a=1, b=5)
ftuple = (x,y) -> f([x,y])
xdomain = range(-2, stop=2, length=100)
ydomain = range(-2, stop=2, length=100)
x0 = [-1.0, -1.0]
levels=[1,2,3,5,10,20,50,100]
fn_name = "rosen-a1b5"

#=
# To generate figs 5-5 to 5-7 of A4O we use this pseudo-rosenbrock function
# (note the 4x[2])
f = x -> (1-x[1])^2 + 100*(4x[2] - x[1]^2)^2
∇f = x -> [2*(200x[1]^3 - 800x[1]*x[2] + x[1] - 1), -800*(x[1]^2 - 4x[2])]
#f = rosenbrock
#∇f = rosenbrock_grad
xdomain = range(-3, stop=2, length=100)
ydomain = range(-0.5, stop=2, length=100)
x0 = [-2,1.5]
levels=[2,10,50,200,500]
fn_name = "rosen2-a1b100"
=#

#=
f(x) = rosenbrock(x, a=1, b=100)
∇f(x) = rosenbrock_grad(x, a=1, b=100)
xdomain = range(-4, stop=4, length=200)
ydomain = range(-4, stop=4, length=200)
x0 = [4.0, -4.0]
levels=range(1,stop=10,length=10) .^ 5
fn_name = "rosen-a1b100"
=#

#=
# Ill-conditioned quadratic
f(x) = x[1]^2 + 100*x[2]^2
∇f(x) = [2*x[1], 200*x[2]]
xdomain = range(-1, stop=1, length=100)
ydomain = xdomain
x0 = [-1.0, -1.0]
levels =  [0.1,1,2,4,9, 16, 25, 36, 49, 64, 81, 100]
fn_name = "quad"
=#

#########################

N = 50
expts = Tuple{DescentMethod, String}[]

#values chosen for rosenbrock(a=1,b=5)
#push!(expts, (GradientDescentLineSearch(), "GD-linesearch"))
#push!(expts, (GradientDescent(0.01), "GD-0.01"))
#push!(expts, (GradientDescent(0.05), "GD-0.05"))
push!(expts, (Momentum(0.005), "momentum"))
push!(expts, (NesterovMomentum(0.005), "nesterov"))
push!(expts, (Adagrad(0.2), "Adagrad"))
push!(expts, (RMSProp(0.2), "RMSProp"))
push!(expts, (Adadelta(), "AdaDelta"))
push!(expts, (Adam(0.2), "Adam"))
push!(expts, (HyperGradientDescent(0.05, 2e-5), "hyper-GD"))
push!(expts, (BFGS(), "BFGS"))
push!(expts, (LBFGS(5), "LBFGS"))
push!(expts, (DFP(), "DFP"))


#=
# values chosen for quadratic
push!(expts, (GradientDescent(1e-3), "gradient descent (1e-3)"))
push!(expts, (GradientDescent(1e-2), "gradient descent (1e-2)"))
push!(expts, (GradientDescentLineSearch(), "gradient descent + linesearch"))
push!(expts, (Momentum(1e-2), "momentum"))
#push!(expts, (NesterovMomentum(0.0002, 0.92, zeros(2)), "Nesterov momentum"))
#push!(expts, (HyperGradientDescent(0.0004, 8e-13, NaN, zeros(2)), "hypermomentum"))
#push!(expts, (HyperNesterovMomentum(0.00023, 1e-12, 0.93, zeros(2), NaN, zeros(2)), "hyper-Nesterov"))
=#


#=
# values chosen for rosenbrock(a=1, b=100)
push!(expts, (GradientDescent(1e-4), "gradient descent (0.0001)"))
push!(expts, (GradientDescent(3e-4), "gradient descent (0.0003)"))
push!(expts, (GradientDescentLineSearch(), "gradient descent + linesearch"))
push!(expts, (Momentum(1e-5), "momentum"))
#push!(expts, (NesterovMomentum(0.0002, 0.92, zeros(2)), "Nesterov momentum"))
#push!(expts, (HyperGradientDescent(0.0004, 8e-13, NaN, zeros(2)), "hypermomentum"))
#push!(expts, (HyperNesterovMomentum(0.00023, 1e-12, 0.93, zeros(2), NaN, zeros(2)), "hyper-Nesterov"))
=#

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
for (M, method_name) in expts
	println("using $method_name")
	plt = contour(xdomain, ydomain, (x,y)->f([x,y]), levels=levels, reuse=false)
	scatter!([x0[1]], [x0[2]], marker=:circle, markercolor=:black,
		markersize=5, label="start")
	scatter!([1.0], [1.0], marker=:star, markercolor=:red,
		markersize=5, label="opt")

	pts = [x0]
	ftrace = [f(x0)]
	function cb(M, fn, grad, x, iter)
		y = fn(x)
		g = grad(x)
		gnorm = norm(g)
		#println("$name, iter=$iter, f=$y, gnorm=$gnorm")
		if abs(y)>1e10 || abs(gnorm)>1e10 || abs(gnorm) < 1e-2; return true; end
		push!(pts, x)
		push!(ftrace, y)
		return false
	end
	xfinal = run_descent_method(M, f, ∇f, x0; max_iter=N, callback=cb)
	xtrace = hcat(pts...) # 2xN
	#println(ftrace)
	plot!(xtrace[1,:], xtrace[2,:], label=method_name, linewidth=3, linecolor=:red)
	display(plt)
	savefig(plt, "Figures/$fn_name-$method_name-x.pdf")
	plt = plot(ftrace, label=method_name; yaxis=:log)
	display(plt)
	savefig(plt, "Figures/$fn_name-$method_name-f.pdf")
end


#Optiml.jl has a different interface
# See https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/
function apply_LBFGS(f, ∇f, x0, N)
	solver = Optim.LBFGS(;linesearch = Optim.BackTracking())
	opts = Optim.Options(iterations = N, store_trace=true, extended_trace=true)
	result = Optim.optimize(rosenbrock, rosenbrock_grad, x0, solver, opts; inplace=false)
	xopt = result.minimizer
	xtrace = Optim.x_trace(result) # N-vector of D-vectors
	N = length(xtrace)
	D = length(xtrace[1])
	xs = hcat(xtrace...) #DxN
	ftrace = Optim.f_trace(result)
	return xopt, xs, ftrace
end

 xopt, xtrace, ftrace = apply_LBFGS(f, ∇f, x0, N)
 method_name = "LBFGS-Optim"
 plt = contour(xdomain, ydomain, (x,y)->f([x,y]), levels=levels, reuse=false)
 scatter!([x0[1]], [x0[2]], marker=:circle, markercolor=:black,
	 markersize=5, label="start")
 scatter!([1.0], [1.0], marker=:star, markercolor=:red,
	 markersize=5, label="opt")
 plot!(xtrace[1,:], xtrace[2,:], label=method_name, linewidth=3, linecolor=:red)
 display(plt)
 savefig(plt, "Figures/$fn_name-$method_name-x.pdf")
 plt = plot(ftrace, label=method_name; yaxis=:log)
 display(plt)
 savefig(plt, "Figures/$fn_name-$method_name-f.pdf")
