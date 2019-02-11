# Reproduce fig 5.1 from A4O book to illustrate zigzag effect
# of gradient descent with linesearch

using Vec
using LinearAlgebra

include("utils.jl")
include("first-order.jl")


f(x) = rosenbrock(x, a=1, b=5)
∇f(x) = rosenbrock_grad(x, a=1, b=5)
ftuple = (x,y) -> f([x,y])
xdomain = range(-2, stop=2, length=100)
ydomain = range(-2, stop=2, length=100)
x0 = [-1.0, -1.0]
levels=[1,2,3,5,10,20,50,100]


N = 50
expts = Tuple{DescentMethod, String}[]
push!(expts, (GradientDescentLineSearchExact(), "exact-linesearch"))
push!(expts, (GradientDescentLineSearch(), "backtracking-linesearch"))
push!(expts, (GradientDescent(0.01), "GD-0.01"))
push!(expts, (GradientDescent(0.05), "GD-0.05"))
push!(expts, (GradientDescent(0.055), "GD-0.055"))

using Plots
pyplot()
for (M, name) in expts
	println("using method $name")
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
	plot!(xtrace[1,:], xtrace[2,:], label=name, linecolor=:red, width=2)
	display(plt)
	savefig(plt, "Figures/zigzag-$name.pdf")
	#plt = plot(ftrace, label=name; yaxis=:log)
	#display(plt)
end



#=
# Version using arrays
fn(x) = rosenbrock(x, a=1, b=5)
dfn(x) = rosenbrock_grad(x, a=1, b=5)
xdomain = range(-2, stop=2, length=100)
ydomain = range(-2, stop=2, length=100)
x0 = [-1.0, -1.0]
levels=[1,2,3,5,10,20,50,100]

pts = [x0]
for i in 1 : 10
    xcur = pts[end]
    g = dfn(xcur)
    d = -normalize(g)
    f1d = a -> begin
        xnew = xcur + a*d
        g2 = dfn(xnew)
    	v1 = dot(g2, d) / norm(d)
		return v1
    end
    alpha = secant_method(f1d, 0.0, 1.0, 0.0001)
    push!(pts, xcur + alpha*d)
end
using Plots
pyplot()
plt = contour(xdomain, ydomain, (x,y) -> fn([x,y]), levels=levels, reuse=false)
scatter!([x0[1]], [x0[2]], marker=:star, markercolor=:red, markersize=5)
xtrace = hcat(pts...) # 2xN
plot!(xtrace[1,:], xtrace[2,:])
display(plt)
=#


#=
# Version using Vec (tuples)
# rosenbrock(a=1, b=5)
f = (x,y) -> (1-x)^2 + 5*(y - x^2)^2
df = (x,y) -> [2*(10*x^3-10*x*y+x-1), 10*(y-x^2)]
xdomain = range(-2, stop=2, length=100)
ydomain = range(-2, stop=2, length=100)
p0 = (-1, -1)

pts2d = Tuple{Float64,Float64}[p0]
for i in 1 : 10
    x,y = pts2d[end]
    dp = normalize(-VecE2{Float64}(df(x, y)...))
	println("dp $dp")
    f1d = a -> begin
        x2 = x + a*dp.x
        y2 = y + a*dp.y

        da = df(x2, y2)
        pa = VecE2{Float64}(da[1], da[2])

        pp = proj(pa, dp, Float64)
		println("da=$da, pp=$pp")
		return pp
    end
    alpha = secant_method(f1d, 0.0, 1.0, 0.0001)
    push!(pts2d, (x + alpha*dp.x, y + alpha*dp.y))
end
using Plots
pyplot()
plt = Plots.contour(xdomain, ydomain, f, levels=[1,2,3,5,10,20,50,100], reuse=false)
Plots.plot!([p[1] for p in pts2d], [p[2] for p in pts2d], label="GD with line search")
display(plt)
=#
