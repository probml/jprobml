using Vec
using LinearAlgebra

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



#http://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/
# The minimum value of 0.0 is at (a,a^2).
function rosenbrock(x; a=1.0, b=100.0)
  return (a - x[1])^2 + b * (x[2] - x[1]^2)^2
end

function rosenbrock_grad!(G, x; a=1.0, b=100.0)
	G[1] = -2.0 * (a-x[1]) -4*b*(x[2] - x[1]^2) * x[1]
	G[2] = 2.0 * b * (x[2] - x[1]^2)
end

function rosenbrock_grad(x; a=1.0, b=1.0)
  G = Vector{Float64}(undef, 2)
  rosenbrock_grad!(G, x, a=a, b=b)
  return G
end

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


# Version using vectors (arrays)
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
