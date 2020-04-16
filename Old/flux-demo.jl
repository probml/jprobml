#https://fluxml.ai/Flux.jl/stable/models/basics/
using Flux.Tracker
using Flux

f(x) = 3x^2 + 2x + 1

# df/dx = 6x + 2
df(x) = Tracker.gradient(f, x)[1]

df(2) # 14.0 (tracked)

# d²f/dx² = 6
d2f(x) = Tracker.gradient(df, x; nest=true)[1]

# This fails: known bug
# https://github.com/FluxML/Flux.jl/issues/566
#d2f(2) # 6.0 (tracked)

f(W, b, x) = W * x + b

Tracker.gradient(f, 2, 3, 4)
#(4.0 (tracked), 1.0, 2.0 (tracked))

W = param(2) # 2.0 (tracked)
b = param(3) # 3.0 (tracked)

f(x) = W * x + b

params = Params([W, b])
grads = Tracker.gradient(() -> f(4), params)

grads[W] # 4.0  df/dW = x = 4
grads[b] # 1.0  df/db = 1


f2(W, b, x) = W * x + b
W = param(2)
b = param(3)
x = param(4)
params2 = Params([W, b, x])
grads2 = Tracker.gradient(() -> f2, params2)


###
Random.seed!(1234)
W = rand(2, 5)
b = rand(2)

predict(x) = W*x .+ b

function loss(x, y)
  ŷ = predict(x)
  sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2) # Dummy data
loss(x, y) # ~ 2.8


W = param(W)
b = param(b)

gs = Tracker.gradient(() -> loss(x, y), Params([W, b]))

using Flux.Tracker: update!

Δ = gs[W]

# Update the parameter and reset the gradient
update!(W, -0.1Δ)

loss(x, y) # ~ 1.4

function linear(in, out)
  W = param(randn(out, in))
  b = param(randn(out))
  x -> W * x .+ b
end

linear1 = linear(5, 3) # we can access linear1.W etc
linear2 = linear(3, 2)

model(x) = linear2(σ.(linear1(x)))

model(rand(5)) # => 2-element vector

############

nin = 5; nhid = 2;
W = param(rand(nhid, nin))
b = param(rand(nhid))
f(x, W, b) = sum(W*x .+ b)
x = rand(nin)
f(x, W, b)
g = Tracker.gradient(() -> f(x,W,b), Params([x,W,b]))


nin = 3
wp = param([-1,0,1])
bp = param(0.0)
f(x, w, b) = sum( w .*x + b)
xin = 1:3
f(xin, wp, bp)
g = Tracker.gradient(() -> f(xin, wp, bp), Params([wp,bp]))

###

#https://stackoverflow.com/questions/49135107/logistic-regression-using-flux-jl


using GLM, DataFrames, Flux.Tracker

Random.seed!(1)
n = 100
df = DataFrame(s1=rand(n), s2=rand(n))
df[:y] = rand(n) .< 1 ./ (1 .+ exp.(-(1 .+ 2 .* df[1] .+ 0.5 .* df[2])))
model2 = glm(@formula(y~s1+s2), df, Binomial(), LogitLink())

x = Matrix(df[1:2])
y = df[3]
N=10; x = rand(N,2); y = rand([false,true], N)
W = param(rand(2,1))
b = param(rand(1))
predict(x) = 1.0 ./ (1.0 .+ exp.(-x*W .- b))
loss(x,y) = -sum(log.(predict(x[y,:]))) - sum(log.(1 .- predict(x[.!y,:])))

function update!(ps, η = .01)
  for w in ps
    w.data .-= w.grad .* η
    w.grad .= 0
  end
end

maxiter = 10
for iter=1:maxiter
  back!(loss(x,y))
  if max(maximum(abs.(W.grad)), abs(b.grad[1])) < 0.01
    break
  end
  update!((W, b))
end

################
# Least squares regression

N = 10; D = 2;
Random.seed!(1)
X = rand(N, D)
X1 = [ones(N) X]
X1t = transpose(X1)
wtrue = rand(D+1)
yobs = X1 * wtrue
loss(w) = sum((X1 * w - yobs) .^2)

# OLS solution
# http://web.stanford.edu/class/ee103/julia_slides/julia_least_squares_slides.pdf
west1 = inv(X1t * X1) * X1t * yobs
west2 = X1 \ yobs
#using LinearAlgebra
#F = qr(X1)
#@test isapprox(X1, F.Q * F.R)
#west3 = inv(F.R)*(F.Q'*yobs)

# Gradient method
wp = param(rand(D+1))
gradloss(wp) = (back!(loss(wp)); wp.grad)
dloss(w) = 2*(X1t * X1 * w - X1t * ytrue)
@test isapprox(gradloss(wp), dloss(wp), 1e-2)


using Optim
x0 = zeros(2)
result1 = optimize(rosenbrock, x0, LBFGS(m=5))
sol1 = Optim.minimizer(result1)
ncalls1 = result1.iterations

result2 = optimize(rosenbrock, x0, LBFGS(m=5); autodiff = :forward)
sol2 = Optim.minimizer(result2)
ncalls2 = result2.iterations

result3 = optimize(rosenbrock, rosenbrock_grad!, x0, LBFGS(m=5))
sol3 = Optim.minimizer(result3)
ncalls3 = result3.iterations
