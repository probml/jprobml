#https://fluxml.ai/Flux.jl/stable/models/basics/
using Flux.Tracker

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
