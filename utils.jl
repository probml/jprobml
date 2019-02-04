

"""
    lin2cart(shape, indices)
Transform vector of linear `indices` to cartesian form, using array dimensions
specified in `shape`, cf matlab's ind2subv.

For discussion, see
https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/20

# Examples:
```
julia> lin2cart((3,4,2), [3,5])
[ (3, 1, 1), (2, 2, 1)]
```


"""
function lin2cart(shape, indices)
    #return Tuple.(CartesianIndices(shape)[indices])
    return [Base._ind2sub(Tuple(shape), i) for i in indices]
end


"""
    cart2lin(shape, indices)
Transform vector of cartesian `indices` to linear form, using array dimensions
specified in `shape`, cf matlab's subv2ind.

For discussion, see
https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/20

# Example:
```
julia> cart2lin([3,4,2], [(3,1,1), (2,2,1)])
[3, 5]
```
"""
function cart2lin(shape, indices)
    #return  LinearIndices(shape)[CartesianIndex.(indices)]
    return [Base._sub2ind(Tuple(shape), i...) for i in indices]
end

"""
    normalize_probdist(probdist)
Normalize a 1d discrete probability distribution to sum to 1.
"""
function normalize_probdist(probdist)
    s = sum(probdist)
    return probdist ./ s
end

#http://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/
# a = 1, b = 100 and the initial values x=0, y=0. The minimum is at (a,a^2).
rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function rosenbrock_grad!(G, x)
  G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
  G[2] = 200.0 * (x[2] - x[1]^2)
end

function rosenbrock_grad(x)
  G = Vector{Float64}(undef, 2)
  G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
  G[2] = 200.0 * (x[2] - x[1]^2)
  return G
end


import Flux
"""
    y, g = val_and_grad(fn, x)
Compute scalar-valued function fn(x) and its derivative using Flux's
revere mode AD.
"""
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
