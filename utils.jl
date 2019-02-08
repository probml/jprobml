
using Test

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
