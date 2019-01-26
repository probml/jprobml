
function square(x)
    y=x^2
    println("square of $x is $y, yeah?")
end


function ind2subv(shape, indices)
    """Map linear indices to cartesian.
    shape: d-tuple with size of each dimension.
    indices: n-iterable with linear indices.
    Returns: n-vector of d-tuples with cartesian indices.

    Based on:
    https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/8
    Similar to this matlab function:
    https://github.com/probml/pmtk3/blob/master/matlabTools/util/ind2subv.m
    """
    n = length(indices)
    d = length(shape)
    cndx = CartesianIndices(Dims(shape))
    out = Array{Tuple}(undef, n)
    for i=1:n
        lndx = indices[i]
        out[i] = cndx[lndx]
    end
    return out
end

function subv2ind(shape, cindices)
    """Return linear indices given vector of cartesian indices.
    shape: d-tuple of dimensions.
    cindices: n-iterable of d-tuples, each containing cartesian indices.
    Returns: n-vector of linear indices.

    Based on:
    https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/8
    Similar to this matlab function:
    https://github.com/tminka/lightspeed/blob/master/subv2ind.m
    """
    lndx = LinearIndices(Dims(shape))
    n = length(cindices)
    out = Array{Int}(undef, n)
    for i = 1:n
        out[i] = lndx[cindices[i]...]
    end
    return out
end
