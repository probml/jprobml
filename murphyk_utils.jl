


function lin2cart(shape, indices)
    """
    Transform linear indices to cartesian, cf ind2subv.
    Example:
    lin2cart((3,4,2), [3,5]) = [ (3, 1, 1), (2, 2, 1)]
    """
    #https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/20
    #return Tuple.(CartesianIndices(shape)[indices])
    return [Base._ind2sub(shape, i) for i in indices]
end

function cart2lin(shape, indices)
    """
    Transform vector of cartesian indices to linear indices, cf subv2ind.
    Example:
    cart2lin([3,4,2], [(3,1,1), (2,2,1)]) =  [3, 5]
    """
    #https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/20
    #return  LinearIndices(shape)[CartesianIndex.(indices)]
    return [Base._sub2ind(shape, i...) for i in indices]
end




test()
