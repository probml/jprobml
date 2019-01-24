
function square(x)
    y=x^2
    println("square of $x is $y, yeah?")
end



function ind2subv_str(dims)
    """Create array A where i'th row is a set of cartesian indices (1 indexed)
    corresponding to element i of an array, with size dims=[b b .. b],
    where b=base and there are n columns.
    A is an int array of size b^n * n.
    This is equivalent to matlab ind2subv(dims, 1:prod(dims)).
    https://homepages.inf.ed.ac.uk/imurray2/code/imurray-matlab/ind2subv.m
    except we require dims to be homogeneous.
    """
    @assert all(dims .== dims[1]) "dims must have same arity"
    b = dims[1]
    n = length(dims)
    indices = Array{Int}(undef, b^n, n)
    for i=1:b^n
        str = string(i-1, base=b, pad=n)
        chars = split(str, "") #[str[i] for i in 1:nindices]
        indices[i,:] = [parse(Int, c) for c in chars]
    end
    return indices .+ 1
end


function ind2subv_str_test()
    L = 2; A = 3;
    dims = fill(A, L)
    indices = ind2subv_str(dims)
    println(indices)
    @assert indices == [1 1; 1 2; 1 3; 2 1; 2 2; 2 3; 3 1; 3 2; 3 3]
end

function subv2ind_str(dims, indices)
    """Create vector v where i'th element is integer version corresponding to
    cartesian indices with value indices[i,:], where dims=[b b .. b].
    Equivalent to matlab subv2ind(dims, indices) defined here:
    https://homepages.inf.ed.ac.uk/imurray2/code/imurray-matlab/subv2ind.m
    except we require dims to be homogeneous.
    """
    @assert all(dims .== dims[1]) "dims must have same arity"
    b = dims[1]
    nrows = size(indices, 1)
    v = Array{Int}(undef, nrows)
    for i=1:nrows
        ndx = [string(k-1) for k in indices[i,:]]
        v[i] = parse(Int, join(ndx), base=b) + 1
    end
    return v
end

function subv2ind_str_test()
    L = 2; A = 3;
    ndx = subv2ind_str(dims, indices)
    println(ndx)
    @assert ndx == 1:(A^L)
end



function ind2sub(shape, idx; rowmajor=false)
    # Translated from
    # https://stackoverflow.com/questions/46782444/how-to-convert-a-linear-index-to-subscripts-with-support-for-negative-strides
    # matlab/fortran is rowmajor, julia/C is columnmajor
    idx = idx - 1 # switch to 0-based indexing
    d = length(shape)
    out = Array{Int}(undef, d)
    if rowmajor
        for i = 1:d
            s = rem(idx, shape[i])
            idx = idx - s
            idx = div(idx, shape[i])
            out[i] = s
        end
    else
        for i = d:-1:1
            s = rem(idx, shape[i])
            idx = idx - s
            idx = div(idx, shape[i])
            out[i] = s
        end
    end
    return out;
end

function ind2subv(shape, indices)
    # https://github.com/probml/pmtk3/blob/master/matlabTools/util/ind2subv.m
    n = length(indices)
    d = length(shape)
    out = Array{Int}(undef, n, d)
    for i=1:n
        out[i,:] = ind2sub(shape, indices[i])
    end
    return out .+ 1
end


function ind2subv_test()
    sub = ind2subv([3 3], 1:9);
    @assert sub == [1 1; 1 2; 1 3; 2 1; 2 2; 2 3; 3 1; 3 2; 3 3]

    sub = ind2subv([2 1 3], 1:6)
    @assert sub == [1 1 1; 1 1 2; 1 1 3; 2 1 1; 2 1 2; 2 1 3]
end


function mystrides(shape::Array{Int,1}; rowmajor=false)::Array{Int}
    n = length(shape)
    str = ones(n)
    if rowmajor
        for i=1:n
            if i==1
                s = 1;
            else
                s = prod(shape[1:i-1])
            end
            str[i] = s
        end
    else
        for i=1:n
            if i==n
                s = 1;
            else
                s = prod(shape[i+1:n])
            end
            str[i] = s
        end
    end
    return str
end


function sub2ind(shape::Array{Int,1}, subscripts::Array{Int,1})::Int
    # Translated from
    # https://stackoverflow.com/questions/46782444/how-to-convert-a-linear-index-to-subscripts-with-support-for-negative-strides
    n = length(shape)
    idx  = 0
    subscripts = subscripts .- 1
    str = mystrides(shape)
    for i=1:n
        sub = subscripts[i]
        s = str[i]
        idx = idx + sub * s
    end
    return idx + 1
end



# Can we not leverage built in functions?
#https://github.com/JuliaLang/julia/blob/4247bbafe650930b9f6da4feecf0a7dcc37e5204/base/abstractarray.jl#L1672


function subv2ind(shape::Array{Int,1}, subs::Array{Int,2})::Array{Int}
    #https://github.com/tminka/lightspeed/blob/master/subv2ind.m
    n = size(subs, 1)
    ndx = zeros(n)
    for i=1:n
        ndx[i] = sub2ind(shape, subs[i,:])
    end
    return ndx
end


function subv2ind_test()
    shape = [3,4,2]
    K = prod(shape)
    subs = ind2subv(shape, 1:K)
    ndx = subv2ind(shape, subs)
    @assert ndx == 1:K

    shape = [4,1,5,2]
    K = prod(shape)
    subs = ind2subv(shape, 1:K)
    ndx = subv2ind(shape, subs)
    @assert ndx == 1:K
end

ind2subv_test()
subv2ind_test()

println("all tests passed")
