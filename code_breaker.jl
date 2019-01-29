
# See the explanation in this doc:
#https://docs.google.com/document/d/1nTpmZhs-VxpJivrQKIVnvmelc02J0F1ki5ChU0EyG68/edit?usp=sharing

include("utils.jl")

"""
     make_code_prior(L, A, partial_code)

 Make uniform prior over codewords of length `L` over alphabet `A` given a
 `partial_code` vector, which is 0 if element is unknown, and otherwise
 contains true value of code at that location.
"""
function make_code_prior(L, A, partial_code)
    shape = Tuple(fill(A, L))
    ncodes = A^L
    codes = [Base._ind2sub(shape, i) for i in 1:ncodes]
    prior = ones(ncodes)
    for i = 1:ncodes
        for l=1:L
            if partial_code[l]>0 && (codes[i][l] != partial_code[l])
                prior[i] = 0
            end
        end
    end
    prior = normalize_probdist(prior);
    return prior
end

function make_code_prior_test()
    L = 2; A = 3;
    partial_code = [0, 2]
    code_prior = make_code_prior(L, A, partial_code)
    #= Possible codes are as follows, ones compatible with partial_code with *
    1 1
    2 1
    3 1
    1 2 *
    2 2 *
    3 2 *
    1 3
    2 3
    3 3
    =#
    @test isapprox(code_prior, [0, 0, 0, 0.3, 0.3, 0.3, 0, 0, 0]; atol=1e-1)
end

#make_code_prior_test()

function encode_discrete(v, domain)
    ndx = findfirst(x -> x==v, domain)
    @assert !isnothing(ndx) "value $v not in domain $domain"
    return ndx
end

function hamming_distance(x, c)
    return sum(x .!= c)
end

function hamming_distance_ints(xndx, cndx, L, A)
    shape = Tuple(fill(A, L))
    xbits = Base._ind2sub(shape, xndx)
    cbits = Base._ind2sub(shape, cndx)
    return hamming_distance(xbits, cbits)
end


"""
    make_CPT_from_fn(f, domain1, domain2, frange)

Make a conditional proability table from a deterministic 2 argument function
`f` with inputs x1 in `domain1`, x2 in `domain2` and output y in `frange`.

We assume the domains and range are finite sets. Returns array of form
cpt[x1,x2,y], where the last dimension sums to 1 for each x1, x2.
"""
function make_CPT_from_fn(f, domain1, domain2, frange)
    cpt = zeros(length(domain1), length(domain2), length(frange))
    for (x1v, x1i) in enumerate(domain1)
        for (x2v, x2i) in enumerate(domain2)
            yv = f(x1v, x2v)
            yi = encode_discrete(yv, frange)
            cpt[x1i, x2i, yi] = 1.0
        end
    end
    return cpt
end


function make_hamming_CPT(L, A)
    domain = 1:A^L # strings of length L on alphabet A
    frange = 0:L # possible values of hamming distance
    f(x1, x2) = hamming_distance_ints(x1, x2, L, A)
    CPT = make_CPT_from_fn(f, domain, domain, frange)
    return CPT
end

function make_hamming_CPT_test()
    L = 2; A = 3;
    CPT = make_hamming_CPT(L, A)
    @test CPT[:,1,:] == [
    1 0 0; # hamming([1,1], [1,1])=1
    0 1 0;
    0 1 0;
    0 1 0;
    0 0 1; # hamming([2,2], [1,1])=2
    0 0 1;
    0 1 0;
    0 0 1;
    0 0 1
    ]
end

"""
    marginalize_CPT_parent(child_CPT, parent_prior)

Compute cpt(p1, c) = sum_p2 child_CPT(p1, p2, c) * parent_prior(p1).
"""
function marginalize_CPT_parent(child_CPT, parent_prior)
    @assert ndims(child_CPT) == 3
    (p1sz, p2sz, csz) = size(child_CPT)
    ppsz = length(parent_prior)
    errmsg = "parent prior has size $ppsz, should be $p2sz"
    @assert (length(parent_prior) == p2sz) errmsg
    CPT = zeros(p1sz, csz)
    for p1 in 1:p1sz
        for c in 1:csz
            CPT[p1, c] = sum(child_CPT[p1, :, c] .* parent_prior)
        end
    end
    return CPT
end

"""
We make a surrogate model for the function y=f(x),
where x is a string in {1,..,A}^L. Our surrogate has the form
fhat(x) = argmax_y CPT_y_x[x, y] where
CPT_y_x[x,y] = sum_c CPT_c[c] * CPT_y_xc[x,c,y]
where CPT_c is a prior over hidden code c.
We assume f(x) in {0,1,...,L}.
"""
mutable struct SurrogateModel
    CPT_c::Array{Float64,1}
    CPT_y_xc::Array{Float64, 3}
    CPT_y_x::Array{Float64, 2}
    L::Int # length of string
    A::Int # alphabet size
    fdom::Array{Int,1} # domain of unknown function
    frange::Array{Int,1} # range of unknown function
end


function encode_x(model, xstr)
    xdomain = Tuple(fill(model.A, model.L))
    xndx = Base._sub2ind(xdomain, xstr...)
    return xndx
end

function decode_x(model, xndx)
    xdomain = Tuple(fill(model.A, model.L))
    xstr = Base._ind2sub(xdomain, xndx)
    return xstr
end

function encode_y(model, yval)
    yndx = encode_discrete(yval, model.frange)
end


function make_surrogate(L, A, partial_code)
    CPT_c = make_code_prior(L, A, partial_code)
    CPT_y_xc = make_hamming_CPT(L, A)
    CPT_y_x = marginalize_CPT_parent(CPT_y_xc, CPT_c)
    model = SurrogateModel(CPT_c, CPT_y_xc, CPT_y_x, L, A, 1:A^L, 0:L)
    return model
end

function make_surrogate_test()
    L = 3; A = 4; partial_code = [1,1,1]
    model = make_surrogate(L, A, partial_code)
    # Prob the model with query strings x which are increasignly
    # far (in Hamming Distance) from the true code of (1,1,1)
    xs = [ [1,1,1], [2,1,1], [2,2,1], [2,2,2] ]
    ps = [ [1.0,0,0,0], [0,1.0,0,0], [0,0,1.0,0], [0,0,0,1.0]] # 0-3 bits away
    for i = 1:length(xs)
        xstr = xs[i]
        prob_y = ps[i]
        xndx = encode_x(model, xstr)
        CPT_y = model.CPT_y_x[xndx,:]
        @test isapprox(CPT_y, prob_y; atol=0.1)
    end
    println("tests passed")
end

function update_surrogate!(model, xstr, yval)
    xdomain = fill(model.A, model.L)
    xndx = encode_x(model, xstr)
    yndx = encode_y(model, yval)
    lik_c = model.CPT_y_xc[xndx, :, yndx]
    post_c = normalize_probdist(lik_c .* model.CPT_c)
    model.CPT_c = post_c
    model.CPT_y_x = marginalize_CPT_parent(model.CPT_y_xc, model.CPT_c)
end

function update_surrogate_test()
    # Start with uniform prior.
    # Make a lucky guess, which has f(x)=0, which implies code=x
    xs = [ [1,1,1], [1,2,3]]
    for xstr in xs
        L = 3; A = 4; partial_code = [0,0,0]
        model = make_surrogate(L, A, partial_code)
        xcode = encode_x(model, xstr)
        yval = 0
        update_surrogate!(model, xstr, yval)
        expected_post = zeros(A^L);
        expected_post[xcode] = 1.0
        @test isapprox(model.CPT_c, expected_post, atol=0.1)
    end
    println("tests passed")
end


function expected_improvement(model, xstr, incumbent_val)
    xndx = encode_x(model, xstr)
    CPT_y = model.CPT_y_x[xndx,:]
    fn(y) = max(0, incumbent_val-y) # if y<incumbent_val, we decrease f
    ei = 0
    for (yi, yv) in enumerate(model.frange)
        ei += CPT_y[yi] * fn(yv)
    end
    return ei
end

function expected_value(model, xstr)
    xndx = encode_x(model, xstr)
    CPT_y = model.CPT_y_x[xndx,:]
    ei = 0
    for (yi, yv) in enumerate(model.frange)
        ei += CPT_y[yi] * yv
    end
    return ei
end

function plot_surrogate(model, true_code_str, incumbent_val)
    xstrings = [decode_x(model, xndx) for xndx in model.fdom]
    ev = [expected_value(model, xstr) for xstr in xstrings]
    eic = [expected_improvement(model, xstr, incumbent_val) for xstr in xstrings]
    obj = [hamming_distance(xstr, true_code_str) for xstr in xstrings]
    nstrings = length(xstrings)
    xs = 1:nstrings
    #https://docs.juliaplots.org/latest/tutorial/
    #:auto, :solid, :dash, :dot
    #plot(xs, obj, color="red", linestyle=:solid)
    plot(xs,[obj ev eic],label=["obj" "ev" "eic"], linestyle=[:solid :dash :dot])
end


L = 3; A = 4; true_code_str = [1,1,1]; partial_code = [0,0,0]
model = make_surrogate(L, A, partial_code)
incumbent_val = maximum(model.frange)
plot_surrogate(model, true_code_str, incumbent_val)

xstr = [1,1,2]; yval = hamming_distance(xstr, true_code_str)
update_surrogate!(model, xstr, yval)
incumbent_val = min(yval , incumbent_val)
plot_surrogate(model, true_code_str, incumbent_val)

xstr = [1,1,4]; yval = hamming_distance(xstr, true_code_str)
update_surrogate!(model, xstr, yval)
incumbent_val = min(yval , incumbent_val)
plot_surrogate(model, true_code_str, incumbent_val)

xstr = [1,1,3]; yval = hamming_distance(xstr, true_code_str)
update_surrogate!(model, xstr, yval)
incumbent_val = min(yval , incumbent_val)
plot_surrogate(model, true_code_str, incumbent_val)
