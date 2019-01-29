# scratch file

# Latent dimensionality, # hidden units.
Dz, Dh = 5, 500

# Components of recognition model / "encoder" MLP.
A, μ, logσ = Dense(28^2, Dh, tanh), Dense(Dh, Dz), Dense(Dh, Dz)
g(X) = (h = A(X); (μ(h), logσ(h)))
z(μ, logσ) = μ + exp(logσ) * randn(Float32)

# Generative model / "decoder" MLP.
f = Chain(Dense(Dz, Dh, tanh), Dense(Dh, 28^2, σ))

using Distributed
# parallel loops:

Sys.CPU_THREADS

# parallel maps:
using LinearAlgebra
function rank_marray()
	marr = [rand(1000,1000) for i=1:10]
	println(map(rank, marr))
end
@benchmark rank_marray()

# memory estimate:  158.40 MiB
#   allocs estimate:  202
#   --------------
#   minimum time:     5.659 s (3.19% GC)
#   median time:      5.659 s (3.19% GC)
#   mean time:        5.659 s (3.19% GC)
#   maximum time:     5.659 s (3.19% GC)
#   --------------
#   samples:          1
#   evals/sample:     1
#
function prank_marray()
	marr = [rand(1000,1000) for i=1:10]
	println(pmap(rank, marr))
end
@benchmark prank_marray()
# BenchmarkTools.Trial:
#   memory estimate:  76.34 MiB
#   allocs estimate:  1128
#   --------------
#   minimum time:     3.517 s (0.82% GC)
#   median time:      3.643 s (2.44% GC)
#   mean time:        3.643 s (2.44% GC)
#   maximum time:     3.769 s (3.95% GC)
#   --------------
#   samples:          2
#   evals/sample:     1
#

using BenchmarkTools

fmap(x) = map(x -> 2x, x)
fcomprehension(x) = [2x for x in x]
fdot(x) = 2 .* x
function floop(x)
    y = similar(x)
    for i in eachindex(x)
        y[i] = 2*x[i]
    end
    return y
end
function floopopt(x)
    y = similar(x)
    @simd for i in eachindex(x)
        @inbounds y[i] = 2*x[i]
    end
    return y
end

x = rand(1000)
@btime fmap($x)
@btime fcomprehension($x)
@btime fdot($x)
@btime floop($x)
@btime floopopt($x);


function buffon(n)
	hit = 0
	for i = 1:n
		mp = rand()
		phi = (rand() * pi) - pi / 2 # angle at which needle falls
		xright = mp + cos(phi)/2  # x-location of needle
		xleft = mp - cos(phi)/2
		# if xright >= 1 || xleft <= 0
		# 	hit += 1
		# end
		# Does needle cross either x == 0 or x == 1?
		p = (xright >= 1 || xleft <= 0) ? 1 : 0
		hit += p
	end
	miss = n - hit
	piapprox = n / hit * 2
end

@time buffon(1e8)
#  3.399998 seconds (5 allocations: 176 bytes)
#3.1413701707991235

function buffon_par(n)
	hit = @distributed (+) for i = 1:n
			mp = rand()
			phi = (rand() * pi) - pi / 2
			xright = mp + cos(phi)/2
			xleft = mp - cos(phi)/2
			(xright >= 1 || xleft <= 0) ? 1 : 0
		end
	miss = n - hit
	piapprox = n / hit * 2
end


@time buffon_par(1e8)
#1.536845 seconds (2.61 k allocations: 137.891 KiB)
#3.141489580637215


#=
# Starter problems from
# http://ucidatascienceinitiative.github.io/IntroToJulia/Html/BasicProblems

using Statistics
function binrv(n, p)
    nheads = 0
    for i=1:n
        u = rand()
        if u < p; nheads += 1; end
    end
    return nheads
end
ntrials = 1000
counts = zeros(ntrials)
n = 10; p = 0.5
for i=1:ntrials
    counts[i] = binrv(n, p)
end
c = counts ./ n
println("mean of $ntrials trials with pheads=$p is $(mean(c))\n")
@assert isapprox(mean(c), p, atol=0.1)


N = 5
using SparseArrays
A = spzeros(N,N)
for i=1:N
    j=i; A[i,j]=-2
    if i>1; j=i-1; A[i,j]=1; end
    if i<N-1; j=i+1; A[i,j]=1; end
end
collect(A)

function fac(x)
    y = one(x)
    for i=1:x
        y = y * i
    end
    return y
end
@assert fac(4)==24
typeof(fac(100))

function fac2(x)
    if x==1
        return x
    else
        return x*fac2(x-1)
    end
end
@assert fac(4) == fac2(4)
typeof(fac(100))
=#
