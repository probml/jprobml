# scratch
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

#=

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
