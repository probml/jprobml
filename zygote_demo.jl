

function f(x,w,n)
    a = 0.0
    for i=1:n
        a += w'*x;
    end
    return a;
end
f2(x,w,n) = n*w'*x;

D = 3;
w = rand(D); x = rand(D);
@assert isapprox(f(x,w,n), f2(x,w,n))

# df/dx = n*w, df/dw = n*x
g = gradient(f,x,w,n)[2] # gradient returns (df/dx, df/dw)
g2 = n*x
@assert isapprox(g, g2)

println("time :")
n = 10000
@time gradient(f, x, w, n)

println("done")
