# Examples of how to fit a binary logistic regression model

#https://fluxml.ai/Flux.jl/v0.1.1/examples/logreg.html
#http://marubon-ds.blogspot.com/2018/05/deep-learning-with-julia-introduction.html

#https://github.com/FluxML/Flux.jl/blob/1cf37ab9eb806368d5702ab8053d5bc5b46180cf/src/layers/stateless.jl

#https://github.com/FluxML/NNlib.jl/blob/master/src/logsoftmax.jl

#https://www.linkedin.com/pulse/creating-deep-neural-network-model-learn-handwritten-digits-mike-gold/


import Flux


function logreg_binary_sample(w::Array{Float64,1}, X::Array{Float64,2})
	DX, N = size(X)
	DW = length(w)
	@assert DX == DW
	p = [Flux.sigmoid(sum(w .* X[:,i])) for i in 1:N]
	labels = p .< rand(N)
	#W = reshape(w, (1, DW))
	#labels = (vec(Flux.sigmoid.(W * X)) .< rand(N))
	return labels
end


function make_test_data_binary()
	Random.seed!(1)
	D = 4;
	N = 5;
	X = randn(D, N) # note that examples are stored in the columns
	w = randn(D)
	y = logreg_binary_sample(w, X)
	return w, X, y
end

function logreg_binary_nll(w, X, y)
	D, N = size(X)
	#p1 = [Flux.sigmoid(sum(w .* X[:,i])) for i in 1:N]
	W = reshape(w, (1, D))
	p = vec(Flux.sigmoid.(W * X)) # 1xN->N vector
	#@assert isapprox(vec(p1), vec(p))
	return -sum(y .* log.(p) + (1 .- y) .* log.(1 .- p)) / N
end

function logreg_binary_nll_test()
	w, X, y = make_test_data_binary()
	D, N = size(X)
	nll1 = logreg_binary_nll(w, X, y)
	W = reshape(w, (1, D));
	logits = vec(W * X)
	nll2 =  sum(Flux.logitbinarycrossentropy.(logits, y)) / N
	@test isapprox(nll1, nll2)
end

function logreg_binary_nll_grad(w, X, y)
	D, N = size(X)
	g = zeros(Float64,D)
	for i=1:N
		x = X[:,i]
		p = Flux.sigmoid(sum(w .* x))
		g[:] = g[:] + (p - y[i]) .* x
	end
	return g / N
end

function logreg_binary_nll_grad_test()
	w0, X, y = make_test_data_binary()
	g1 = logreg_binary_nll_grad(w0, X, y)
	g2 = Flux.gradient(w -> logreg_binary_nll(w, X, y), w0)[1]
	@test isapprox(g1, g2)
end


import Optim
function logreg_fit_test()
	W, X, Y = make_test_data()
	w0 = vec(randn(size(W)))
	objective(w) = logreg_nll(w, X, Y)
	result = Optim.optimize(objective, w0, Optim.LBFGS())
	West = reshape(Optim.minimizer(result), (C,D))
	ncalls = result.iterations
end

logreg_binary_nll_test()
logreg_binary_nll_grad_test()
