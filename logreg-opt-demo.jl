# Examples of how to fit a logistic regression model

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
	nll1 = logreg_binary_nll(w, X, y)
	W = reshape(w, (1, length(w)));
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

"""
	logreg_sample(W, X)

Sample from a logistic regression model with C classes and D features.

Inputs:
`W`: C*D matrix of weights.
`X`: D*N matrix of feature vectors, one example per column.

Outputs:
`labels`: N-vector of integer labels from {1,...,C}.
"""
function logreg_sample(W::Array{Float64,2}, X::Array{Float64,2})
	C, DW = size(W)
	DX, N = size(X)
	@assert DW == DX
	labels = Array{Int}(undef, N)
	# Binary case
	#y = Flux.logistic.(X * w) .< rand(N)
	for i=1:N
		probs = Flux.softmax(W * X[:,i])
		dist = Distributions.Categorical(probs)
		labels[i] = rand(dist)
	end
	return labels
end
# @code_warntype logreg_sample(W,X) # LGTM


function make_test_data()
	Random.seed!(1)
	C = 3;
	D = 4;
	N = 5;
	X = randn(D, N) # note that examples are stored in the columns
	W = randn(C, D)
	Y_cat = logreg_sample(W, X)
	Y_onehot = Flux.onehotbatch(Y_cat, 1:C) # C*N matrix of bools
	@test Y_cat[1] == findfirst(x -> x, Y_onehot[:,1])
	return W, X, Y_onehot
end

"""
	logreg_nll_matrix(W, X, Y)

Compute average negative log likelihood for a logistic regression model.

Inputs:
`W`: C*D matrix of weights.
`X`: D*N matrix of feature vectors, one example per column.
`Y`: C*N one hot matrix of labels.

Outputs:
nll: -1/N * sum_{n=1}^N sum_{c=1}^C Y[c,n] * log(Pr[c,n])
 where Pr[:,n] = softmax(W*X[:,n])
"""
function logreg_nll_matrix(W::Array{Float64,2}, X::Array{Float64,2},
					Y::Flux.OneHotMatrix{Array{Flux.OneHotVector,1}})
	CW, DW = size(W)
	DX, NX = size(X)
	CY, NY = size(Y)
	@assert DW == DX
	@assert CY == CW
	@assert N == NY
	nll = 0.0
	for i=1:NX
		x = X[:,i]
		y = Y[:,i]
		probs = Flux.softmax(W*x)
		nll += -sum(y .* log.(probs)) # PMLv1 p255
	end
	return nll / N
end



"""
	logreg_nll(w, X, Y)

Compute average negative log likelihood for a logistic regression model.

Inputs:
`w`: C*D 1D vector of weights, c changing fastest.
`X`: D*N matrix of feature vectors, one example per column.
`Y`: C*N one hot matrix of labels.

Outputs:
nll: -1/N * sum_{n=1}^N sum_{c=1}^C Y[c,n] * log(Pr[c,n])
 where Pr[:,n] = softmax(W*X[:,n])
"""
function logreg_nll(w::Array{Float64,1}, X::Array{Float64,2},
					Y::Flux.OneHotMatrix{Array{Flux.OneHotVector,1}})
	D, NX = size(X)
	C, NY = size(Y)
	@assert NX == NY
	W = reshape(w, (C, D))
	return logreg_nll_matrix(W, X, Y)
end


function logreg_nll_test()
	W, X, Y = make_test_data()
	nll = logreg_nll(vec(W), X, Y)
	logits = W*X # CxD
	nll2 = Flux.logitcrossentropy(logits, Y)
	@test isapprox(nll, nll2)
end

function logreg_nll_grad(w, X, Y)
	D, NX = size(X)
	C, NY = size(Y)
	@assert NX == NY
	W = reshape(w, (C, D))
	G = zeros(Float64, C, D)
	for i=1:NX
		x = X[:,i]
		y = Y[:,i]
		probs = Flux.softmax(W*x)
		delta = probs - y
		for c=1:C
			G[c,:] += delta[c] .* x
		end
	end
	return vec(G)
end

function logreg_nll_grad_test()
	W0, X, Y = make_test_data()
	w0 = vec(W0)
	g = logreg_nll_grad(w0, X, Y)
	g2 = Flux.gradient(w -> logreg_nll(w, X, Y), w0)
	@test isapprox(g, g2)
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

#logreg_fit_test()
#logreg_sample_test()
#logreg_nll_test()
#logreg_nll_grad_test()
