# Examples of how to fit a logistic regression model

#https://fluxml.ai/Flux.jl/v0.1.1/examples/logreg.html
#http://marubon-ds.blogspot.com/2018/05/deep-learning-with-julia-introduction.html

#https://github.com/FluxML/Flux.jl/blob/1cf37ab9eb806368d5702ab8053d5bc5b46180cf/src/layers/stateless.jl

#https://github.com/FluxML/NNlib.jl/blob/master/src/logsoftmax.jl

#https://www.linkedin.com/pulse/creating-deep-neural-network-model-learn-handwritten-digits-mike-gold/


import Flux


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


#logreg_fit_test()
#logreg_sample_test()
#logreg_nll_test()
#logreg_nll_grad_test()
