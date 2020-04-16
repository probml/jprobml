# Examples of how to fit a binary logistic regression model using various methods.
# We compute gradient analytically or using flux.
# We use GLM package as a unit test of correctness.
# https://stackoverflow.com/questions/49135107/logistic-regression-using-flux-jl


using Flux

function logreg_binary_predict(w, X)
	DX, N = size(X)
	DW = length(w)
	@assert DX == DW
	logits = [sum(w .* X[:,i]) for i in 1:N]
	probs = [Flux.sigmoid(l) for l in logits]
	#W = reshape(w, (1, D))
	#probs = vec(Flux.sigmoid.(W * X)) # 1xN->N vector
	labels = [(p > 0.5 ? 1 : 0) for p in probs]
	return probs, labels, logits
end

function logreg_binary_sample(w, X)
	probs = logreg_binary_predict(w, X)[1]
	D, N = size(X)
	labels = probs .< rand(N)
	return labels
end

function make_test_data_binary(;N=20, D=2)
	Random.seed!(1)
	X = randn(D, N) # note that examples are stored in the columns
	w = randn(D)
	y = logreg_binary_sample(w, X)
	return w, X, y
end

function logreg_binary_nll(w, X, y)
	D, N = size(X)
	p = logreg_binary_predict(w, X)[1]
	return -sum(y .* log.(p) + (1 .- y) .* log.(1 .- p)) / N
end

function logreg_binary_nll_test()
	w, X, y = make_test_data_binary()
	D, N = size(X)
	nll1 = logreg_binary_nll(w, X, y)
	probs, labels, logits = logreg_binary_predict(w, X)
	nll2 =  sum(Flux.logitbinarycrossentropy.(logits, y)) / N
	@test isapprox(nll1, nll2)
end

function logreg_binary_nll_grad(w, X, y)
	D, N = size(X)
	g = zeros(Float64,D)
	for i=1:N
		x = X[:,i]
		p = Flux.sigmoid(sum(w .* x))
		g[:] += (p - y[i]) .* x
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
function logreg_binary_fit_bfgs(X, y; w0=nothing, max_iter=100)
	D, N = size(X)
	if isnothing(w0); w0 = randn(D); end
	objective(w) = logreg_binary_nll(w, X, y)
	grad(w) = logreg_binary_nll_grad(w, X, y)
	solver = Optim.LBFGS(;linesearch = Optim.BackTracking())
	opts = Optim.Options(iterations = max_iter)
	result = Optim.optimize(objective, grad, w0, solver, opts; inplace=false)
	w_est = result.minimizer
	return w_est
end

include("first-order.jl")
function logreg_binary_fit_adam(X, y; w0=nothing, max_iter=100)
	D, N = size(X)
	if isnothing(w0); w0 = randn(D); end
	objective(w) = logreg_binary_nll(w, X, y)
	grad(w) = logreg_binary_nll_grad(w, X, y)
	solver = Adam(0.2)
	w_est = run_descent_method(solver, objective, grad, w0; max_iter=max_iter)
	return w_est
end

function logreg_binary_fit_flux(X, y; w0=nothing, max_iter=100)
	D, N = size(X)
	#initW(out, in) = reshape(w0, (1,D))
	#https://github.com/FluxML/Flux.jl/issues/332
 	#model = Chain(Dense(D, 1; initW = initW), sigmoid)
	#model = Chain(Dense(D, 1), sigmoid)
	model = Chain(Dense(D, 2), softmax)
	#loss(x, y) = Flux.binarycrossentropy(model(x), y)
	loss(x, y) = crossentropy(model(x), y)
	Y = onehotbatch(y, 0:1)
	data = (X,Y)
	callback() = @show(loss(X, Y))
	opt = Flux.Optimise.ADAM()
	for epoch=1:max_iter
		Flux.train!(loss, params(model), data, opt, cb = callback)
	end
end

#=
	D = 2; N = 20;
	X = randn(D, N)
	y = rand([0,1], N)
	D, N = size(X)
	data = repeated((X,Y),1)

	#model = Chain(Dense(D, 1), sigmoid) #  DimensionMismatch("multiplicative identity defined only for square matrices")
	#model = Dense(D, 1, sigmoid) #LoadError: no method matching eps(::TrackedArray{…,Array{Float32,2}})
	#loss(x, y) = Flux.binarycrossentropy(model(x), y)
	#Y = y
	model = Chain(Dense(D, 2), softmax)
	#model = Dense(D, 2, softmax) #  MethodError: no method matching similar(::Float32)
	loss(x, y) = crossentropy(model(x), y)
	Y = onehotbatch(y, 0:1)

	callback() = @show(loss(X, Y))
	opt = Flux.Optimise.ADAM()
	Flux.train!(loss, params(model), data, opt, cb = callback)
=#

#=
#https://github.com/FluxML/model-zoo/blob/master/vision/mnist/mlp.jl

imgs = MNIST.images() #60k-vector, imgs[1] is 28x28 matrix of Grayscale
# Stack images into one large batch
XX = hcat(float.(reshape.(imgs, :))...) # 784x60k of float

labels = MNIST.labels()
Y = onehotbatch(labels, 0:9) # 10k60k OneHotMatrix
=#


using GLM, DataFrames
#https://discourse.julialang.org/t/how-to-fit-a-glm-to-all-unnamed-features-of-arbitrary-design-matrix/20490/3
function logreg_binary_fit_glm(X, y)
	D, N = size(X)
	XT = permutedims(X)
	model = fit(GeneralizedLinearModel, XT, y, Bernoulli())
	#df_y = DataFrame(y[:, :], [:yname])
	#df_x_names = [Symbol("x$i") for i in 1:size(Xt)[2]]
	#df_x = DataFrame(Xt, df_x_names)
	#df = hcat(df_y, df_x)
	#lhs = :yname
	# include all variables :x1 to :xD but exclude intercept (hence -1)
	#rhs = Expr(:call, :+, -1, df_x_names...)
	#model = glm(@eval(@formula($(lhs) ~ $(rhs))), df, Bernoulli(), LogitLink())
	return coef(model)
end

function logreg_binary_fit_test()
	wtrue, X, y = make_test_data_binary(;D=2)
	D, N = size(X)
	Random.seed!(1)
	winit = randn(D)
	wopt = logreg_binary_fit_glm(X, y)
	ll_opt = logreg_binary_nll(wopt, X, y)
	println("glm: ll $ll_opt, w $wopt")
	expts = Tuple{Any, String}[]
	push!(expts, (logreg_binary_fit_bfgs, "bfgs"))
	push!(expts, (logreg_binary_fit_adam, "adam"))
	push!(expts, (logreg_binary_fit_flux, "flux"))
	for (solver, method_name) in expts
		west = solver(X, y, w0=winit, max_iter=100)
		ll = logreg_binary_nll(west, X, y)
		println("$method_name, ll $ll, w $west")
		#@test isapprox(west, wopt; atol=0.01)
		#@test isapprox(ll, ll_opt; atol=0.1)
	end
end

logreg_binary_nll_test()
logreg_binary_nll_grad_test()
logreg_binary_fit_test()
