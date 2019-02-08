# Examples of how to fit a binary logistic regression model using various methods.
# We consider batch (BFGS) solvers and SGD.
# We compute gradient analytically or using flux.
# We use GLM package as a unit test of correctness.
# https://stackoverflow.com/questions/49135107/logistic-regression-using-flux-jl


import Flux

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
function logreg_binary_fit_optim(X, y)
	D, N = size(X)
	w0 = randn(D)
	objective(w) = logreg_binary_nll(w, X, Y)
	grad(w) = logreg_binary_nll_grad(w, X, Y)
	solver = Optim.LBFGS(;linesearch = Optim.BackTracking())
	opts = Optim.Options(iterations = 100)
	result = Optim.optimize(objective, grad, w0, solver, opts; inplace=false)
	w_est = result.minimizer
	return w_est, result
end

import GLM, DataFrames
function logreg_binary_fit_glm(X, y)
	println("This will fail because . syntax not supported by GLM")
	D, N = size(X)
	Xt = permutedims(X)
	df = DataFrame(Xt)
	df[:y] = y
	# regress on all inputs :x1 ... :xd
	model = glm(@formula(y ~.), df, Binomial(), LogitLink())
	return coef(model)
	# The . notation is not supported in GLM.jl and the solution below fails.
	# https://stackoverflow.com/questions/44222763/specify-only-target-variable-in-glm-jl
end



function logreg_binary_fit_test()
	w, X, y = make_test_data_binary(;D=2)
	w_est1, result = logreg_binary_fit_optim(X, y)
	#w_est2 = logreg_binary_fit_glm(X, y)
	Xt = permutedims(X)
	df = DataFrame(Xt)
	df[:y] = y
	# Fit data without an intercept term by specifying -1
	model = glm(@formula(y ~ x1 + x2 -1), df, Binomial(), LogitLink())
	w_est2 = coef(model)
	@test isapprox(w_est1, w_est2; atol=0.01)
end

logreg_binary_nll_test()
logreg_binary_nll_grad_test()
logreg_binary_fit_test()


#=
# StatsModels.jl defines the Formula type and @formula macro
# https://github.com/JuliaStats/StatsModels.jl/blob/4701a1bd221f6281371f254f69c2f95c19e02a92/src/formula.jl
function make_formula_all_features(df, target)
	col_symbols = Set(names(df))
	feature_symbols = setdiff(col_symbols, Set([target]))
	feature_strings = [string(s) for s in feature_symbols]
	all_features = join(feature_strings, " + " )
	formula = string(target) * " ~ " * all_features
	return formula
end

function make_formula_test()
	n = 10
	df = DataFrame(X1=randn(n), X2=randn(n), Y=randn(n))
	ols = lm(@formula(Y ~ X1 + X2), df)
	formula = make_formula_all_features(df, :Y)
	f_expr = Meta.parse(formula)
	ols2 = lm(@formula(f_expr), df)
end
=#
