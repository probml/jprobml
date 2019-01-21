#https://discourse.julialang.org/t/learning-bayesian-data-analysis-using-julia/5370/14

# mu ~ N(mu0, 1/tau0)
# 1/sigma^2 ~ Ga(alpha, beta)
# xn ~ N(mu, sigma)

using Distributions

function mc(x, iterations, μ0 = 0., τ0 = 1e-7, α = 0.0001, β = 0.0001)
    n = length(x)
    sumx = sum(x)
    μ, τ = sumx/n, 1/var(x)
    μs, τs = Float64[], Float64[]
    for i in 1:iterations
        μ = rand(Normal((τ0*μ0 + τ*sumx)/(τ0 + n*τ), 1/sqrt(τ0 + n*τ)))
        τ = rand(Gamma(α + n/2, 1/(β + 0.5*sum((xᵢ-μ)^2 for xᵢ in x))))
        push!(μs, μ)
        push!(τs, τ)
    end
    μs, τs
end

n = 100
mu = 0.2
sigma = 1.7^(-0.5)
x = rand(Normal(mu, sigma), n)
nsamples = 1000
μs, τs = mc(x, nsamples)
println("mean μ = ", mean(μs), " ± ", std(μs), ", truth = ", mu) 
@printf("mean μ = %5.3f,  ± std(μs) = %5.3f, truth = %5.3f", mean(μs), std(μs), mu) 
println("precision τ = ", mean(τs), " ± ", std(τs), ", truth = ", 1/sigma) 