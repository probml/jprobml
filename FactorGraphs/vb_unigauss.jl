#https://github.com/biaslab/ForneyLab.jl/blob/master/demo/3_variational_estimation_iid_gaussian.ipynb
# Variational Bayes for a 1d Gaussian

# Generate toy data set
Random.seed!(1)
n = 5
m_data = 3.0
w_data = 4.0
y_data = sqrt(1/w_data)*randn(n) .+ m_data;

using(ForneyLab)

g = FactorGraph()

# Priors
@RV m ~ GaussianMeanVariance(0.0, 100.0)
@RV w ~ Gamma(0.01, 0.01)

# Observarion model
y = Vector{Variable}(undef, n)
for i = 1:n
    @RV y[i] ~ GaussianMeanPrecision(m, w)
    placeholder(y[i], :y, index=i)
end

# Specify recognition factorization
q = RecognitionFactorization(m, w, ids=[:M, :W])

# Inspect the subgraph for m
#ForneyLab.draw(q.recognition_factors[:M])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)
println(algo)

algo_F = freeEnergyAlgorithm(q)
println(algo_F)

# Load algorithms
eval(Meta.parse(algo))
eval(Meta.parse(algo_F));

data = Dict(:y => y_data)

# Initial recognition distributions
marginals = Dict(:m => vague(GaussianMeanVariance),
                 :w => vague(Gamma))

n_its = 2*n
F = Vector{Float64}(undef, n_its) # Initialize vector for storing Free energy
m_est = Vector{Float64}(undef, n_its)
w_est = Vector{Float64}(undef, n_its)
for i = 1:n_its
    stepM!(data, marginals)
    stepW!(data, marginals)

    # Store free energy
    F[i] = freeEnergy(data, marginals)

    # Store intermediate estimates
    m_est[i] = mean(marginals[:m])
    w_est[i] = mean(marginals[:w])
end



using Plots; pyplot()

# Plot free energy to check for convergence
plot(1:n_its, F, c=:black, lab="Free energy")
xlabel!("Iteration")
ylabel!("F")

# Plot estimated mean
ys = m_est; l = 1.0 ./ sqrt.(w_est); u = 1.0 ./ sqrt.(w_est);
plot([ys ys], fillrange=[ys.-l ys.+u], fillalpha=0.3, c=:red, lab="μ")
ylabel!("Estimate")
m_mle = mean(y_data)
m_emp = fill(m_mle, n_its)
plot!(1:n_its, m_emp, c=:blue, lab="\overline{μ}")
