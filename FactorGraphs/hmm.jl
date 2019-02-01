
# Variational Bayes for a discrete observation HMM

#https://github.com/biaslab/ForneyLab.jl/blob/master/demo/8_hidden_markov_model_estimation.ipynb

using ForneyLab

n_samples = 20
A_data = [0.9 0.0 0.1; 0.1 0.9 0.0; 0.0 0.1 0.9] # Transition probabilities (some transitions are impossible)
B_data = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9] # Observation noise

s_0_data = [1.0, 0.0, 0.0] # Initial state

# Generate some data
Random.seed!(1)
s_data = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the states
x_data = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the observations
s_t_min_data = s_0_data
for t = 1:n_samples
    global s_t_min_data
    a = A_data*s_t_min_data
    s_data[t] = sample(ProbabilityDistribution(Categorical, p=a./sum(a))) # Simulate state transition
    b = B_data*s_data[t]
    x_data[t] = sample(ProbabilityDistribution(Categorical, p=b./sum(b))) # Simulate observation

    s_t_min_data = s_data[t]
end
;

g = FactorGraph()

@RV A ~ Dirichlet(ones(3,3)) # Vague prior on transition model
@RV B ~ Dirichlet([10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0]) # Stronger prior on observation model
@RV s_0 ~ Categorical(1/3*ones(3))

s = Vector{Variable}(undef, n_samples) # one-hot coding
x = Vector{Variable}(undef, n_samples) # one-hot coding
s_t_min = s_0
for t = 1:n_samples
    global s_t_min
    @RV s[t] ~ Transition(s_t_min, A)
    @RV x[t] ~ Transition(s[t], B)

    s_t_min = s[t]

    placeholder(x[t], :x, index=t, dims=(3,))
end;

# Define the recognition factorization
q = RecognitionFactorization(A, B, [s_0; s], ids=[:A, :B, :S])

# Generate VMP algorithm
algo = variationalAlgorithm(q)

# Construct variational free energy evaluation code
algo_F = freeEnergyAlgorithm(q);

# Load algorithms
eval(Meta.parse(algo))
eval(Meta.parse(algo_F))

# Initial recognition distributions
marginals = Dict{Symbol, ProbabilityDistribution}(
    :A => vague(Dirichlet, (3,3)),
    :B => vague(Dirichlet, (3,3)))

# Initialize data
data = Dict(:x => x_data)
n_its = 20

# Run algorithm
F = Vector{Float64}(undef, n_its)
for i = 1:n_its
    stepS!(data, marginals)
    stepB!(data, marginals)
    stepA!(data, marginals)

    F[i] = freeEnergy(data, marginals)
end
;

using PyPlot

# Plot free energy
clf()
plot(1:n_its, F, color="black")
grid("on")
xlabel("Iteration")
ylabel("Free Energy")
xlim(0,n_its);
gcf()

figure(figsize=(10,5))
# Collect state estimates
x_obs = [findfirst(x_i.==1.0) for x_i in x_data]
s_true = [findfirst(s_i.==1.0) for s_i in s_data]

# Plot simulated state trajectory and observations
clf()
subplot(211)
plot(1:n_samples, x_obs, "k*", label="Observations x", markersize=7)
plot(1:n_samples, s_true, "k--", label="True state s")
yticks([1.0, 2.0, 3.0], ["Red", "Green", "Blue"])
grid("on")
xlabel("Time")
legend(loc="upper left")
xlim(0,n_samples)
ylim(0.9,3.1)
title("Data set and true state trajectory")
gcf()

# Plot inferred state sequence
subplot(212)
m_s = [mean(marginals[:s_*t]) for t=1:n_samples]
m_s_1 = [m_s_t[1] for m_s_t in m_s]
m_s_2 = [m_s_t[2] for m_s_t in m_s]
m_s_3 = [m_s_t[3] for m_s_t in m_s]

fill_between(1:n_samples, zeros(n_samples), m_s_1, color="red")
fill_between(1:n_samples, m_s_1, m_s_1 + m_s_2, color="green")
fill_between(1:n_samples, m_s_1 + m_s_2, ones(n_samples), color="blue")
xlabel("Time")
ylabel("State belief")
grid("on")
title("Inferred state trajectory");
gcf()

# True state transition probabilities
clf()
PyPlot.plt[:matshow](A_data, vmin=0.0, vmax=1.0)
ttl = title("True state transition probabilities")
ttl[:set_position]([.5, 1.15])
yticks([0, 1, 2], ["Red", "Green", "Blue"])
xticks([0, 1, 2], ["Red", "Green", "Blue"], rotation="vertical")
colorbar()
gcf()

# Inferred state transition probabilities
PyPlot.plt[:matshow](mean(marginals[:A]),  vmin=0.0, vmax=1.0)
ttl = title("Inferred state transition probabilities")
ttl[:set_position]([.5, 1.15])
yticks([0, 1, 2], ["Red", "Green", "Blue"])
xticks([0, 1, 2], ["Red", "Green", "Blue"], rotation="vertical")
colorbar();
gcf()
