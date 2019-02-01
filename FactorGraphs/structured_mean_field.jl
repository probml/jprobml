
# Mean field vs structured mean field for a 1d Gaussian SSM with unknown process noise variance
#https://github.com/biaslab/ForneyLab.jl/blob/master/demo/7_structured_variational_estimation.ipynb

Random.seed!(1)
n_samples = 100
v_data = 1.0
w_data = 10.0
s_0_data = 0.0

s_data = Vector{Float64}(undef, n_samples)
x_data = Vector{Float64}(undef, n_samples)
s_t_min_data = s_0_data
for t = 1:n_samples
    global s_t_min_data
    s_data[t] = s_t_min_data + sqrt(1/w_data)*randn() # Evolution model
    x_data[t] = s_data[t] + sqrt(v_data)*randn() # Observation model

    s_t_min_data = s_data[t]
end
;

using ForneyLab

g = FactorGraph()

@RV w ~ Gamma(0.01, 0.01)
@RV s_0 ~ GaussianMeanVariance(0.0, 100.0)

s = Vector{Variable}(undef, n_samples)
x = Vector{Variable}(undef, n_samples)
s_t_min = s_0
for t = 1:n_samples
    global s_t_min
    @RV s[t] ~ GaussianMeanPrecision(s_t_min, w)
    @RV x[t] ~ GaussianMeanVariance(s[t], v_data)

    s_t_min = s[t]

    placeholder(x[t], :x, index=t)
end
;


# Define recognition factorization
RecognitionFactorization()

q_w = RecognitionFactor(w)
q_s_0 = RecognitionFactor(s_0)

q_s = Vector{RecognitionFactor}(undef, n_samples)
for t=1:n_samples
    q_s[t] = RecognitionFactor(s[t])
end

println("generate code")
algo_w_mf = variationalAlgorithm(q_w, name="WMF")
algo_s_mf = variationalAlgorithm([q_s_0; q_s], name="SMF")
algo_F_mf = freeEnergyAlgorithm(name="MF");


println("compilinig")
eval(Meta.parse(algo_w_mf))
eval(Meta.parse(algo_s_mf))
eval(Meta.parse(algo_F_mf))

# Initialize data
data = Dict(:x => x_data)
n_its = 40

# Initial recognition distributions
marginals_mf = Dict{Symbol, ProbabilityDistribution}(:w => vague(Gamma))
for t = 0:n_samples
    marginals_mf[:s_*t] = vague(GaussianMeanPrecision)
end

# Run algorithm
F_mf = Vector{Float64}(undef, n_its)
for i = 1:n_its
    println("iteration $i")
    stepWMF!(data, marginals_mf)
    stepSMF!(data, marginals_mf)

    F_mf[i] = freeEnergyMF(data, marginals_mf)
end
;

# Define the recognition factorization
q_struct = RecognitionFactorization(w, [s_0; s]; ids=[:WStruct, :SStruct])

println("generate code")
algo_struct = variationalAlgorithm(q_struct)
algo_F_struct = freeEnergyAlgorithm(name="Struct")


println("compile code")
eval(Meta.parse(algo_struct))
eval(Meta.parse(algo_F_struct))

# Initial recognition distributions
marginals_struct = Dict{Symbol, ProbabilityDistribution}(:w => vague(Gamma))

# Run algorithm
F_struct = Vector{Float64}(undef, n_its)
for i = 1:n_its
    println("iteration $i")
    stepSStruct!(data, marginals_struct)
    stepWStruct!(data, marginals_struct)

    F_struct[i] = freeEnergyStruct(data, marginals_struct)
end
;

using PyPlot

# Collect state estimates
m_s_mf = [mean(marginals_mf[:s_*t]) for t=0:n_samples]
v_s_mf = [var(marginals_mf[:s_*t]) for t=0:n_samples]
m_s_struct = [mean(marginals_struct[:s_*t]) for t=0:n_samples]
v_s_struct = [var(marginals_struct[:s_*t]) for t=0:n_samples]

# Plot estimated state
figure(figsize=(8,4))
#subplot(211)
plot(collect(1:n_samples), x_data, "*", label="Data", color="black")
plot(collect(0:n_samples), [0.0; s_data], "k-", label="True hidden state")
plot(collect(0:n_samples), m_s_struct, "b-", label="Structured state estimate")
fill_between(collect(0:n_samples), m_s_struct-sqrt.(v_s_struct), m_s_struct+sqrt.(v_s_struct), color="b", alpha=0.3);
plot(collect(0:n_samples), m_s_mf, "r-", label="Mean-field state estimate")
fill_between(collect(0:n_samples), m_s_mf-sqrt.(v_s_mf), m_s_mf+sqrt.(v_s_mf), color="r", alpha=0.3);
grid("on")
xlabel("Time")
legend(loc="upper left", bbox_to_anchor=[-0.05, 1.5]);
gcf()

#subplot(212)
clf()
plot(1:n_its, F_struct, color="blue", label="Structured")
plot(1:n_its, F_mf, color="red", label="Mean-field")
grid("on")
legend()
yscale("log")
xlabel("VMP iteration")
ylabel("Variational free energy")
tight_layout();
gcf()
