
# Expectation propogation for a 1D linear Gaussian SSM with binary observations

#https://github.com/biaslab/ForneyLab.jl/blob/master/demo/5_expectation_propagation.ipynb


using SpecialFunctions
using Random
# Generate data set

Random.seed!(123)
n_samples = 40
σ(x) = 0.5 + 0.5*erf.(x./sqrt(2))

u_data = 0.1
x_data = []
y_data = []
x_prev = -2.0
for t=1:n_samples
    global x_prev
    push!(x_data, x_prev + u_data + sqrt(0.01)*randn()) # State transition
    push!(y_data, σ(x_data[end]) > rand()); # Observation
    x_prev = x_data[end]
end

using ForneyLab

g = FactorGraph()

# State prior
@RV x_0 ~ GaussianMeanVariance(0.0, 100.0)

x = Vector{Variable}(undef, n_samples)
d = Vector{Variable}(undef, n_samples)
y = Vector{Variable}(undef, n_samples)
x_t_min = x_0
for t = 1:n_samples
    global x_t_min
    @RV d[t] ~ GaussianMeanVariance(u_data, 0.01)
    @RV x[t] = x_t_min + d[t]
    @RV y[t] ~ Sigmoid(x[t])

    # Data placeholder
    placeholder(y[t], :y, index=t)

    # Reset state for next step
    x_t_min = x[t]
end

println("generating code")
algo = expectationPropagationAlgorithm(x);
println(algo) # Uncomment to inspect algorithm code

println("compiling code")
eval(Meta.parse(algo));

messages = init()
marginals = Dict()
data = Dict(:y => y_data)

n_its = 4*n_samples
for i = 1:n_its
    println("iteration $i")
   step!(data, marginals, messages)
end


using PyPlot

clf()
# Extract posterior statistics
m_x = [mean(marginals[:x_*t]) for t = 1:n_samples]
v_x = [var(marginals[:x_*t]) for t = 1:n_samples]

plot(collect(1:n_samples), x_data, "k--", label="true x")
plot(collect(1:n_samples), m_x, "b-", label="estimated x")
fill_between(collect(1:n_samples), m_x-sqrt.(v_x), m_x+sqrt.(v_x), color="b", alpha=0.3);
grid("on")
xlabel("t")
xlim(1, n_samples)
ylim(-2, 2)
legend(loc=7)

ax = gca()
ax[:twinx]()
plot(collect(1:n_samples), y_data, "b*", label="y")
yticks([0.0, 1.0], ["False", "True"]);
ylim(-0.1, 1.1);
gcf()
