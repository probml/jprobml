

#https://github.com/biaslab/ForneyLab.jl/blob/master/demo/1_state_estimation_forward_only.ipynb
# Kalman filtering

#Data
Random.seed!(1)
n_samples = 20
x = [t for t=1:n_samples] # True state
y = x + sqrt(200.0)*randn(n_samples); # Noisy observations of state

using ForneyLab

g = FactorGraph()

# declare priors as random variables
@RV m_x_t_min # m_x_t_min = Variable(id=:m_x_t_min)
@RV v_x_t_min # v_x_t_min = Variable(id=:v_x_t_min)
@RV x_t_min ~ GaussianMeanVariance(m_x_t_min, v_x_t_min)

# System equations
# u = 1.0; v = 200.0
@RV n_t ~ GaussianMeanVariance(0.0, 200.0)
@RV x_t = x_t_min + 1.0
@RV y_t = x_t + n_t

# Name variable for ease of lookup
#x_t.id = :x_t;

# Placeholders for prior
placeholder(m_x_t_min, :m_x_t_min) # placeholder(:m_x_t_min) does not work
placeholder(v_x_t_min, :v_x_t_min)

# Placeholder for data
placeholder(y_t, :y_t);

#ForneyLab.draw(g)

algo = sumProductAlgorithm(x_t) # Figure out a schedule and compile to Julia code
println(algo)
# Define algorithm
eval(Meta.parse(algo))

# Define values for prior statistics
m_x_0 = 0.0
v_x_0 = 1000.0

m_x = Vector{Float64}(undef, n_samples)
v_x = Vector{Float64}(undef, n_samples)


m_x_t = m_x_0
v_x_t = v_x_0
for t = 1:n_samples
    # To allow assignment to global variables (in REPL scope),
    # we need to declare them:
    # https://github.com/JuliaLang/julia/issues/28789
    global m_x_t, v_x_t
    # Prepare data and prior statistics
    data = Dict(:y_t       => y[t],
                :m_x_t_min => m_x_t,
                :v_x_t_min => v_x_t)

    # Execute algorithm
    marginals = step!(data)

    # Extract posterior statistics
    m_x_t = mean(marginals[:x_t])
    v_x_t = var(marginals[:x_t])

    # Store to buffer
    m_x[t] = m_x_t
    v_x[t] = v_x_t
end


#http://docs.juliaplots.org/latest/examples/pyplot/
#ys = Vector[rand(10), rand(20)]
#plot(ys, color=[:black :orange], line=(:dot, 4), marker=([:hex :d], 12, 0.8, Plots.stroke(3, :gray)))

#using PyPlot
using Plots; pyplot()

xs = 1:n_samples
scatter(xs, y, color=[:blue], label="y")
plot!(xs, x, color=[:black], label="x")
plot!(xs, m_x, color=[:red], label="estimated x")
#grid("on")
ys = m_x; l = sqrt.(v_x); u = sqrt.(v_x)
#fill_between(xs, y.-l, y.+_u, color="b", alpha=0.3)
plot!([ys ys], fillrange=[ys.-l ys.+u], fillalpha=0.3, c=:red)
xlabel!("t")
#legend!(loc="upper left");
fname = "Figures/forney-kalman-filter.png"
savefig(fname)


#==
plot(collect(1:n_samples), y, "b*", label="y")
plot(collect(1:n_samples), x, "k--", label="true x")

plot(collect(1:n_samples), m_x, "b-", label="estimated x")
fill_between(collect(1:n_samples), m_x-sqrt.(v_x), m_x+sqrt.(v_x), color="b", alpha=0.3);
grid("on")
xlabel("t")
legend(loc="upper left");

PyPlot.plot(collect(1:n_samples), y, "b*", label="y")
PyPlot.plot(collect(1:n_samples), x, "k--", label="true x")
PyPlot.plot(collect(1:n_samples), m_x, "b-", label="estimated x")
PyPlot.fill_between(collect(1:n_samples), m_x-sqrt.(v_x), m_x+sqrt.(v_x), color="b", alpha=0.3);
PyPlot.grid("on")
PyPlot.xlabel("t")
PyPlot.legend(loc="upper left");
=#
