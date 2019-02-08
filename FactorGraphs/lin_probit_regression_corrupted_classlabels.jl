# Tested with Julia v1.1.0 and ForneyLab v0.9.1

using LinearAlgebra, PyPlot, ForneyLab
import Distributions: MvNormal

# Custom message update rule to interpret integers as Categorical Distributions
# with one-hot coding.
@sumProductRule(:node_type     => Transition,
                :outbound_type => Message{Categorical},
                :inbound_types => (Message{PointMass}, Nothing, Message{PointMass}),
                :name          => SPTransitionIn1PVP)

function ruleSPTransitionIn1PVP(msg_out::Message{PointMass, Univariate},
                                msg_in1::Nothing,
                                msg_a::Message{PointMass, MatrixVariate})
    # Convert integer in {1,2} to Categorical with probability vector p
    p = zeros(2)
    p[msg_out.dist.params[:m]] = 1.0
    a = msg_a.dist.params[:m]'*p

    Message(Univariate, Categorical, p=a./sum(a))
end

# EP algorithm building
function build_algorithm(dim::Int, N::Int64, alpha::Float64; output_file="ep_algo.jl")
    fg = FactorGraph()
    A = [1.0-alpha alpha; alpha 1.0-alpha] # label flipping probability matrix
    @RV w ~ GaussianMeanVariance(constant(zeros(dim)), constant(100 * eye(dim)), id=:w)
    sites = Variable[w]
    for t=1:N
        x_t = placeholder(Variable(id=:x_*t), :x, index=t, dims=(dim,))
        @RV dp_t ~ DotProduct(w, x_t, id=:dp_*t)
        @RV z_t ~ Sigmoid(dp_t, id=:z_*t) # uncorrupted label
        @RV y_t ~ Transition(z_t, constant(A, id=:A_*t), id=:y_*t) # corrupted label
        placeholder(y_t, :y, index=t, datatype=Int)
        push!(sites, dp_t)
    end

    # ForneyLab.draw(fg)
    ep_schedule = expectationPropagationSchedule(sites)
    messagePassingAlgorithm(ep_schedule, marginalSchedule(w), file=output_file)
end

# Synthetic data generation
function generate_data(N::Int, α::Float64)
    X = zeros(3, N)
    y = Vector{Int64}(undef, N)
    dists = [MvNormal([-1.0; -0.75], 0.5*eye(2)); MvNormal([1.25; 1.0], 0.3*diagm(0 => [1.0; 1.5]))]
    for i=1:N
        y[i] = rand([1;2])
        X[1:2,i] = rand(dists[y[i]])
        X[3,i] = 1.0
        if rand() < α # Flip class label
            y[i] = (y[i] == 1) ? 2 : 1
        end
    end

    return X, y
end

# Plotting
function plot_state(X, y, β_dist=false)
    figure()
    scatter(X[1,:], X[2,:], c=float(y).-1.0)
    # PyPlot.gray()

    if isa(β_dist, ProbabilityDistribution)
        # σ(β[1]*x[1] + β[2]*x[2] + β[3]) = 0.5
        # ⇛ β[1]*x[1] + β[2]*x[2] + β[3] = 0
        # ⇛ x[2] = - β[1]/β[2]*x[1] - β[3]/β[2]
        #        = a * x[1] + b
        x1 = [minimum(X[1,:]); maximum(X[1,:])]
        for i=1:100
            β = ForneyLab.sample(β_dist)
            a,b = [-1*β[1]; -1*β[3]] ./ β[2]
            plot(x1, a*x1.+b, "-", color="k", alpha=0.2)
        end

        xlim(x1)
        ylim([minimum(X[2,:]); maximum(X[2,:])])
        PyPlot.xticks([])
        PyPlot.yticks([])
    end
    # savefig("../figures/lin_class_posterior.eps")
end

# Generate training data
N = 100
α = 0.2 # Label corruption probability
# Random.seed!(1234)
X_train, y_train = generate_data(N, α)
# plot_state(X_train, y_train)

# Build inference algorithm
# Comment the following line to skip regeneration of the algorithm
build_algorithm(3, N, α, output_file="ep_algo.jl")

data = Dict(
        :x => [X_train[:,t] for t=1:N],
        :y => y_train)

# Perform inference
include("ep_algo.jl")
messages = init()
marginals = Dict()
for t=1:20
    println("Step $t")
    step!(data, marginals, messages)
end
println("DONE!")
println(marginals)

# Plot result
plot_state(X_train, y_train, marginals[:w])
savefig("lin_class_posterior_robust.pdf")
PyPlot.show()
