using Turing
using StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
end

#  Run sampler, collect results
niters = 10
nparticles = 5
x = 1.5; y = 2;
@info "Starting SMC"
c1 = sample(gdemo(x, y), SMC(niters))
@info "Starting PG"
c2 = sample(gdemo(x, y), PG(nparticles, niters))
@info "Starting HMC"
c3 = sample(gdemo(x, y), HMC(niters, 0.1, 5))
@info "Staring Gibbs"
c4 = sample(gdemo(x, y), Gibbs(niters, PG(10, 2, :m), HMC(2, 0.1, 5, :s)))
@info "starting HMCDA"
c5 = sample(gdemo(x, y), HMCDA(niters, 0.15, 0.65))
@info "Starting NUTS"
c6 = sample(gdemo(x, y), NUTS(niters,  0.65))
@info "Done"
# Summarise results (currently requires the master branch from MCMCChain)
describe(c3)

# Plot and save results
p = plot(c3)
savefig("gdemo-plot.png")
