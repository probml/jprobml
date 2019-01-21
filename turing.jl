using Turing
using StatPlots

# Define a simple Normal model with unknown mean and variance.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
end

#  Run sampler, collect results
c1 = sample(gdemo(1.5, 2), SMC(1000))
c2 = sample(gdemo(1.5, 2), PG(10,1000))
c3 = sample(gdemo(1.5, 2), HMC(1000, 0.1, 5))
c4 = sample(gdemo(1.5, 2), Gibbs(1000, PG(10, 2, :m), HMC(2, 0.1, 5, :s)))
c5 = sample(gdemo(1.5, 2), HMCDA(1000, 0.15, 0.65))
c6 = sample(gdemo(1.5, 2), NUTS(1000,  0.65))

# Summarise results (currently requires the master branch from MCMCChain)
describe(c3)

# Plot and save results
p = plot(c3)
savefig("gdemo-plot.png")