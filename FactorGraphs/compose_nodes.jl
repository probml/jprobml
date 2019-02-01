
# Composite nodes
#https://github.com/biaslab/ForneyLab.jl/blob/master/demo/6_composite_nodes.ipynb

using ForneyLab

# Define factor graph for x1 = x0 + b*u1, where x0 and u1 have Gaussian priors, and b is a constant.
# This is a part of the information filter graph from the introduction.
g = FactorGraph()

b = [1.0; 0.5]

@RV x_0 ~ GaussianMeanVariance(ones(2), eye(2))
@RV u_1 ~ GaussianMeanVariance(1.0, 1.0)
@RV x_1 = x_0 + b*u_1;

flat_schedule = sumProductSchedule(x_1)

ForneyLab.draw(g, schedule=flat_schedule) # Inspect the resulting schedule

println(flat_schedule)

# Define a composite node for z = x + b*y
@composite GainAddition (z, x, y) begin
    # Specify the 'internal factor graph' of the GainAddion composite node.
    # z, x, and y can be used as if they are existing Variables in this block.
    b = [1.0; 0.5]

    @RV z = x + b*y
end

g2 = FactorGraph()

@RV x_0 ~ GaussianMeanVariance(ones(2), eye(2))
@RV u_1 ~ GaussianMeanVariance(1.0, 1.0)
@RV x_1 ~ GainAddition(x_0, u_1);

composite_schedule = sumProductSchedule(x_1)

ForneyLab.draw(g2, schedule=composite_schedule)

println(composite_schedule)

println(composite_schedule[end].internal_schedule)

@sumProductRule(:node_type     => GainAddition,                                 # our custom composite node
                :outbound_type => Message{GaussianMeanPrecision},               # this rule produces a GaussianMeanPrecision msg
                :inbound_types => (Nothing, Message{Gaussian}, Message{Gaussian}), # msg towards first interface, incoming types
                :name          => SPGainAdditionOutVGG)                         # name of the update rule;

shortcut_schedule = sumProductSchedule(x_1)
println(shortcut_schedule)
