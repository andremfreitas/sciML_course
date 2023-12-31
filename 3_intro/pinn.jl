using Flux
using Statistics
using Plots

"""
ODE:    u' = cos(2πt)   , t ∈ [0,1]     , IC: u(0) = u_0

GOAL : approximate it with a NN (scalar=>scalar) but instead approximating the function with the NN, we will use
a transformed equation that is forced to satistfy the IC.
"""

NNODE = Chain(x -> [Float32.(x)], # Take in a scalar and transform it into an array
           Dense(1 => 32,tanh),
           Dense(32 => 1),
           first) # Take first value, i.e. return a scalar

g(t) = t*NNODE(t) + 1f0

ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ)-g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)

opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(loss())
  end
end
display(loss())
Flux.train!(loss, Flux.params(NNODE), data, opt; cb=cb)

# Plotting
t = 0:0.001:1.0
plot(t,g.(t),label="NN")
plot!(t,1.0 .+ sin.(2π.*t)/2π, label = "True Solution")
savefig("comparison.png")