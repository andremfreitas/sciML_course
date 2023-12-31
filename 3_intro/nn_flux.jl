using Flux

# Defining a NN using Flux
NN2 = Chain(Dense(10 => 32,tanh),
           Dense(32 => 32,tanh),
           Dense(32 => 5))
println(NN2(Float32.(rand(10))))

# Flux.jl as a library is written in pure Julia, which means  that every
# piece of this syntax is just sugar over some Julia code that we can specialize ourselves

# For example, the activation function is just a scalar Julia function. If we wanted to replace
# it by something like the quadratic function,we can just use an anonymous function to define
# the scalar function we would like to use:

NN3 = Chain(Dense(10 => 32,x->x^2),
            Dense(32 => 32,x->max(0,x)),
            Dense(32 => 5))
println(NN3(Float32.(rand(10))))

# Digging into the construction of a NN library
W = [randn(32,10),randn(32,32),randn(5,32)]
b = [zeros(32),zeros(32),zeros(5)]
simpleNN(x) = W[3]*tanh.(W[2]*tanh.(W[1]*x + b[1]) + b[2]) + b[3]

using InteractiveUtils
println(@which Dense(10 => 32,tanh))    # @which returns some function info and the source code location
