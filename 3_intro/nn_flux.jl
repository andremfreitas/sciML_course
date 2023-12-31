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

