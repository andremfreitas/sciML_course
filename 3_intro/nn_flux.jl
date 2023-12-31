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

#=# ########################################################################

# Source code for Dense

struct Dense{F, M<:AbstractMatrix, B}
    weight::M
    bias::B
    σ::F
    function Dense(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
      b = create_bias(W, bias, size(W,1))
      new{F,M,typeof(b)}(W, b, σ)
    end
  end
  
function Dense((in, out)::Pair{<:Integer, <:Integer}, σ = identity;
                 init = glorot_uniform, bias = true)
    Dense(init(out, in), bias, σ)
  end
=#  #######################################################################3

# So okay, Dense objects are just functions that have weight and
#  bias matrices inside of them. Now what does *Chain* do?

println(@which Chain(1,2,3))

#=
struct Chain{T<:Tuple}
  layers::T
  Chain(xs...) = new{typeof(xs)}(xs)
end

applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))

(c::Chain)(x) = applychain(c.layers, x)
=#

# Note: The ... is known that the slurp operator, which allows for "slurping up"
# multiple arguments into a single object

# Note2: everything (including functions) boils down to structs in julia


#############
## TRAINING #
#############

# "Training" a neural network is simply the process of finding weights that minimize a loss function.

NN = Chain(Dense(10 => 32,tanh),
           Dense(32 => 32,tanh),
           Dense(32 => 5))
loss() = sum(abs2,sum(abs2,NN(rand(10)).-1) for i in 1:100)

# 'params' is a helper function on Chain which recursively gathers all of the defining parameters. 
p = Flux.params(NN)

Flux.train!(loss, p, Iterators.repeated((), 10000), ADAM(0.1))

println(loss())
