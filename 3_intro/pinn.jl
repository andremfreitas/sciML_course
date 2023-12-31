using Flux
NNODE = Chain(x -> [x], # Take in a scalar and transform it into an array
           Dense(1 => 32,tanh),
           Dense(32 => 1),
           first) # Take first value, i.e. return a scalar
NNODE(1.0)
