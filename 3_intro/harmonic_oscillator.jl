using Flux
using DifferentialEquations
using Plots

"""
Let's assume that we are taking measurements of (position,force)
in some real one-dimensional spring pushing and pulling against a wall.

By Hookes law we know that the force acting on the spring is:
                F_spring = -kx

To make the problem more complicated, we assume we have spring with some mass deformities, which
leads to a force:
                F_spring = -kx + 0.1sin(x)

By Newton's second law (F = ma), we have (mass=1):

                F_spring = acceleration

                        <=>

                x'' = -kx + 0.1sin(x)
"""

# We can use DifferentialEquations.jl to solve this ODE
k = 1.0
force(dx,x,k,t) = -k*x + 0.1sin(x)
prob = SecondOrderODEProblem(force,1.0,0.0,(0.0,10.0),k)
sol = solve(prob)
plot(sol,label=["Velocity" "Position"])
savefig("oscillator.png")   

"""
Let's say we want to learn how to predict the force applied on the spring at each point in space, F(x). 
We want to learn a function, so this is the job for machine learning! However, we only have 6 measurements, 
which includes the information about (position,velocity,force) at evenly spaced times:
"""
plot_t = 0:0.01:10
data_plot = sol(plot_t)
positions_plot = [state[2] for state in data_plot]
force_plot = [force(state[1],state[2],k,plot_t) for state in data_plot]

# Generate the dataset
t = 0:3.3:10
dataset = sol(t)
position_data = [state[2] for state in sol(t)]
force_data = [force(state[1],state[2],k,t) for state in sol(t)]

plot(plot_t,force_plot,xlabel="t",label="True Force")
scatter!(t,force_data,label="Force Measurements")
savefig("data_assim_problem_oscillator.png")

"""
Can we train a neural network to approximate the expected force at any location for this spring? 
To see whether this is possible with a standard neural network, let's just do it. Let's define a neural 
network to be F(x) and see if we can learn the force function!
"""

NNForce = Chain(x -> [Float32.(x)],
           Dense(1 => 32,tanh),
           Dense(32 => 1),
           first)

loss() = sum(abs2,NNForce(position_data[i]) - force_data[i] for i in 1:length(position_data))

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
Flux.train!(loss, Flux.params(NNForce), data, opt; cb=cb)

learned_force_plot = NNForce.(positions_plot)

plot(plot_t,force_plot,xlabel="t",label="True Force")
plot!(plot_t,learned_force_plot,label="Predicted Force")
scatter!(t,force_data,label="Force Measurements")
savefig("NN_oscillator_results.png")

"""
Ouch. The problem is that a neural network can approximate any function, so it approximated
a function that fits the data, but not the correct function. We somehow need to have more data... 
but where can we get more data?

Well, even a first year undergrad in physics will know Hooke's law. Let's explore this (although in reality we know
that our system is slightly different from the usual Hooke's law).
"""

force2(dx,x,k,t) = -k*x
prob_simplified = SecondOrderODEProblem(force2,1.0,0.0,(0.0,10.0),k)
sol_simplified = solve(prob_simplified)
plot(sol,label=["Velocity" "Position"])
plot!(sol_simplified,label=["Velocity Simplified" "Position Simplified"])
savefig("sol_simplified.png")

random_positions = [2rand()-1 for i in 1:100] # random values in [-1,1]
loss_ode() = sum(abs2,NNForce(x) - (-k*x) for x in random_positions)

# λ is some weight factor to control the regularization against the physics assumption
λ = 0.1
composed_loss() = loss() + λ*loss_ode()

# Training
opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(composed_loss())
  end
end
display(composed_loss())
Flux.train!(composed_loss, Flux.params(NNForce), data, opt; cb=cb)

learned_force_plot = NNForce.(positions_plot)

plot(plot_t,force_plot,xlabel="t",label="True Force")
plot!(plot_t,learned_force_plot,label="Predicted Force")
scatter!(t,force_data,label="Force Measurements")
savefig("comp_pinn_oscillator.png")