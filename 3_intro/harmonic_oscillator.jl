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