using DifferentialEquations
using Plots

"""

The Lorenz equations are three ODEs, which describe a chaotic dynamical system:

 dx/dt =   σ (y - x) 
 dy/dt = x (ρ - z) - y 
 dz/dt = xy - βz

 """


function lorenz(du,u,p,t)
 du[1] = p[1]*(u[2]-u[1])
 du[2] = u[1]*(p[2]-u[3]) - u[2]
 du[3] = u[1]*u[2] - p[3]*u[3]
end

# Initial Conditions
u0 = [1.0,0.0,0.0]

# Time span to solve the problem
tspan = (0.0,100.0)

# Define our parameters
p = (10.0,28.0,8/3)