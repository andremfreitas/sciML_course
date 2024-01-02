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

# Defining the ODE problem through DifferentialEquations.jl
prob = ODEProblem(lorenz,u0,tspan,p)

# Solve the problem by calling 'solve'
sol = solve(prob)

# Plot solution
plot(sol)
savefig("solution.png")

# We can also plot phase space diagrams by telling it which vars to
# compare on which axis. Let's plot this in the (x,y,z) plane:
plot(sol,vars=(1,2,3))
savefig("phase_space_diagram.png")

# Note that the sentinal to time is 0, so we can also do (t,y,z) with:
plot(sol,vars=(0,2,3))
savefig("time_yz.png")

println(sol(0.5))
println(sol(0.255555))
println(sol(0.255556))