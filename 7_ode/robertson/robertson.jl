using DifferentialEquations
using Sundials
using ParameterizedFunctions
using Plots

"""
Biochemistry: Robertson Equations

Biochemical equations commonly display large separation of timescales which lead to a stiffness
phenomena that will be investigated later. The classic "hard" equations for ODE integration thus
tend to come from biology (not physics!) due to this property. One of the standard models is the
Robertson model, which can be described as:
"""

function rober(du,u,p,t)
    y₁,y₂,y₃ = u
    k₁,k₂,k₃ = p
    du[1] = -k₁*y₁+k₃*y₂*y₃
    du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
    du[3] =  k₂*y₂^2
end
prob = ODEProblem(rober,[1.0,0.0,0.0],(0.0,1e5),(0.04,3e7,1e4))
sol = solve(prob,Rosenbrock23())
plot(sol)
savefig("rob_solution.png")

plot(sol, xscale=:log10, tspan=(1e-6, 1e5), layout=(3,1))
savefig("rob_sol2.png")