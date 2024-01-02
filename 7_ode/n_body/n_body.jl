using OrdinaryDiffEq
using Plots

"""
N-body problems and Astronomy

An example of this is the Pleiades problem, which is an approximation to a 7-star chaotic system.
 It can be written as:
"""

function pleiades(du,u,p,t)
  @inbounds begin
  x = view(u,1:7)   # x
  y = view(u,8:14)  # y
  v = view(u,15:21) # x′
  w = view(u,22:28) # y′
  du[1:7] .= v
  du[8:14].= w
  for i in 15:28
    du[i] = zero(u[1])
  end
  for i=1:7,j=1:7
    if i != j
      r = ((x[i]-x[j])^2 + (y[i] - y[j])^2)^(3/2)
      du[14+i] += j*(x[j] - x[i])/r
      du[21+i] += j*(y[j] - y[i])/r
    end
  end
  end
end
tspan = (0.0,3.0)
prob = ODEProblem(pleiades,[3.0,3.0,-1.0,-3.0,2.0,-2.0,2.0,3.0,-3.0,2.0,0,0,-4.0,4.0,0,0,0,0,0,1.75,-1.5,0,0,0,-1.25,1,0,0],tspan)

sol = solve(prob,Vern8(),abstol=1e-10,reltol=1e-10)
plot(sol)
savefig("pleiades_solution.png")

tspan = (0.0,200.0)
prob = ODEProblem(pleiades,[3.0,3.0,-1.0,-3.0,2.0,-2.0,2.0,3.0,-3.0,2.0,0,0,-4.0,4.0,0,0,0,0,0,1.75,-1.5,0,0,0,-1.25,1,0,0],tspan)
sol = solve(prob,Vern8(),abstol=1e-10,reltol=1e-10)
plot(sol,vars=((1:7),(8:14)))
savefig("pleiades_trajectory.png")