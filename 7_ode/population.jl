using DifferentialEquations
using Plots

function lotka(du,u,p,t)
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
  end
  
  p = [1.5,1.0,3.0,1.0]
  prob = ODEProblem(lotka,[1.0,1.0],(0.0,10.0),p)
  sol = solve(prob)
  plot(sol)
  savefig("population_solution.png")
