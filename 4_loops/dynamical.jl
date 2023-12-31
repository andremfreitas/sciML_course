"""
`solve_system(f,u0,n)`

Solves the dynamical system

``u_{n+1} = f(u_n)``

for N steps. Returns the solution at step `n` with parameters `p`.

"""
function solve_system(f,u0,p,n)
  u = u0
  for i in 1:n-1
    u = f(u,p)
  end
  u
end


f(u,p) = u^2 - p*u

println(solve_system(f,1.0,0.25,1000))
