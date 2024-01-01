using Plots

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
println(solve_system(f,1.22,0.25,1000))
println(solve_system(f,1.25,0.25,1000))
println(solve_system(f,1.251,0.25,20))


# Multi dimensional systems implementation

function lorenz(u,p)
  α,σ,ρ,β = p
  du1 = u[1] + α*(σ*(u[2]-u[1]))
  du2 = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
  du3 = u[3] + α*(u[1]*u[2] - β*u[3])
  [du1,du2,du3]
end
p = (0.02,10.0,28.0,8/3)
solve_system(lorenz,[1.0,0.0,0.0],p,1000)

function solve_system_save(f,u0,p,n)
  u = Vector{typeof(u0)}(undef,n)
  u[1] = u0
  for i in 1:n-1
    u[i+1] = f(u[i],p)
  end
  u
end
to_plot = solve_system_save(lorenz,[1.0,0.0,0.0],p,1000)

x = [to_plot[i][1] for i in 1:length(to_plot)]
y = [to_plot[i][2] for i in 1:length(to_plot)]
z = [to_plot[i][3] for i in 1:length(to_plot)]
plot(x,y,z)
savefig("lorentz.png")



# Use push! instead of indexing
function solve_system_save_push(f,u0,p,n)
  u = Vector{typeof(u0)}(undef,1)
  u[1] = u0
  for i in 1:n-1
    push!(u,f(u[i],p))
  end
  u
end

@time solve_system_save(lorenz,[1.0,0.0,0.0],p,1000)
@time solve_system_save_push(lorenz,[1.0,0.0,0.0],p,1000)

# Its also possible to do this comparison using benchmark tools
using BenchmarkTools

@btime solve_system_save(lorenz,[1.0,0.0,0.0],p,1000)
@btime solve_system_save_push(lorenz,[1.0,0.0,0.0],p,1000)

function solve_system_save_matrix(f,u0,p,n)
  u = Matrix{eltype(u0)}(undef,length(u0),n)
  u[:,1] = u0
  for i in 1:n-1
    u[:,i+1] = f(u[:,i],p)
  end
  u
end

@btime solve_system_save_matrix(lorenz,[1.0,0.0,0.0],p,1000)

# Where is this cost coming from? A large portion of the cost is due 
# to the slicing on the u, which we can fix with a view:

function solve_system_save_matrix_view(f,u0,p,n)
  u = Matrix{eltype(u0)}(undef,length(u0),n)
  u[:,1] = u0
  for i in 1:n-1
    u[:,i+1] = f(@view(u[:,i]),p)
  end
  u
end
@btime solve_system_save_matrix_view(lorenz,[1.0,0.0,0.0],p,1000)


# Since we are only ever using single columns as a unit, notice that there isn't any benefit
# to keeping the whole thing contiguous, and in fact there are some downsides (cache is harder
# to optimize because the longer cache lines are unnecessary, the views need to be used). Also, 
# growing the matrix adaptively is not a very good idea since every growth requires both allocating
# memory and copying over the old values:

function solve_system_save_matrix_resize(f,u0,p,n)
  u = Matrix{eltype(u0)}(undef,length(u0),1)
  u[:,1] = u0
  for i in 1:n-1
    u = hcat(u,f(@view(u[:,i]),p))
  end
  u
end
@btime solve_system_save_matrix_resize(lorenz,[1.0,0.0,0.0],p,1000)


"""
This lecture continues on but it is not very useful for me atm so I'll skip it for now.
"""