using BenchmarkTools

"""
Dual numbers

Thus, to extend the idea of complex step differentiation beyond complex analytic functions, we define a new number
type, the dual number. A dual number is a multidimensional number where the sensitivity of the function is
propagated along the dual portion.

Here we will now start to use ϵ as a dimensional signifier, like i, j, or k numbers. In order for this to
work out, we need to derive an appropriate algebra for our numbers. To do this, we will look at Taylor series
to make our algebra reconstruct differentiation.

Note that the chain rule has been explicitly encoded in the derivative part.

f(a+ϵ)=f(a)+ϵf′(a)

to first order. If we have two functions

f⇝f(a)+ϵf′(a)
g⇝g(a)+ϵg′(a)

then we can manipulate these Taylor expansions to calculate combinations of these functions as follows. Using the nilpotent algebra, we have that:

(f+g)=[f(a)+g(a)]+ϵ[f′(a)+g′(a)]
(f⋅g)=[f(a)⋅g(a)]+ϵ[f(a)⋅g′(a)+g(a)⋅f′(a)]

From these we can infer the derivatives by taking the component of ϵ
. These also tell us the way to implement these in the computer.


*Computer representation*
"""

# Each function requires two pieces of information and some particular "behavior",
# so we store these in a struct. It's common to call this a "dual number":
struct Dual{T}
    val::T
    der::T
end

# Each Dual object represents a function. We define arithmetic operations to mirror
# performing those operations on the corresponding functions.
# We must first import the operations from Base:

Base.:+(f::Dual, g::Dual) = Dual(f.val + g.val, f.der + g.der)
Base.:+(f::Dual, α::Number) = Dual(f.val + α, f.der)
Base.:+(α::Number, f::Dual) = f + α

#=
You can also write:
import Base: +
f::Dual + g::Dual = Dual(f.val + g.val, f.der + g.der)
=#

Base.:-(f::Dual, g::Dual) = Dual(f.val - g.val, f.der - g.der)

# Product Rule
Base.:*(f::Dual, g::Dual) = Dual(f.val*g.val, f.der*g.val + f.val*g.der)
Base.:*(α::Number, f::Dual) = Dual(f.val * α, f.der * α)
Base.:*(f::Dual, α::Number) = α * f

# Quotient Rule
Base.:/(f::Dual, g::Dual) = Dual(f.val/g.val, (f.der*g.val - f.val*g.der)/(g.val^2))
Base.:/(α::Number, f::Dual) = Dual(α/f.val, -α*f.der/f.val^2)
Base.:/(f::Dual, α::Number) = f * inv(α) # Dual(f.val/α, f.der * (1/α))

Base.:^(f::Dual, n::Integer) = Base.power_by_squaring(f, n)  # use repeated squaring for integer powers


# We can now define Duals and manipulate them:

fd = Dual(3, 4)
gd = Dual(5, 6)

println(fd + gd)
println(fd * gd)
println(fd * (gd + gd))

#####################
## PERFORMANCE ######
#####################

add(a1, a2, b1, b2) = (a1+b1, a2+b2)

a, b, c, d = 1, 2, 3, 4
@btime add($(Ref(a))[], $(Ref(b))[], $(Ref(c))[], $(Ref(d))[])

a = Dual(1, 2)
b = Dual(3, 4)

add(j1, j2) = j1 + j2
add(a, b)
@btime add($(Ref(a))[], $(Ref(b))[])

# @code_native add(1, 2, 3, 4)         --- @code_native -- Prints the native assembly instruction
                                            # generated for running the method matching the given generic function
                                            # and type signature to io.


"""
Defining higher order primitives
--------------------------------

We can also define functions of Dual objects, using the chain rule. To speed up our derivative function, we
can directly hardcode the derivative of known functions which we call primitives. If f is a Dual representing the function f
, then exp(f) should be a Dual representing the function exp∘f
, i.e. with value exp(f(a))
 and derivative (exp∘f)′(a)=exp(f(a))f′(a)
:
"""

import Base:exp
exp(f::Dual) = Dual(exp(f.val), exp(f.val) * f.der)

println(fd)
println(exp(fd))

"""
 Differentiating arbitrary functions
 -----------------------------------


"""




"""
 Implementation of higher dimensional forward mode AD
 -----------------------------------
 We can implement derivatives of functions f:Rn→R
 by adding several independent partial derivative components to our dual numbers.

We can think of these as ϵ
 perturbations in different directions, which satisfy ϵ2i=ϵiϵj=0
, and we will call ϵ
 the vector of all perturbations. Then we have

f(a+ϵ)=f(a)+∇f(a)⋅ϵ+O(ϵ2),
where a∈Rn
 and ∇f(a)
 is the gradient of f
 at a
, i.e. the vector of partial derivatives in each direction. ∇f(a)⋅ϵ
 is the directional derivative of f
 in the direction ϵ
.

We now proceed similarly to the univariate case:

(f+g)(a+ϵ)=[f(a)+g(a)]+[∇f(a)+∇g(a)]⋅ϵ
(f⋅g)(a+ϵ)=[f(a)+∇f(a)⋅ϵ][g(a)+∇g(a)⋅ϵ]=f(a)g(a)+[f(a)∇g(a)+g(a)∇f(a)]⋅ϵ

We will use the StaticArrays.jl package for efficient small vectors:
"""

using StaticArrays

struct MultiDual{N,T}
    val::T
    derivs::SVector{N,T}
end

import Base: +, *

function +(f::MultiDual{N,T}, g::MultiDual{N,T}) where {N,T}
    return MultiDual{N,T}(f.val + g.val, f.derivs + g.derivs)
end

function *(f::MultiDual{N,T}, g::MultiDual{N,T}) where {N,T}
    return MultiDual{N,T}(f.val * g.val, f.val .* g.derivs + g.val .* f.derivs)
end

gcubic(x, y) = x*x*y + x + y

(a, b) = (1.0, 2.0)

xx = MultiDual(a, SVector(1.0, 0.0))
yy = MultiDual(b, SVector(0.0, 1.0))

println(gcubic(xx, yy))

# We can calculate the Jacobian of a function Rn→Rm
  #  by applying this to each component function:
  
fsvec(x, y) = SVector(x*x + y*y , x + y)

println(fsvec(xx, yy))

# It would be possible (and better for performance in many cases) to store all of the partials in a matrix instead. => Jacobian

"""
Forward-mode AD is implemented in a clean and efficient way in the ForwardDiff.jl package:
"""

using ForwardDiff

ForwardDiff.gradient( xx -> ( (x, y) = xx; x^2 * y + x*y ), [1, 2])