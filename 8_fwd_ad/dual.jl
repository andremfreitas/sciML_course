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

