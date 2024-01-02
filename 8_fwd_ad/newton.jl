using ForwardDiff, StaticArrays

function newton_step(f, x0)
    J = ForwardDiff.jacobian(f, x0)
    δ = J \ f(x0)

    return x0 - δ
end

function newton(f, x0)
    x = x0

    for i in 1:10
        x = newton_step(f, x)
        @show x
    end

    return x
end

fsvec2(xx) = ( (x, y) = xx;  SVector(x^2 + y^2 - 1, x - y) )

x0 = SVector(3.0, 5.0)

x = newton(fsvec2, x0)
