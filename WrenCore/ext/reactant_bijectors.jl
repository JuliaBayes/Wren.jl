# Keep parameter-dependent bounds inside the compiler's control flow.
Reactant.@reactant_overlay function Plaice.with_logabsdet_jacobian(
    t::Plaice.Truncate,
    y::Number,
)
    Reactant.@trace result = if isfinite(t.lower)
        if isfinite(t.upper)
            width = t.upper - t.lower
            a = abs(y)
            (width * Plaice.logistic(y) + t.lower, log(width) - a - 2 * Plaice.log1pexp(-a))
        else
            (exp(y) + t.lower, y)
        end
    elseif isfinite(t.upper)
        (t.upper - exp(y), y)
    else
        (y, zero(y))
    end
    return result
end

Reactant.@reactant_overlay (t::Plaice.Truncate)(y::Number) =
    first(Plaice.with_logabsdet_jacobian(t, y))
