"""
    _source_location(__source__)

Construct an expression that, when evaluated, gives a string like `func @ file.jl:10`
pointing to where a macro was called. The function name is only available on Julia 1.13
and later, where `Base.@__FUNCTION__` exists.
"""
function _source_location(source::LineNumberNode)
    file = source.file === nothing ? "unknown" : basename(String(source.file))
    loc = "$(file):$(source.line)"
    return if isdefined(Base, Symbol("@__FUNCTION__"))
        :(string(nameof(Base.@__FUNCTION__), " @ ", $loc))
    else
        loc
    end
end

"""
    @_issue_error str

Throw an error for something that should not happen, asking the user to report it as a
bug. This is a macro (not a function) so that the error message points to where it was
called from.
"""
macro _issue_error(str)
    loc = _source_location(__source__)
    return quote
        error(
            "[",
            $loc,
            "] ",
            $(esc(str)),
            ".\n\nPlease report this issue at https://github.com/JuliaBayes/Wren.jl/issues. Thank you!\n",
        )
    end
end

"""
    WREN_COMPILATION_DEBUG[] = true

Set this to `true` to print debug messages during model compilation.
"""
const WREN_COMPILATION_DEBUG = Ref(false)

"""
    @_compilation_debug msg

Print `msg` if `WREN_COMPILATION_DEBUG[]` is `true`, together with where this macro was
called from.
"""
macro _compilation_debug(msg)
    loc = _source_location(__source__)
    return quote
        if $WREN_COMPILATION_DEBUG[]
            @info string("[", $loc, "] ", $(esc(msg)))
        end
    end
end
