"""
    _function_name()

Construct an expression that, when evaluated, gives the name of the function that a macro
was called from. The function name is only available on Julia 1.13 and later, where
`Base.@__FUNCTION__` exists; on older versions this returns `nothing`.
"""
function _function_name()
    return if isdefined(Base, Symbol("@__FUNCTION__"))
        :(string(nameof(Base.@__FUNCTION__)))
    else
        nothing
    end
end

"""
    _source_location(__source__)

Construct an expression that, when evaluated, gives a string like `func @ file.jl:10`
pointing to where a macro was called. On Julia versions before 1.13, this is just
`file.jl:10`.
"""
function _source_location(source::LineNumberNode)
    file = source.file === nothing ? "unknown" : basename(String(source.file))
    loc = "$(file):$(source.line)"
    fname = _function_name()
    return fname === nothing ? loc : :(string($fname, " @ ", $loc))
end

"""
    @wren_error str

Throw an error for something that should not happen, asking the user to report it as a
bug. This is a macro (not a function) so that the error message points to where it was
called from.
"""
macro wren_error(str)
    loc = _source_location(__source__)
    return quote
        error(
            "[",
            $loc,
            "] ",
            $(esc(str)),
            "\n\nPlease report this issue at https://github.com/JuliaBayes/Wren.jl/issues.\nThank you!\n",
        )
    end
end

"""
    @wren_debug msg

Log `msg` as a debug message, together with the name of the function this macro was called
from (on Julia 1.13 and later). To see these messages, set `ENV["JULIA_DEBUG"] = "WrenCore"`.
"""
macro wren_debug(msg)
    # The logger already shows the file and line, so we only add the function name.
    fname = _function_name()
    full_msg = fname === nothing ? esc(msg) : :(string("[", $fname, "] ", $(esc(msg))))
    # Pass the caller's location explicitly, or else the logger would see the location of
    # the `@debug` call inside this macro.
    file = __source__.file === nothing ? "unknown" : String(__source__.file)
    line = __source__.line
    return quote
        @debug $full_msg _module = $__module__ _file = $file _line = $line
    end
end
