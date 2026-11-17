module WrenCheatsheetTests

@info "testing cheatsheets.jl"

using Test

const api_cards =
    joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "cheatsheets", "api")

# Runs one card's script in a fresh module, as a reader would run it in a new session.
function run_card(card)
    path = joinpath(api_cards, card * ".jl")
    Core.eval(@__MODULE__, :(module $(Symbol(card, "_card"))
    include($path)
    end))
    return nothing
end

@testset "cheatsheets.jl" begin
    @testset "API card $card" for card in ("models", "conditioning", "inference", "chains")
        @test run_card(card) === nothing
    end
end

end
