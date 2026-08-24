module WrenRewriteTests

using Wren
using Wren: remove_plate, add_plate, set_arg
using Test
using Distributions
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using LinearAlgebra: I
using Random: Xoshiro
using VarNames

@wren function rw_inner()
    a ~ Normal()
    b ~ Normal(a)
end

@wren function rw_base()
    x ~ Normal()
    y ~ Normal()
end

# Hand-written models that rewriting should reproduce exactly.
@wren function hand_lognormal()
    x ~ LogNormal()
    y ~ Normal()
end
@wren function hand_submodel()
    x ~ rw_inner()
    y ~ Normal()
end
@wren function hand_fixed()
    x = 1.5
    y ~ Normal()
end

# Two-level nesting (defined at module level so that the inner model
# functions resolve inside the models' evaluation functions).
@wren function nested_mid()
    inner ~ rw_inner()
    c ~ Normal()
end
@wren function nested_top()
    m ~ nested_mid()
    d ~ Normal()
end

# Plated models, plus hand-written equivalents of rewriting the plated
# site family.
@wren function plated(N)
    y = zeros(N)
    x ~ Normal()
    @plate for i in 1:N
        y[i] ~ Normal(x)
    end
end
@wren function plated_exp(N)
    y = zeros(N)
    x ~ Normal()
    @plate for i in 1:N
        y[i] ~ Exponential(2.0)
    end
end
@wren function plated_idx(N)
    y = zeros(N)
    x ~ Normal()
    @plate for i in 1:N
        y[i] ~ Normal(x, i)
    end
end

# A model with literal indexed sites.
@wren function indexed_model(sub)
    x = zeros(2)
    x[1] ~ sub
    x[2] ~ Normal()
    y ~ Normal()
end

# Plain-loop models, plus hand-written equivalents of rewriting the
# looped site family.
@wren function looped(N)
    y = zeros(N)
    x ~ Normal()
    for i in 1:N
        y[i] ~ Normal(x)
    end
end
@wren function looped_exp(N)
    y = zeros(N)
    x ~ Normal()
    for i in 1:N
        y[i] ~ Exponential(2.0)
    end
end
@wren function looped_idx(N)
    y = zeros(N)
    x ~ Normal()
    for i in 1:N
        y[i] ~ Normal(x, i)
    end
end

# Hand-written targets for family collapse, fixing, and construct
# replacement.
@wren function family_vec(N)
    x ~ Normal()
    y ~ product_distribution(fill(Exponential(2.0), N))
end
@wren function family_fixed_hand(v)
    x ~ Normal()
    y = v
end
@wren function vec_site(N)
    mu ~ Normal()
    y ~ MvNormal(zeros(N), I)
end
@wren function plated_std(N)
    mu ~ Normal()
    y = zeros(N)
    @plate for i in 1:N
        y[i] ~ Normal()
    end
end
@wren function looped_std(N)
    mu ~ Normal()
    y = zeros(N)
    for i in 1:N
        y[i] ~ Normal()
    end
end
@wren function looped_pair(N)
    y = zeros(N)
    z = zeros(N)
    for i in 1:N
        y[i] ~ Normal()
        z[i] ~ Normal(y[i])
    end
end
@wren function looped_bad(N)
    y = zeros(N)
    w = zeros(N)
    for i in 1:N
        w[i] = y[i] + 1.0
        y[i] ~ Normal()
    end
end
@wren function looped_bad_toplevel(N)
    y = zeros(N)
    s = sum(y)
    for i in 1:N
        y[i] ~ Normal(s)
    end
end
@wren function plated_pair(N)
    y = zeros(N)
    z = zeros(N)
    @plate for i in 1:N
        y[i] ~ Normal()
        z[i] ~ Normal(y[i])
    end
end

# Statement display for comparing model IRs. Plate names are gensym'd
# per model definition, so they are normalised away.
function _stmt_str(model)
    str = sprint(show, MIME("text/plain"), model.stmts)
    return replace(str, r"##plate#\d+" => "##plate#")
end

# Assert that two models are observably identical: same sites drawn,
# same log density components at the same params, same vectorised
# dimension, and same gradients.
function _components(m, params)
    _, accs = evaluate(m, FromParams(params), OriginalSpace(), Wren.default_accumulators())
    return accs
end

function assert_equivalent(m1, m2, params)
    @test Set(keys(rand(m1))) == Set(keys(rand(m2)))
    accs1 = _components(m1, params)
    accs2 = _components(m2, params)
    @test logprior(accs1) ≈ logprior(accs2)
    @test loglikelihood(accs1) ≈ loglikelihood(accs2)
    @test logjoint(accs1) ≈ logjoint(accs2)
    vm1 = VecModel(m1, AutoForwardDiff())
    vm2 = VecModel(m2, AutoForwardDiff())
    v = rand(vm1)
    @test length(v) == length(rand(vm2))
    lp1, grad1 = lp_and_grad(vm1, v)
    lp2, grad2 = lp_and_grad(vm2, v)
    @test lp1 ≈ lp2
    @test grad1 ≈ grad2
    return nothing
end

@testset "rewrite" begin
    @testset "macro form" begin
        @testset "distribution, IR-identical to hand-written" begin
            m = @rewrite rw_base() begin
                x ~ LogNormal()
            end
            @test _stmt_str(m) == _stmt_str(hand_lognormal())
            assert_equivalent(m, hand_lognormal(), @vnt begin
                x := 0.5
                y := -0.3
            end)
            # the original model is untouched
            @test rw_base()[@vn(x)] == Normal()
            @test m[@vn(x)] == LogNormal()
        end

        @testset "submodel, IR-identical to hand-written" begin
            m = @rewrite rw_base() begin
                x ~ rw_inner()
            end
            @test _stmt_str(m) == _stmt_str(hand_submodel())
            assert_equivalent(m, hand_submodel(), @vnt begin
                x.a := 0.2
                x.b := -0.1
                y := 0.4
            end)
        end

        @testset "fixed value, IR-identical to hand-written" begin
            m = @rewrite rw_base() begin
                x = 1.5
            end
            @test _stmt_str(m) == _stmt_str(hand_fixed())
            @test Set(keys(rand(m))) == Set([@vn(y)])
            assert_equivalent(m, hand_fixed(), VarNamedTuple(; y=0.7))
        end

        @testset "multiple statements and back again" begin
            m = @rewrite rw_base() begin
                x ~ rw_inner()
                y = 2.0
            end
            @test Set(keys(rand(m))) == Set([@vn(x.a), @vn(x.b)])
            m2 = @rewrite m begin
                x ~ Normal()
                y ~ Normal()
            end
            @test _stmt_str(m2) == _stmt_str(rw_base())
            @test Set(keys(rand(m2))) == Set([@vn(x), @vn(y)])
        end

        @testset "spliced expressions see model arguments" begin
            @wren function with_arg(a)
                x ~ Normal()
                y ~ Normal(x)
            end
            m = @rewrite with_arg(3.0) begin
                x = 2 * a
            end
            pars = rand(m)
            @test !(@vn(x) in keys(pars))
            # downstream code sees the fixed value: y ~ Normal(6.0)
            m_cond = m | VarNamedTuple(; y=6.0)
            accs = rand_with_logp(m_cond)
            @test loglikelihood(accs) ≈ logdensityof(Normal(6.0), 6.0)
        end

        @testset "@observe patch" begin
            @wren function latent_y(y)
                x ~ Normal()
                y ~ Normal(x)
            end
            m = @rewrite latent_y(0.3) begin
                @observe y ~ Normal(x)
            end
            accs = rand_with_logp(m)
            pars = parameters(accs)
            @test only(keys(pars)) == @vn(x)
            @test loglikelihood(accs) ≈ logdensityof(Normal(pars[@vn(x)]), 0.3)
        end

        @testset "evaluation function is cached per call site" begin
            resub(m) = @rewrite m begin
                x ~ LogNormal()
            end
            m1 = resub(rw_base())
            m2 = resub(rw_base())
            @test m1.evaluation_fn === m2.evaluation_fn
        end

        @testset "macro over function form drops the hidden argument" begin
            s = rewrite(rw_base(), @vn(x) => LogNormal())
            @test haskey(s.args, Symbol("#rewrite#x"))
            m = @rewrite s begin
                x ~ Normal()
            end
            @test !haskey(m.args, Symbol("#rewrite#x"))
            @test _stmt_str(m) == _stmt_str(rw_base())
            @test Set(keys(rand(m))) == Set([@vn(x), @vn(y)])
        end

        @testset "errors" begin
            @test_throws LoadError macroexpand(@__MODULE__, :(@rewrite m begin
                foo(1)
            end))
            m = rw_base()
            err = try
                @rewrite m begin
                    z ~ Normal()
                end
                nothing
            catch e
                e
            end
            @test err isa Exception
            @test occursin("no site named `z`", sprint(showerror, err))
            @test occursin("x, y", sprint(showerror, err))
        end
    end

    @testset "function form" begin
        @testset "distribution -> distribution" begin
            model = rw_base()
            s = rewrite(model, @vn(x) => LogNormal())
            assert_equivalent(s, hand_lognormal(), @vnt begin
                x := 0.5
                y := -0.3
            end)
            @test s[@vn(x)] == LogNormal()
            @test model[@vn(x)] == Normal()
        end

        @testset "distribution -> submodel and back" begin
            model = rw_base()
            s = rewrite(model, @vn(x) => rw_inner())
            assert_equivalent(s, hand_submodel(), @vnt begin
                x.a := 0.2
                x.b := -0.1
                y := 0.4
            end)
            back = rewrite(s, @vn(x) => Normal())
            @test Set(keys(rand(back))) == Set([@vn(x), @vn(y)])
        end

        @testset "fixed values" begin
            model = rw_base()
            s = rewrite(model, @vn(x) => 1.5)
            @test s[@vn(x)] == 1.5
            assert_equivalent(s, hand_fixed(), VarNamedTuple(; y=0.7))
            # no density contribution from the fixed site
            accs = rand_with_logp(s)
            y = parameters(accs)[@vn(y)]
            @test logprior(accs) ≈ logdensityof(Normal(), y)
            # VecModel dimension shrinks from 2 to 1
            @test length(rand(VecModel(s, AutoForwardDiff()))) == 1
            @test length(rand(VecModel(model, AutoForwardDiff()))) == 2
        end

        @testset "downstream code sees the fixed value" begin
            @wren function chained()
                x ~ Normal()
                y ~ Normal(x)
            end
            s = rewrite(chained(), @vn(x) => 1.5)
            cond = s | VarNamedTuple(; y=2.0)
            accs = rand_with_logp(cond)
            @test isempty(keys(parameters(accs)))
            @test loglikelihood(accs) ≈ logdensityof(Normal(1.5), 2.0)
        end

        @testset "multiple pairs" begin
            s = rewrite(rw_base(), @vn(x) => LogNormal(), @vn(y) => 0.5)
            @test Set(keys(rand(s))) == Set([@vn(x)])
            @test s[@vn(y)] == 0.5
        end

        @testset "getindex sugar" begin
            m = rw_base()
            s = m[@vn(x)=>LogNormal()]
            @test s[@vn(x)] == LogNormal()
            s2 = s[@vn(x)=>2.5]
            @test s2[@vn(x)] == 2.5
        end

        @testset "repeated rewrites reuse the evaluation function" begin
            model = rw_base()
            s1 = rewrite(model, @vn(x) => LogNormal())
            s2 = rewrite(model, @vn(x) => LogNormal())
            @test s1.evaluation_fn === s2.evaluation_fn
            # rewriting an already-rewritten site keeps the same fn
            s3 = rewrite(s1, @vn(x) => LogNormal(1.0))
            @test s3.evaluation_fn === s1.evaluation_fn
            # but changing the form (sample -> fixed) rebuilds it
            s4 = rewrite(s1, @vn(x) => 1.0)
            @test s4.evaluation_fn !== s1.evaluation_fn
            @test Set(keys(rand(s4))) == Set([@vn(y)])
        end

        @testset "getindex errors for body-dependent rhs" begin
            @wren function dependent()
                x ~ Normal()
                y ~ Normal(x)
            end
            @test_throws ErrorException dependent()[@vn(y)]
        end

        @testset "errors" begin
            model = rw_base()
            @test_throws ErrorException rewrite(model, @vn(z) => Normal())
            @test_throws ErrorException rewrite(model, @vn(x[1]) => Normal())
        end

        @testset "show is as-written" begin
            # The hidden argument is folded into the displayed
            # statements, so the model shows exactly as if it had been
            # written with the new right-hand side.
            m = rewrite(rw_base(), @vn(x) => LogNormal())
            str = sprint(show, MIME("text/plain"), m)
            @test occursin(r"x ~ (Distributions\.)?LogNormal", str)
            @test !occursin("#rewrite#", str)
            @test occursin("arguments (none)", str)
            f = rewrite(rw_base(), @vn(x) => 1.5)
            fstr = sprint(show, MIME("text/plain"), f)
            @test occursin("x = 1.5", fstr)
            @test !occursin("#rewrite#", fstr)
            s = rewrite(rw_base(), @vn(x) => rw_inner())
            sstr = sprint(show, MIME("text/plain"), s)
            @test occursin("x ~ Model(rw_inner)", sstr)
            @test !occursin("#rewrite#", sstr)
            # the compact form names the conditioned variables
            c = rewrite(rw_base(), @vn(x) => rw_inner() | VarNamedTuple(; b=1.0))
            @test occursin("x ~ Model(rw_inner | b)", sprint(show, MIME("text/plain"), c))
        end
    end

    @testset "plate sites" begin
        m = rewrite(plated(3), @vn(x) => LogNormal())
        @test Set(keys(rand(m))) == Set([@vn(x), @vn(y[1]), @vn(y[2]), @vn(y[3])])

        @testset "collapse to a single statement, function form" begin
            pm = rewrite(
                plated(3),
                @vn(y) => product_distribution(fill(Exponential(2.0), 3)),
            )
            assert_equivalent(pm, family_vec(3), @vnt begin
                x := 0.3
                y := [0.5, 1.0, 1.5]
            end)
            @test Set(keys(rand(pm))) == Set([@vn(x), @vn(y)])
            str = sprint(show, MIME("text/plain"), pm)
            @test !occursin("#rewrite#", str)
            @test !occursin("@plate", str)
        end

        @testset "collapse to a single statement, macro form" begin
            mm = @rewrite plated(3) begin
                y ~ product_distribution(fill(Exponential(2.0), N))
            end
            @test _stmt_str(mm) == _stmt_str(family_vec(3))
            assert_equivalent(mm, family_vec(3), @vnt begin
                x := 0.3
                y := [0.5, 1.0, 1.5]
            end)
        end

        @testset "elementwise rhs stays inside the plate" begin
            mm = @rewrite plated(3) begin
                y ~ Normal(x, i)
            end
            @test _stmt_str(mm) == _stmt_str(plated_idx(3))
            assert_equivalent(mm, plated_idx(3), @vnt begin
                x := 0.3
                y := [0.5, 1.0, 1.5]
            end)
        end

        @testset "fixing a family to a value" begin
            fm = @rewrite plated(3) begin
                y = [0.5, 1.0, 1.5]
            end
            for ff in (rewrite(plated(3), @vn(y) => [0.5, 1.0, 1.5]), fm)
                assert_equivalent(
                    ff,
                    family_fixed_hand([0.5, 1.0, 1.5]),
                    VarNamedTuple(; x=0.3),
                )
                @test Set(keys(rand(ff))) == Set([@vn(x)])
            end
            ff = rewrite(plated(3), @vn(y) => [0.5, 1.0, 1.5])
            @test occursin("y = [0.5, 1.0, 1.5]", sprint(show, MIME("text/plain"), ff))
            # un-fixing gives a plain vector-valued site
            back = rewrite(ff, @vn(y) => product_distribution(fill(Exponential(2.0), 3)))
            @test Set(keys(rand(back))) == Set([@vn(x), @vn(y)])
            @test length(rand(back)[@vn(y)]) == 3
        end

        @testset "replacing a statement with a construct and back" begin
            elem = @rewrite vec_site(3) begin
                y = zeros(N)
                @plate for i in 1:N
                    y[i] ~ Normal()
                end
            end
            @test _stmt_str(elem) == _stmt_str(plated_std(3))
            assert_equivalent(elem, plated_std(3), @vnt begin
                mu := 0.1
                y := [0.4, -0.2, 0.9]
            end)
            back = @rewrite elem begin
                y ~ MvNormal(zeros(N), I)
            end
            @test _stmt_str(back) == _stmt_str(vec_site(3))
            assert_equivalent(back, vec_site(3), @vnt begin
                mu := 0.1
                y := [0.4, -0.2, 0.9]
            end)
        end

        @testset "swapping a plate for a loop" begin
            lp = @rewrite plated(3) begin
                y = zeros(N)
                for i in 1:N
                    y[i] ~ Normal(x)
                end
            end
            assert_equivalent(lp, looped(3), @vnt begin
                x := 0.3
                y := [0.5, 1.0, 1.5]
            end)
        end

        @testset "element-granular targets error informatively" begin
            err = try
                rewrite(plated(3), @vn(y[1]) => Normal())
                nothing
            catch e
                e
            end
            @test err isa Exception
            @test occursin("issues/19", sprint(showerror, err))
        end
    end

    @testset "looped sites" begin
        @testset "collapse to a single statement, function form" begin
            # The allocation-hijack repro: targeting the loop family
            # yields the as-written single statement, and the
            # allocation is never turned into a tilde statement.
            lm = rewrite(looped(3), @vn(y) => Exponential(2.0))
            @test Set(keys(rand(lm))) == Set([@vn(x), @vn(y)])
            @test rand(lm)[@vn(y)] > 0
            @test !occursin("y = zeros(N)", _stmt_str(lm))
            lv = rewrite(
                looped(3),
                @vn(y) => product_distribution(fill(Exponential(2.0), 3)),
            )
            assert_equivalent(lv, family_vec(3), @vnt begin
                x := 0.3
                y := [0.5, 1.0, 1.5]
            end)
        end

        @testset "elementwise rhs stays inside the loop" begin
            lm = @rewrite looped(3) begin
                y ~ Normal(x, i)
            end
            @test _stmt_str(lm) == _stmt_str(looped_idx(3))
            assert_equivalent(lm, looped_idx(3), @vnt begin
                x := 0.3
                y := [0.5, 1.0, 1.5]
            end)
        end

        @testset "fixing a family to a value" begin
            fm = @rewrite looped(3) begin
                y = [0.5, 1.0, 1.5]
            end
            for ff in (rewrite(looped(3), @vn(y) => [0.5, 1.0, 1.5]), fm)
                assert_equivalent(
                    ff,
                    family_fixed_hand([0.5, 1.0, 1.5]),
                    VarNamedTuple(; x=0.3),
                )
                @test Set(keys(rand(ff))) == Set([@vn(x)])
            end
        end

        @testset "replacing a statement with a loop and back" begin
            elem = @rewrite vec_site(3) begin
                y = zeros(N)
                for i in 1:N
                    y[i] ~ Normal()
                end
            end
            @test _stmt_str(elem) == _stmt_str(looped_std(3))
            assert_equivalent(elem, looped_std(3), @vnt begin
                mu := 0.1
                y := [0.4, -0.2, 0.9]
            end)
            back = @rewrite elem begin
                y ~ MvNormal(zeros(N), I)
            end
            @test _stmt_str(back) == _stmt_str(vec_site(3))
            assert_equivalent(back, vec_site(3), @vnt begin
                mu := 0.1
                y := [0.4, -0.2, 0.9]
            end)
        end

        @testset "swapping a loop for a plate" begin
            pl = @rewrite looped(3) begin
                y = zeros(N)
                @plate for i in 1:N
                    y[i] ~ Normal(x)
                end
            end
            assert_equivalent(pl, plated(3), @vnt begin
                x := 0.3
                y := [0.5, 1.0, 1.5]
            end)
        end

        @testset "multi-statement bodies" begin
            f = rewrite(looped_pair(3), @vn(y) => [1.0, 2.0, 3.0])
            # z stays looped and reads the fixed values
            zvals = [1.1, 2.2, 2.9]
            accs = _components(f, @vnt begin
                z := zvals
            end)
            @test logprior(accs) ≈
                  sum(logdensityof(Normal(v), z) for (v, z) in zip([1.0, 2.0, 3.0], zvals))
            c = rewrite(
                looped_pair(3),
                @vn(y) => product_distribution(fill(Exponential(2.0), 3)),
            )
            @test Set(keys(rand(c))) == Set([@vn(y), @vn(z[1]), @vn(z[2]), @vn(z[3])])
            # a plated multi-statement body keeps its other family
            pc = rewrite(
                plated_pair(3),
                @vn(y) => product_distribution(fill(Exponential(2.0), 3)),
            )
            @test Set(keys(rand(pc))) == Set([@vn(y), @vn(z[1]), @vn(z[2]), @vn(z[3])])
            # statements reading the family before it is defined are
            # rejected, both inside the loop body and before the construct
            @test_throws ErrorException rewrite(looped_bad(3), @vn(y) => [1.0, 2.0, 3.0])
            @test_throws ErrorException rewrite(
                looped_bad_toplevel(3),
                @vn(y) => [1.0, 2.0, 3.0],
            )
        end

        @testset "element-granular targets error" begin
            err = try
                rewrite(looped(3), @vn(y[1]) => Normal())
                nothing
            catch e
                e
            end
            @test err isa Exception
            @test occursin("issues/19", sprint(showerror, err))
        end

        @testset "observed families stay observed" begin
            @wren function loop_obs(data)
                y = zeros(2)
                x ~ Normal()
                for i in 1:2
                    @observe data[i] ~ Normal(x)
                end
            end
            err = try
                rewrite(loop_obs([0.1, 0.2]), @vn(data) => Normal())
                nothing
            catch e
                e
            end
            @test err isa Exception
            @test occursin("observed", sprint(showerror, err))
            # an elementwise `~` replacement keeps the observed flavour
            m2 = @rewrite loop_obs([0.1, 0.2]) begin
                data ~ Normal(x, 1.0 * i)
            end
            accs = _components(m2, VarNamedTuple(; x=0.3))
            @test loglikelihood(accs) ≈
                  logdensityof(Normal(0.3, 1.0), 0.1) + logdensityof(Normal(0.3, 2.0), 0.2)
        end

        @testset "construct patch error paths" begin
            # a construct cannot replace an observed statement
            @wren function obs_site(obs)
                x ~ Normal()
                @observe obs ~ Normal(x)
            end
            err = try
                m = obs_site(0.5)
                @rewrite m begin
                    obs = zeros(2)
                    for i in 1:2
                        obs[i] ~ Normal(x)
                    end
                end
                nothing
            catch e
                e
            end
            @test err isa Exception
            @test occursin("observed statement", sprint(showerror, err))
            # nor a site with more than one statement
            @wren function twice()
                y ~ Normal()
                y ~ Exponential(1.0)
            end
            err2 = try
                m = twice()
                @rewrite m begin
                    y = zeros(2)
                    for i in 1:2
                        y[i] ~ Normal()
                    end
                end
                nothing
            catch e
                e
            end
            @test err2 isa Exception
            @test occursin("more than one statement", sprint(showerror, err2))
            # nor statements nested inside another statement
            @wren function branchy(c)
                if c
                    y ~ Normal()
                else
                    y ~ Exponential(1.0)
                end
            end
            err4 = try
                m = branchy(true)
                @rewrite m begin
                    y = zeros(2)
                    for i in 1:2
                        y[i] ~ Normal()
                    end
                end
                nothing
            catch e
                e
            end
            @test err4 isa Exception
            @test occursin("nested inside", sprint(showerror, err4))
            # nor a site that does not exist
            err3 = try
                m = rw_base()
                @rewrite m begin
                    z = zeros(2)
                    for i in 1:2
                        z[i] ~ Normal()
                    end
                end
                nothing
            catch e
                e
            end
            @test err3 isa Exception
            @test occursin("no site named `z`", sprint(showerror, err3))
        end
    end

    @testset "plate and loop wrappers" begin
        @testset "remove_plate" begin
            m = remove_plate(plated(3))
            @test _stmt_str(m) == _stmt_str(looped(3))
            assert_equivalent(m, looped(3), @vnt begin
                x := 0.3
                y := [0.5, 1.0, 1.5]
            end)
            @test_throws ErrorException remove_plate(looped(3))
            @test_throws ErrorException remove_plate(plated(3), :nope)
            @wren function named_plate(N)
                y = zeros(N)
                @plate name = :obs for i in 1:N
                    y[i] ~ Normal()
                end
            end
            m2 = remove_plate(named_plate(2), :obs)
            @test Set(keys(rand(m2))) == Set([@vn(y[1]), @vn(y[2])])
        end

        @testset "add_plate" begin
            m = add_plate(looped(3), @vn(y))
            @test _stmt_str(m) == _stmt_str(plated(3))
            assert_equivalent(m, plated(3), @vnt begin
                x := 0.3
                y := [0.5, 1.0, 1.5]
            end)
            @test_throws ErrorException add_plate(plated(3), @vn(y))
            @test_throws ErrorException add_plate(looped_bad(3), @vn(y))
        end

        @testset "set_arg" begin
            m = plated(3)
            m5 = set_arg(m, :N, 5)
            @test m5.evaluation_fn === m.evaluation_fn
            @test Set(keys(rand(m5))) == Set([@vn(x); [@vn(y[i]) for i in 1:5]])
            @test_throws ErrorException set_arg(m, :M, 5)
        end

        @testset "element reuse across a resized model" begin
            # The route sketched in issue #19: drop the plate, extend N,
            # and evaluate reusing stored element values with a prior
            # fallback for the new tail.
            short = remove_plate(plated(3))
            fitted = rand(short)
            long = set_arg(short, :N, 5)
            _, accs = evaluate(
                long,
                FromParams(fitted, FromPrior(Xoshiro(99))),
                OriginalSpace(),
                Wren.default_accumulators(),
            )
            pars = parameters(accs)
            @test Set(keys(pars)) == Set([@vn(x); [@vn(y[i]) for i in 1:5]])
            for i in 1:3
                @test pars[@vn(y[i])] == fitted[@vn(y[i])]
            end
            @test pars[@vn(x)] == fitted[@vn(x)]
        end
    end

    @testset "indexed sites" begin
        m = indexed_model(rw_inner())
        @test Set(keys(rand(m))) == Set([@vn(x[1].a), @vn(x[1].b), @vn(x[2]), @vn(y)])

        @testset "function form" begin
            s = rewrite(m, @vn(x[1]) => Normal())
            @test Set(keys(rand(s))) == Set([@vn(x[1]), @vn(x[2]), @vn(y)])
            @test s[@vn(x[1])] == Normal()
            # and back to a submodel again
            s2 = rewrite(s, @vn(x[1]) => rw_inner())
            @test Set(keys(rand(s2))) == Set([@vn(x[1].a), @vn(x[1].b), @vn(x[2]), @vn(y)])
        end

        @testset "fixing an indexed site" begin
            f = rewrite(m, @vn(x[2]) => 0.5)
            @test Set(keys(rand(f))) == Set([@vn(x[1].a), @vn(x[1].b), @vn(y)])
            @test f[@vn(x[2])] == 0.5
            fstr = sprint(show, MIME("text/plain"), f)
            @test occursin("x[2] = 0.5", fstr)
        end

        @testset "macro form" begin
            mm = @rewrite indexed_model(rw_inner()) begin
                x[2] ~ LogNormal()
            end
            draw = rand(mm)
            @test draw[@vn(x[2])] > 0
        end
    end

    @testset "assignment and expression rewrites" begin
        @testset "deterministic to deterministic, both forms" begin
            @wren function det_model()
                c = 1.0
                y ~ Normal(c)
            end
            # No tilde statements named `c` exist, so the assignment is
            # a legitimate target (the family guard must not block it).
            mm = @rewrite det_model() begin
                c = 2.5
            end
            fm = rewrite(det_model(), @vn(c) => 2.5)
            for m in (mm, fm)
                cond = m | VarNamedTuple(; y=3.0)
                accs = rand_with_logp(cond)
                @test loglikelihood(accs) ≈ logdensityof(Normal(2.5), 3.0)
            end
            @test occursin("c = 2.5", _stmt_str(mm))
        end

        @testset "multi-line and block right-hand sides" begin
            @wren function mapper(xs)
                w = xs
                y ~ Normal(sum(w))
            end
            m = @rewrite mapper([1.0, 2.0]) begin
                w = map(x -> x^2, xs)
            end
            cond = m | VarNamedTuple(; y=5.0)
            accs = rand_with_logp(cond)
            @test loglikelihood(accs) ≈ logdensityof(Normal(5.0), 5.0)

            b = @rewrite rw_base() begin
                x ~ begin
                    d = Normal(0.0, 2.0)
                    d
                end
            end
            params = VarNamedTuple(; x=0.4, y=0.1)
            @test logjoint(b, params) ≈
                  logdensityof(Normal(0.0, 2.0), 0.4) + logdensityof(Normal(), 0.1)
        end

        @testset "complex original right-hand sides" begin
            @wren function complex_rhs(a)
                x ~ Normal(exp(a), sqrt(2.0))
                y ~ Normal()
            end
            m = rewrite(complex_rhs(0.3), @vn(x) => LogNormal())
            draw = rand(m)
            @test draw[@vn(x)] > 0
            @test logjoint(m, draw) ≈
                  logdensityof(LogNormal(), draw[@vn(x)]) +
                  logdensityof(Normal(), draw[@vn(y)])
        end
    end

    @testset "conditioning push-down" begin
        @testset "prefixed paths reach into submodels" begin
            for m in (hand_submodel(), rewrite(rw_base(), @vn(x) => rw_inner()))
                cond = m | @vnt begin
                    x.b := 1.5
                end
                accs = rand_with_logp(cond)
                pars = parameters(accs)
                @test !(@vn(x.b) in keys(pars))
                a = pars[@vn(x.a)]
                @test loglikelihood(accs) ≈ logdensityof(Normal(a), 1.5)
                # the pushed-down values are visible on the inner model
                @test conditioned(cond[@vn(x)])[@vn(b)] == 1.5
            end
        end

        @testset "merging and mixed conditioning" begin
            m = hand_submodel()
            cond = m | @vnt begin
                x.b := 1.5
                y := 0.5
            end
            accs = rand_with_logp(cond)
            @test only(keys(parameters(accs))) == @vn(x.a)
            cond2 = cond | @vnt begin
                x.a := 0.25
            end
            accs2 = rand_with_logp(cond2)
            @test isempty(keys(parameters(accs2)))
            @test loglikelihood(accs2) ≈
                  logdensityof(Normal(), 0.25) +
                  logdensityof(Normal(0.25), 1.5) +
                  logdensityof(Normal(), 0.5)
        end

        @testset "raw value onto a submodel site errors" begin
            m = hand_submodel()
            @test_throws ErrorException m | VarNamedTuple(; x=1.0)
        end

        @testset "two-level nesting" begin
            cond = nested_top() | @vnt begin
                m.inner.b := 0.5
            end
            accs = rand_with_logp(cond)
            pars = parameters(accs)
            @test !(@vn(m.inner.b) in keys(pars))
            a = pars[@vn(m.inner.a)]
            @test loglikelihood(accs) ≈ logdensityof(Normal(a), 0.5)
        end
    end

    @testset "decondition" begin
        @testset "top-level values" begin
            m = rw_base() | VarNamedTuple(; y=0.5)
            @test Set(keys(rand(m))) == Set([@vn(x)])
            d = decondition(m, @vn(y))
            @test Set(keys(rand(d))) == Set([@vn(x), @vn(y)])
        end

        @testset "reaches through two levels of push-down" begin
            m = nested_top() | @vnt begin
                m.inner.b := 0.5
            end
            @test Set(keys(rand(m))) == Set([@vn(m.inner.a), @vn(m.c), @vn(d)])
            d2 = decondition(m, @vn(m))
            @test Set(keys(rand(d2))) ==
                  Set([@vn(m.inner.a), @vn(m.inner.b), @vn(m.c), @vn(d)])
        end

        @testset "reaches into pushed-down submodels" begin
            m = hand_submodel() | @vnt begin
                x.b := 1.5
                y := 0.5
            end
            d = decondition(m, @vn(x.b))
            @test Set(keys(rand(d))) == Set([@vn(x.a), @vn(x.b)])
            # deconditioning the whole site removes everything inside
            d2 = decondition(m, @vn(x), @vn(y))
            @test Set(keys(rand(d2))) == Set([@vn(x.a), @vn(x.b), @vn(y)])
        end
    end

    @testset "Enzyme AD" begin
        adtype = AutoEnzyme(; mode=Reverse)

        @testset "test_ad on rewritten models" begin
            two_level = @rewrite nested_top() begin
                d ~ LogNormal()
            end
            rewritten = [
                "to distribution" => rewrite(rw_base(), @vn(x) => LogNormal()),
                "to submodel" => rewrite(rw_base(), @vn(x) => rw_inner()),
                "fixed value" => rewrite(rw_base(), @vn(x) => 1.5),
                "pushed-down conditioning" =>
                    rewrite(rw_base(), @vn(x) => rw_inner()) | @vnt(begin
                        x.b := 1.5
                    end),
                "two-level nesting" => two_level,
                "collapsed plated family" =>
                    rewrite(plated(3), @vn(y) => Exponential(2.0)),
                "collapsed looped family" =>
                    rewrite(looped(3), @vn(y) => Exponential(2.0)),
                "fixed family" => rewrite(plated(3), @vn(y) => [0.5, 1.0, 1.5]),
                "elementwise via construct" => @rewrite(vec_site(3), begin
                    y = zeros(N)
                    @plate for i in 1:N
                        y[i] ~ Normal()
                    end
                end),
                "plate removed" => remove_plate(plated(3)),
                "plate added" => add_plate(looped(3), @vn(y)),
            ]
            for (i, (label, m)) in enumerate(rewritten)
                @testset "$label" begin
                    result = test_ad(
                        m,
                        adtype;
                        latent_source=FromPrior(Xoshiro(500 + i)),
                        verbose=false,
                    )
                    @test result isa ModelRunResult
                    @test result.lp_actual ≈ result.lp_expected
                    @test result.grad_actual ≈ result.grad_expected
                end
            end
        end

        @testset "rewritten models match hand-written ones" begin
            # test_ad checks that the gradient is correct; this checks
            # the as-if-written rule, i.e. that the rewritten model has
            # exactly the hand-written model's lp and gradient.
            function test_same_lp_and_grad(m1, m2)
                vm1 = VecModel(m1, adtype)
                v = rand(vm1)
                lp1, grad1 = lp_and_grad(vm1, v)
                lp2, grad2 = lp_and_grad(VecModel(m2, adtype), v)
                @test lp1 ≈ lp2
                @test grad1 ≈ grad2
                return nothing
            end
            test_same_lp_and_grad(rewrite(rw_base(), @vn(x) => rw_inner()), hand_submodel())
            prod3 = product_distribution(fill(Exponential(2.0), 3))
            test_same_lp_and_grad(rewrite(plated(3), @vn(y) => prod3), family_vec(3))
            test_same_lp_and_grad(rewrite(looped(3), @vn(y) => prod3), family_vec(3))
            elem = @rewrite vec_site(3) begin
                y = zeros(N)
                @plate for i in 1:N
                    y[i] ~ Normal()
                end
            end
            test_same_lp_and_grad(elem, plated_std(3))
        end
    end
end

end # module WrenRewriteTests
