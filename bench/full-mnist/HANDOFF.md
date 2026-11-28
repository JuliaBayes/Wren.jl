# Handoff: full MNIST on GPU

Status as of 2026-11-28.

## What this is

`gpu.jl` times six versions of the MNIST multinomial logistic regression model on the full
MNIST training set (60,000 images × 784 raw pixels):

 1. DPPL, as written in `bench/adhoc/basic/models.jl` (y passed as an argument)
 2. DPPL, with y created inside the model and supplied by conditioning
 3. DPPL, with the observations replaced by a vectorised likelihood in `@addlogprob!`
 4. Wren, plain `for` loop over observations
 5. Wren, `@batch` loop over observations
 6. Wren, with the observations replaced by the same vectorised likelihood in
    `@addloglikelihood`

All six run with Enzyme reverse mode on the CPU. Variants 5 and 6 also run with Reactant.
See the comments at the top of `gpu.jl` for how to run it and how to switch between
Float32 and Float64.

## Results so far

Laptop: Apple M5 Pro, 15 threads. GPU: 1× A100 SXM4 40 GB, 30 vCPUs. All times are
lp+grad (lp alone in brackets). The laptop Float32 and Reactant-on-CPU numbers used 1 s
timings, so they are rough.

| variant                                 | laptop Float64 | laptop Float32 | A100 Float64        |
|:--------------------------------------- | --------------:| --------------:| -------------------:|
| 1. DPPL (y as arg)                      | 41.9 ms (12.4) | 32.6 ms (8.7)  |                     |
| 2. DPPL (conditioned y)                 | 41.5 ms (12.4) | 31.7 ms (8.7)  |                     |
| 3. DPPL (`@addlogprob!`)                | 29.3 ms (10.3) | 25.3 ms (10.1) |                     |
| 4. Wren (no `@batch`)                   | 35.9 ms (10.8) | 24.1 ms (7.3)  |                     |
| 5. Wren (`@batch`)                      | 30.6 ms (10.1) | 21.5 ms (6.4)  |                     |
| 6. Wren (`@addloglikelihood`)           | 28.8 ms (10.4) | 24.4 ms (10.0) |                     |
| 5. Wren (`@batch`), Reactant            | 13.8 ms (4.8)  | 8.6 ms (2.4)   | **170.9 ms (47.8)** |
| 6. Wren (`@addloglikelihood`), Reactant | 9.0 ms (4.4)   | 5.8 ms (2.3)   | **171.4 ms (47.9)** |

On the laptop, "Reactant" means Reactant on the CPU. The A100 run used
`WREN_BENCH_CPU=false`, so it has no CPU numbers. The Float32 run on the A100 has not been
done yet.

The good news: the Wren model compiles for the GPU without changes and gives the same
results as the CPU (the script checks this).

The bad news: on the A100 it is about 12× slower than Reactant on the laptop CPU, and
slower than plain Enzyme on the CPU. Compiling was also very slow: more than about
7 minutes of wall-clock time for the first model, with the GPU idle and the CPU busy.

## What we know about the slowness

  - Variants 5 and 6 take the same time to the millisecond. So the likelihood and `@batch`
    are not the problem. The time goes into the part they share: `W * X' .+ b`, the
    `vcat`, `softmax`/`logsumexp`, or moving data around.
  - Even `lp` alone is slow (48 ms), so it is not only the gradient.
  - The matrix multiply is under 1 GFLOP. At 48 ms that would be about 18 GFLOPS, when an
    A100 should manage several TFLOPS even in Float64. So probably a very poor kernel
    choice somewhere.
  - `nvidia-smi` showed 30 GB used. That is normal: XLA reserves 75% of GPU memory up
    front.

## Suspects

 1. **The transpose in `W * X'`.** Reactant issue
    [#373](https://github.com/EnzymeAD/Reactant.jl/issues/373) (open) says that `mul!`
    with wrapped arrays like `Adjoint` makes an actual transposed copy first
    (`materialize_traced_array`) and then calls `dot_general`. A copy alone should only
    cost a fraction of a millisecond on an A100, but XLA may then choose a bad kernel for
    the layout. Fix to try: pass `Xt = permutedims(X)` (784 × 60,000) and write `W * Xt`.
 2. **`X` being traced.** WrenCore's Reactant extension calls `Reactant.to_rarray` on all
    model arguments. The source cites
    [#3167](https://github.com/EnzymeAD/Reactant.jl/issues/3167) (now closed) and
    [#3195](https://github.com/EnzymeAD/Reactant.jl/issues/3195) (still open) as the
    reason. A traced `X` is a program input, so XLA must take its layout as given. On the
    other hand, an untraced `X` would mean building a 376 MB constant into the program,
    which may make compiling worse. Since #3167 is closed, it may be worth checking
    whether the workaround is still needed.
 3. **The argument copy on every call.** The Reactant `VecModel` uses
    `EvalContext{true}()`, which copies the model arguments on each call. The comment
    there says it is unclear whether Enzyme-MLIR needs this. XLA probably removes the
    copy, but that hasn't been checked.
 4. **Autotuning making the compile slow.** On a GPU, XLA tries several kernels and times
    them. Try `XLA_FLAGS=--xla_gpu_autotune_level=0` to see whether compiling gets faster,
    and check whether the run times change. (It's not certain that Reactant passes
    `XLA_FLAGS` through.)

Related issues that are probably not the cause:
[#3381](https://github.com/EnzymeAD/Reactant.jl/issues/3381) and
[#3583](https://github.com/EnzymeAD/Reactant.jl/issues/3583) (`Adjoint` bugs that cause
errors, not slowness), and [#3511](https://github.com/EnzymeAD/Reactant.jl/issues/3511)
(slow first call caused by thousands of buffers; this model has only a few).

## Next steps

 1. Write a short diagnostic script that times each piece under Reactant on the GPU:

      + `W * X'` vs `W * Xt`, with `X` traced vs untraced
      + then adding `.+ b`, the `vcat`, and `softmax`/`logsumexp` one at a time
      + the full model with and without the argument copy
      + each in Float64 and Float32

    This should find the slow step in a few minutes of GPU time, rather than another long
    benchmark run.

 2. Record a GPU profile with Reactant's profiler to see which kernels take the time.

 3. Once the cause is fixed, run `gpu.jl` again in Float64 and Float32.

## Notes for running on the cloud machine

  - The Manifest uses Julia 1.13.4.
  - The `add f32` commit has to be pushed, and pulled on the cloud machine, before the
    Float32 run. The older version of `gpu.jl` always uses Float64 and doesn't print a
    `Float type:` line.
  - Reactant's CUDA download is about 1.5 GB. It is only downloaded once per machine.
  - Run inside `tmux` or with `nohup ... &`, so that a dropped SSH connection doesn't
    kill a long compile.
  - To see what a running script is doing without stopping it:
    `kill -USR1 <pid>`. On Linux, Julia prints a short profile and carries on.
  - `TIME+` in `top` is CPU time added up across threads, not wall-clock time.
  - Copy results back from the laptop with
    `scp ubuntu@<ip>:'<repo>/bench/full-mnist/f*.txt' .`
