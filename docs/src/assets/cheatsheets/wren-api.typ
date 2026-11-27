// Wren.jl API refresher: A4 landscape, three columns of accent cards.
// The code boxes show regions of the scripts in api/, one per card, which the Wren test
// suite runs.
// The logo lives one level up, so compile from this directory with
//   typst compile --root .. wren-api.typ

#let julia = (
  green: rgb("#389826"),
  purple: rgb("#9558B2"),
  red: rgb("#CB3C33"),
  blue: rgb("#4063D8"),
)
#let ink = rgb("#1A1A2E")
#let neutral = rgb("#555a66")

#let body-font = "IBM Plex Sans"
#let code-font = "JuliaMono"
#let body-size = 7.4pt
#let code-size = 6.9pt
#let gutter = 4.5mm

#let page-margin = (x: 10mm, y: 9mm)
#let column-width = (297mm - 2 * page-margin.x - 2 * gutter) / 3

// Every line inside a code box, empty or not, is exactly one line pitch tall:
// fixed text edges give each line the same box, and leading is zero.
#let line-pitch = 9.5pt
#let line-top-edge = 7pt

#let panel-inset-x = 6pt
// From the baseline of a panel title to the rule below it; card titles keep the same
// distance to their first panel.
#let header-gap = line-pitch / 2 + line-pitch - line-top-edge

#let wren-docs = "https://pysm.dev/Wren.jl/"
#let flexichains-docs = "https://juliabayes.org/FlexiChains.jl/stable/"

// The accent mixed 9% into white in sRGB, as Sass `mix` does on the website.
#let tint(accent) = color.mix((accent, 9%), (white, 91%), space: rgb)

// Phosphor SVGs draw with `currentColor`, which Typst does not resolve.
#let icon(name, fill, height) = image(
  bytes(read("icons/" + name + "-bold.svg").replace("currentColor", fill.to-hex())),
  format: "svg",
  height: height,
)

// The lines of api/<card>.jl between `#region name` and the next `#endregion`, with
// their common indentation removed.
#let snippet(card, name) = {
  let path = "scripts/" + card + ".jl"
  let lines = read(path).split("\n")
  let first = lines.position(line => line.trim() == "#region " + name)
  assert(first != none, message: path + " has no region " + name)
  let body = lines.slice(first + 1)
  let len = body.position(line => line.trim() == "#endregion")
  assert(len != none, message: "region " + name + " in " + path + " has no #endregion")
  body = body.slice(0, len)
  let indents = body.filter(line => line.trim() != "").map(line => line.len() - line.trim(at: start).len())
  assert(indents.len() > 0, message: "region " + name + " in " + path + " is empty")
  let indent = calc.min(..indents)
  body.map(line => if line.trim() == "" { "" } else { line.slice(indent) }).join("\n")
}

#let pitfall-row(accent, body) = grid(
  columns: (auto, 1fr),
  column-gutter: 3pt,
  icon("warning", accent.darken(15%), 0.75em),
  text(fill: accent.darken(30%), size: 6.8pt, body),
)

#let panel-header(accent, title, url) = grid(
  columns: (1fr, auto),
  text(size: 8pt, weight: 600, fill: accent.darken(20%), title),
  link(url, icon("link", accent, 7pt)),
)

#let outlined-panel(accent, title, url, code, pitfall: none) = {
  set text(top-edge: line-top-edge, bottom-edge: line-top-edge - line-pitch)
  set par(leading: 0pt)
  let box-stroke = 1pt + color.mix((accent, 16%), (white, 84%), space: rgb)
  block(
    width: 100%,
    fill: white,
    stroke: box-stroke,
    radius: 5pt,
    grid(
      columns: 1fr,
      inset: (x: panel-inset-x, y: line-pitch / 2),
      panel-header(accent, title, url),
      grid.hline(stroke: box-stroke),
      raw(code, lang: "julia", block: true),
      ..if pitfall != none { (grid.hline(stroke: box-stroke), pitfall-row(accent, pitfall)) },
    ),
  )
}

// Bootstrap's translucent border colour, which the website uses between panel rows.
#let panel-rule-stroke = 1pt + black.transparentize(82.5%)

// The rule above the pitfall row stops where the text does, as on the website.
#let tinted-panel(accent, title, url, code, pitfall: none) = {
  set text(top-edge: line-top-edge, bottom-edge: line-top-edge - line-pitch)
  set par(leading: 0pt)
  let rule = block(inset: (x: panel-inset-x), line(length: 100%, stroke: panel-rule-stroke))
  let row(body) = block(width: 100%, inset: (x: panel-inset-x, y: line-pitch / 2), body)
  block(
    width: 100%,
    fill: tint(accent),
    radius: 5pt,
    stack(
      row(stack(spacing: line-pitch / 2, panel-header(accent, title, url), raw(code, lang: "julia", block: true))),
      ..if pitfall != none { (rule, row(pitfall-row(accent, pitfall))) },
    ),
  )
}

#let panel = tinted-panel

#let card-title(accent, title) = text(size: 10.5pt, weight: 700, fill: accent, title)

#let card(accent, title, height: auto, ..panels) = block(
  width: 100%,
  height: height,
  fill: white,
  stroke: 2pt + accent,
  radius: 8pt,
  inset: (x: 7pt, top: header-gap, bottom: 7pt),
  stack(card-title(accent, title), v(header-gap), stack(spacing: line-pitch / 2, ..panels.pos())),
)

// Cards side by side in one row, all as tall as the tallest of them.
#let level-cards(..card-makers) = context {
  let heights = card-makers.pos().map(make => measure(make(auto), width: column-width).height)
  let tallest = calc.max(..heights)
  grid(
    columns: (1fr,) * heights.len(),
    column-gutter: gutter,
    ..card-makers.pos().map(make => make(tallest)),
  )
}

#set page(paper: "a4", flipped: true, margin: page-margin)
#set text(font: body-font, size: body-size, fill: ink)
#set par(leading: 0.45em)
#set raw(theme: "juliabayes-light.tmTheme")
#show raw: set text(font: code-font, size: code-size)

#let masthead = grid(
  columns: (auto, auto),
  column-gutter: 5mm,
  align: (horizon, horizon),
  image("../logo.svg", height: 52pt),
  stack(
    spacing: 8pt,
    text(size: 18pt, weight: 700, "Wren.jl"),
    text(size: 11pt, fill: neutral, "API overview"),
  ),
)

#let models = julia.purple
#let evaluation = julia.blue
#let inference = julia.green
#let chains = julia.red
#let dists = neutral

#let models-card = card(
  models,
  "Models",
  panel(
    models,
    "Sample statements",
    wren-docs + "tutorials/syntax/#sample-with-distributions",
    snippet("models", "sample-statements"),
  ),
  panel(models, "Model", wren-docs + "tutorials/workflow/#workflow", snippet("models", "model")),
)

#let conditioning-card = card(
  evaluation,
  "Conditioning & evaluation",
  height: 100%,
  panel(evaluation, "Conditioning", wren-docs + "tutorials/models/#Conditioning", snippet("conditioning", "conditioning")),
)

#let inference-card(height) = card(
  inference,
  "Inference",
  height: height,
  panel(inference, "Sampling", wren-docs + "tutorials/mcmc/#nuts", snippet("inference", "sampling")),
  panel(
    inference,
    "Point estimates",
    wren-docs + "howtos/optimization/#Convenience-functions",
    snippet("inference", "point-estimates"),
  ),
)

#let chains-card(height) = card(
  chains,
  "Chains",
  height: height,
  panel(chains, "Indexing", flexichains-docs + "indexing/#Chains", snippet("chains", "indexing")),
  panel(chains, "Summaries", flexichains-docs + "summarising/#Individual-statistics", snippet("chains", "summaries")),
  panel(chains, "Diagnostics", flexichains-docs + "summarising/#mcmc-diagnostics", snippet("chains", "diagnostics")),
)

#let dists-panel(title, ..signatures) = panel(
  dists,
  title,
  wren-docs + "tutorials/distributions/#wrendists",
  signatures.pos().join("\n"),
)

#let dists-column(..panels) = stack(spacing: line-pitch / 2, ..panels)

#let dists-card = block(
  width: 100%,
  height: 100%,
  fill: white,
  stroke: 2pt + dists,
  radius: 8pt,
  inset: (x: 7pt, top: header-gap, bottom: 7pt),
  stack(
    card-title(dists, "WrenDists"),
    v(header-gap),
    grid(
      columns: (1fr, 1fr, 1fr),
      column-gutter: line-pitch / 2,
      dists-column(
        dists-panel("Unbounded", "Normal(μ, σ)", "Cauchy(μ, σ)", "Laplace(μ, b)", "TDist(ν, μ, σ)", "Flat()"),
        dists-panel("Bounded", "Beta(α, β)", "Uniform(a, b)"),
      ),
      dists-column(
        dists-panel(
          "Positive",
          "Exponential(θ)",
          "Gamma(α, θ)",
          "InverseGamma(α, θ)",
          "LogNormal(μ, σ)",
          "Weibull(α, θ)",
          "HalfNormal(μ, σ)",
          "HalfCauchy(μ, σ)",
          "HalfTDist(ν, μ, σ)",
        ),
        dists-panel("Multivariate", "MvNormalScale(μ, L)", "Dirichlet(α)", "Multinomial(n, p)"),
      ),
      dists-column(
        dists-panel(
          "Discrete",
          "Bernoulli(p)",
          "BernoulliLogit(logitp)",
          "Binomial(n, p)",
          "BinomialLogit(n, logitp)",
          "Categorical(p)",
          "Geometric(p)",
          "NegativeBinomial(r, p)",
          "Poisson(λ)",
          "PoissonLog(logλ)",
        ),
      ),
    ),
  ),
)

#grid(
  columns: (column-width, 1fr),
  rows: 1fr,
  column-gutter: gutter,
  grid(
    rows: (auto, auto, 1fr),
    row-gutter: gutter,
    masthead, models-card, conditioning-card,
  ),
  grid(
    rows: (auto, 1fr),
    row-gutter: gutter,
    level-cards(inference-card, chains-card),
    dists-card,
  ),
)
