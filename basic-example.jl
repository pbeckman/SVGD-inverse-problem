using LinearAlgebra, Distributions, Plots, Plots.Measures

include("./svgd.jl")

function plot_histogram(x; title="", legend=:none)
  M = length(x)
  xgrid = range(-3, stop=3, length=100)
  pl = histogram(
    getindex.(x, 1), bins=-3:0.2:3, xlims=(-3, 3), ylims=(0, M/10*1.2),
    label="SVGD particles", title=title, legend=legend
    )
  plot!(pl, 
    xgrid, M/5*pdf.(Normal(), xgrid), 
    linewidth=3, label="Target distribution"
    )
  return pl
end

println("Setting up problem...\n")

# squared exponential kernel function 
k(xi, xj; l=1.0)      = exp(-norm(xi - xj)^2 / l^2)
grad_k(xi, xj; l=1.0) = 2/l^2 * norm(xi - xj) * sign.(xi - xj) * exp(-norm(xi - xj)^2/l^2)

# standard normal target distribution
logp(x) = -norm(x)^2 / 2 
grad_logp(x) = -sign.(x) * norm(x)

# choose uniform random initial particles
M = 200
x = [[2*(rand() - 0.5)] for _=1:M]

# iteration parameters
step_size = 1e-1
iter      = 200

# run SVGD and make some intermediate plots
gr(size=(300, 300))
pls = []
push!(pls, plot_histogram(x, title="Iteration 0", legend=:topright))
display(pls[end])
for i=1:4
  println("Running SVGD iterations $((i-1)*Int64(iter/4)) - $(i*Int64(iter/4))...")
  x .= SVGD(k, grad_k, grad_logp, x, step_size, iter/4)
  push!(pls, plot_histogram(x, title="Iteration $(i*Int64(iter/4))"))
  display(pls[end])
end

# plot all histograms
l = @layout [a b c d e]
pl = plot(pls..., layout=l, size=(1500, 300), margin=5mm)
display(pl)
println("\nDisplaying final plot...")
readline(stdin)

