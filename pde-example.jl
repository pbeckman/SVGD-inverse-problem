using LinearAlgebra, Plots, Plots.Measures
import ForwardDiff: gradient

include("./svgd.jl")

function plot_distribution(x, logc, c_true; title="", legend=:none)
  M = length(logc)
  pl = plot(
    x, exp.(sum(logc) / M), 
    linewidth=2, label="SVGD posterior mean", 
    title=title, legend=legend
    )
  # xgrid = range(0, stop=1, length=100)
  plot!(pl, 
    x, c_true, 
    linewidth=2, label="True permeability"
    )
  return pl
end

# squared exponential kernel function for SVGD space
k(xi, xj; l=1.0)      = exp(-norm(xi - xj)^2 / l^2)
grad_k(xi, xj; l=1.0) = 2/l^2 * norm(xi - xj) * sign.(xi - xj) * exp(-norm(xi - xj)^2/l^2)

function solve_poisson(c, f, a, b)
  n = length(c)
  h = 1/(n-1)

  # build finite difference system matrix
  A = 1/h^2 * Tridiagonal(
    c[1:end-1],
    -vcat(2c[1], c[2:end-1]+c[3:end], 2c[end]), 
    c[2:end]
    )
  
  # enforce boundary conditions u(0) = a, u(1) = b
  A[1, 1:2]         .= [1, 0]
  A[end, end-1:end] .= [0, 1]

  # solve PDE
  u = (A \ vcat(a, f, b))[2:end-1]

  return u
end

# regular grid
n = 10
x = range(0, stop=1.0, length=n)
# boundary conditions u(0) = a, u(1) = b
a, b = 1, -1
# forcing
f = zeros(n-2)
# true permeability
c_true = 0.5 .+ x .^ 2
# iid noise variance for gaussian likelihood
s2 = 1e-2
# generate data
y = solve_poisson(c_true, f, a, b) .+ sqrt(s2)*randn(n-2)

# Matern nu = 3/2 covariance function for prior
function k_prior(xi, xj; sc=1.0, l=1.0) 
  d = norm(xi - xj) / l
  return sc * (1 + sqrt(3)*d) * exp(-sqrt(3)*d)
end
# compute prior covariance matrix
K_prior = [k_prior(xi, xj) for xi in x, xj in x]

# posterior target distribution
function logp(logc, f, a, b, y, s2, K_prior) 
  u = solve_poisson(exp.(logc), f, a, b)
  return -norm(y - u)^2 / (2*s2) - dot(logc, K_prior \ logc) / 2
end
grad_logp(logc) = gradient(logc -> logp(logc, f, a, b, y, s2, K_prior), logc)

# choose initial particles from prior
M    = 2
logc = collect.(eachcol(cholesky(K_prior).L * randn(n, M)))

# iteration parameters
step_size = 1e-3
iter      = 1000

# run SVGD and make some intermediate plots
gr(size=(300, 300))
pls = []
push!(pls, plot_distribution(x, logc, c_true, title="Iteration 0", legend=:topright))
display(pls[end])
for i=1:4
  println("Running SVGD iterations $((i-1)*Int64(iter/4)) - $(i*Int64(iter/4))...")
  logc .= SVGD(k, grad_k, grad_logp, logc, step_size, iter/4, verbose=true)
  push!(pls, plot_distribution(x, logc, c_true, title="Iteration $(i*Int64(iter/4))"))
  display(pls[end])
end

# plot all histograms
l = @layout [a b c d e]
pl = plot(pls..., layout=l, size=(1500, 300), margin=5mm)
display(pl)
println("\nDisplaying final plot...")
readline(stdin)
