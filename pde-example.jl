using LinearAlgebra, Plots, Plots.Measures
import ForwardDiff: gradient, Dual

include("./svgd.jl")

function plot_distribution(x, logc, c_true; title="", legend=:none)
  M = length(logc)
  mu = reduce(.+, logc) / M
  pl = plot(
    x, reduce(.+, logc) / M, 
    label="SVGD posterior mean (±2σ)",
    # marker=2,
    linewidth=3, c=:orange,
    ribbon=2sqrt.(sum((.^).([lc .- mu for lc in logc], 2)) / M), fillalpha=0.5,
    title=title, legend=legend
    )
  # xgrid = range(0, stop=1, length=100)
  plot!(pl, 
    x, logc, 
    label="",
    linewidth=1, alpha=0.2
    )
  plot!(pl, 
    x, log.(c_true), 
    label="True log-permeability",
    # marker=2,
    linewidth=3, c=:purple
    )
  return pl
end

println("Setting up problem...\n")

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


  # if eltype(c) <: Dual
  #   # @show getfield.(c, :value)
  #   @show eigvals(getfield.(Matrix(A), :value))
  # else
  #   # @show c
  #   @show eigvals(Matrix(A))
  # end

  # solve PDE
  u = (A \ vcat(a, f, b))[2:end-1]
  
  return u
end

# regular grid for observations and PDE discretization
n = 30
x = range(0, stop=1.0, length=n)
# boundary conditions u(0) = a, u(1) = b
a, b = 0, 0
# forcing
f = ones(n-2)
# true permeability
# c_true = 1e-1*(1 .+ x .^ 2)
c_true = 1e-1*(1 .+ x .^ 2 .+ 0.2sin.(4pi*x))
# iid noise variance for gaussian likelihood
s2 = 1e-7
# generate data
u_true = solve_poisson(c_true, f, a, b)
y      = u_true .+ sqrt(s2) * randn(n-2)

# Matern nu = 3/2 covariance function for prior
sc_prior, l_prior = 0.1, 10
function k_prior(xi, xj) 
  d = norm(xi - xj) / l_prior
  return sc_prior * (1 + sqrt(3)*d) * exp(-sqrt(3)*d)
end
# compute prior covariance matrix
K_prior = [k_prior(xi, xj) for xi in x, xj in x]
# select prior mean
mu_prior = fill(sum(log.(c_true) / n), n)

# posterior target distribution
function logp(logc, f, a, b, y, s2, K_prior, mu_prior) 
  u = solve_poisson(exp.(logc), f, a, b)
  return -norm(y - u)^2 / (2*s2) - dot((logc .- mu_prior), K_prior \ (logc .- mu_prior)) / 2
end
grad_logp(logc) = gradient(logc -> logp(logc, f, a, b, y, s2, K_prior, mu_prior), logc)

# choose initial particles from prior
M    = 20
logc = collect.(eachcol(cholesky(K_prior).L * randn(n, M) .+ mu_prior))

# squared exponential kernel function for SVGD space
l = sum([norm(xi - xj) for xi in logc, xj in logc]) / M^2
l /= 10
@show l
k(xi, xj)      = exp(-norm(xi - xj)^2 / l^2)
grad_k(xi, xj) = -2/l^2 * (xi - xj) * exp(-norm(xi - xj)^2/l^2)

# iteration parameters
step_size = 1e-8
iter      = 100

# run SVGD and make some intermediate plots
gr(size=(300, 300))
pls = []
push!(pls, plot_distribution(x, logc, c_true, title="Iteration 0", legend=:topright))
display(pls[end])
nfig = 6
for i=1:(nfig-1)
  println("Running SVGD iterations $((i-1)*Int64(iter/(nfig-1))) - $(i*Int64(iter/(nfig-1)))...")
  logc .= SVGD(
    k, grad_k,
    grad_logp, logc, 
    step_size, round(iter/(nfig-1)), 
    verbose=false
    # bounds=(fill(-3, n), fill(-1, n))
    )
  push!(pls, plot_distribution(x, logc, c_true, title="Iteration $(i*Int64(iter/(nfig-1)))"))
  display(pls[end])
end

# plot all histograms
pl_data = plot(
  x, vcat(a, u_true, b), 
  label="True solution",
  linewidth=3, marker=2,  c=:blue
  )
plot!(pl_data,
  x, vcat(a, solve_poisson(exp.(sum(logc) / M), f, a, b), b), 
  label="Solution with mean permeability",
  linewidth=3, marker=2, c=:red, linestyle=:dash
  )
scatter!(pl_data, 
  x[2:end-1], y, 
  label="Noisy data",
  marker=5, c=:black
  )
l = @layout [a{0.3w} grid(2,3)]
pl = plot(pl_data, pls..., layout=l, size=(1500, 500), margin=2mm)
display(pl)
println("\nDisplaying final plot...")
# readline(stdin)
