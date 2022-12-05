using LinearAlgebra, Plots, Plots.Measures
using Interpolations
import ForwardDiff: gradient

include("./svgd.jl")

function plot_distribution(x, logc, c_true; title="", legend=:none)
  M = length(logc)
  pl = plot(
    x, sum(logc) / M, 
    linewidth=2, marker=2, label="SVGD posterior mean (±2σ)", 
    ribbon=2sqrt.(sum((.^).(logc, 2)) / M), fillalpha=0.5,
    title=title, legend=legend
    )
  # xgrid = range(0, stop=1, length=100)
  plot!(pl, 
    x, log.(c_true), 
    linewidth=2, marker=2, label="True log-permeability"
    )
  return pl
end

println("Setting up problem...\n")

# squared exponential kernel function for SVGD space
k(xi, xj; l=1.0)      = exp(-norm(xi - xj)^2 / l^2)
grad_k(xi, xj; l=1.0) = 2/l^2 * norm(xi - xj) * sign.(xi - xj) * exp(-norm(xi - xj)^2/l^2)

function L_mat(c)
  n = length(c)
  h = 1/(n-1)
  L = 1/h^2 * Tridiagonal(
      c[1:end-1],
      -vcat(2c[1], c[2:end-1]+c[3:end], 2c[end]), 
      c[2:end]
  )
  return L
end

function M_mat(u)
  # Implicitly applies c_{i+1} = 0 BCs on c!
  n = length(u)
  h = 1/(n-1)
  M = 1/h^2 * Bidiagonal(
      vcat(u[1], u[2:end].-u[1:end-1]),
      u[2:end].-u[1:end-1],
      :U
  )
  return M
end

function evidence(y, f, u_0, c_0, σ2, Λ)
  # p(y|f, σ2, Λ, u_0, c_0)
  # where f is PDE rhs [-∇.(c∇u) = f]
  # σ2 is Gaussian noise variance
  # Λ is precision of Gaussian prior for c
  # u_0, c_0 are (u,c) around which PDE is linearized
  n = length(y)
  L = L_mat(c_0)
  Linv = inv(L)
  M = M_mat(u_0)
  μ = L\(M*2f)
  Σ = σ2*I(n) + Linv*M*Λ*M'*Linv'
  S = cholesky(Σ)
  return 1/sqrt(2pi)*exp((y.-μ)'*(S\(y.-μ)) - 0.5*logdet(S))
end

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

  # if eltype(c) == Float64
  #   @show c
  # else
  #   @show getfield.(c, :value)
  # end

  # solve PDE
  u = (A \ vcat(a, f, b))[2:end-1]
  
  return u
end

function svgd_solve!(logc, x, y, l, M, step_size, iter, verb=False, c_debug=nothing)
  if verb
    gr(size=(300, 300))
    pls = []
    push!(pls, plot_distribution(x, logc, c_true, title="Iteration 0", legend=:topright))
    display(pls[end])
  end
  for i=1:5
    println("Running SVGD iterations $((i-1)*Int64(iter/5)) - $(i*Int64(iter/5))...")
    logc .= SVGD(
      (xi, xj) -> k(xi, xj, l=l), (xi, xj) -> grad_k(xi, xj, l=l), 
      grad_logp, logc, 
      step_size, iter/5, 
      verbose=false
      )
    if verb
      push!(pls, plot_distribution(x, logc, c_true, title="Iteration $(i*Int64(iter/5))"))
      display(pls[end])
    end
  end
end

function sample(xx,X,U)
  # sample a rectangular grid (X,U) onto irregular points xx[i]
  # xx should be an array (in 1D) or an array of tuples (in ND)
  # X should be an array (in 1D) or a tuple of arrays (in ND)
  interp_linear = linear_interpolation(X,U)
  return [interp_linear(xx[i]) for i = 1:length(xx)]
end

# regular grid for PDE discretization
n = 20
x = range(0, stop=1.0, length=n)
# boundary conditions u(0) = a, u(1) = b
a, b = 1, -1
# forcing
f = zeros(n-2)
# true permeability
c_true = 0.1 .+ 0.5*x .^ 2
# iid noise variance for gaussian likelihood
s2 = 1e-2
# generate data
u_true = solve_poisson(c_true, f, a, b)
#y      = u_true .+ sqrt(s2) * randn(n-2)

# Irregular grid for observations
n_obs = 17
x_obs = rand(n_obs)
y     = sample(x_obs,x,u_true) .+ sqrt(s2) * randn(n-2)

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

# lengthscale for SVGD kernel
l = 0.01
# choose initial particles from prior
M    = 20
logc = collect.(eachcol(cholesky(K_prior).L * randn(n, M)))

# iteration parameters
step_size = 1e-5
iter      = 10000

# run SVGD and make some intermediate plots
svgd_solve!(logc, x, y, l, M, step_size, iter, true, c_true)

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
l = @layout [a{0.4w} grid(2,3)]
pl = plot(pl_data, pls..., layout=l, size=(1500, 500), margin=5mm)
display(pl)
println("\nDisplaying final plot...")
readline(stdin)


# Model selection
n_trial = 10:30
ev_trial = zeros(size(n_trial))
for i = 1:length(n_trial)
  n = n_trial[i]
  x = range(0, stop=1.0, length=n)
  K_prior = [k_prior(xi, xj) for xi in x, xj in x]
  Kc = cholesky(K_prior)
  logc = collect.(eachcol(Kc.L * randn(n, M)))
  Λ = inv(Kc)
  ev_trial[i] = evidence(y, f, u_0, c_0, s2, Λ)
end