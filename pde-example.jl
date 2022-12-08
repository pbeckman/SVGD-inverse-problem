using LinearAlgebra, Plots, Plots.Measures
import ForwardDiff: gradient, Dual

include("./svgd.jl")

function lin_interp(x_ref, u_ref, xs)
  us = zeros(eltype(u_ref), length(xs))
  for (i, x) in enumerate(xs)
    ind   = findfirst(xr -> xr >= x, x_ref)
    us[i] = u_ref[ind-1] + (x - x_ref[ind-1])/(x_ref[ind] - x_ref[ind-1]) * (u_ref[ind] - u_ref[ind-1])
  end
  return us
end

function plot_distribution(x, logc, c_true; title="", legend=:none, ylabel="")
  M = length(logc)
  mu = reduce(.+, logc) / M
  pl = plot( 
    x, logc, 
    label="",
    linewidth=1, alpha=0.2, c=:darkorange2,
    title=title, legend=legend, xlabel="x", ylabel=ylabel
    )
  plot!(pl,
    x, reduce(.+, logc) / M, 
    label="post. mean",
    # marker=2,
    linewidth=3, c=:darkorange, linestyle=:dash,
    ribbon=2sqrt.(sum((.^).([lc .- mu for lc in logc], 2)) / M), fillalpha=0.3
    )
  # xgrid = range(0, stop=1, length=100)
  plot!(pl, 
    x, log.(c_true), 
    label="true",
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
  A = -1/h^2 * Tridiagonal(
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
  u = A \ vcat(a, f, b)
  
  return u
end

# boundary conditions u(0) = a, u(1) = b
a, b = 0, 0
# true permeability
# c_true_func(x) = 1e-1*(1 + x^2)
c_true_func(x) = 1e-1*(1 + x^2 + 0.2sin(4pi*x))
# iid noise variance for gaussian likelihood
s2 = 1e-5
# compute reference solution
n_ref = 40
f_ref = ones(n_ref-2)
x_ref = range(0, stop=1.0, length=n_ref)
u_ref = solve_poisson(c_true_func.(x_ref), f_ref, a, b)
# observation locations
n = 30
x = rand(n)
# generate data
y = lin_interp(x_ref, u_ref, x) .+ sqrt(s2) * randn(n)
# discretization of PDE
nd = 30
fd = ones(nd-2)
xd = range(0, stop=1.0, length=nd)

# Matern nu = 3/2 covariance function for prior
sc_prior, l_prior = 0.01, 2
function k_prior(xi, xj) 
  d = norm(xi - xj) / l_prior
  return sc_prior * (1 + sqrt(3)*d) * exp(-sqrt(3)*d)
end
# compute prior covariance matrix
K_prior = [k_prior(xi, xj) for xi in xd, xj in xd]
# select prior mean
mu_prior = fill(sum(log.(c_true_func.(xd)) / nd), nd)

# posterior target distribution
function logp(logc, xd, fd, a, b, y, s2, K_prior, mu_prior)
  ud = solve_poisson(exp.(logc), fd, a, b)
  return -norm(y - lin_interp(xd, ud, x))^2 / (2*s2) - dot((logc .- mu_prior), K_prior \ (logc .- mu_prior)) / 2
end

# gradient of log posterior, computed using adjoint method
function adj_grad_logp(logc, xd, fd, a, b, y, s2, K_prior, mu_prior)
  n = length(logc)
  h = 1/(n-1)
  # solve for u and adjoint variable p
  ud = solve_poisson(exp.(logc), fd, a, b)
  pd = solve_poisson(exp.(logc), (y-ud)/s2, 0, 0)
  # finite difference derivatives of u and p
  dudx = vcat(ud[1]-a, ud[2:end]-ud[1:end-1])*h
  dpdx = vcat(pd[1]-0, pd[2:end]-pd[1:end-1])*h
  # evaluate gradient formula
  grad = -2*K_prior \ logc
  for i = 1:n
    grad[i] += exp(logc[i]) * dudx[i] * dpdx[i]
  return grad
end

grad_logp(logc) = gradient(logc -> logp(logc, xd, fd, a, b, y, s2, K_prior, mu_prior), logc)

# choose initial particles from prior
M    = 10
logc = collect.(eachcol(cholesky(K_prior).L * randn(nd, M) .+ mu_prior))

# squared exponential kernel function for SVGD space
l = sum([norm(xi - xj) for xi in logc, xj in logc]) / M^2
l /= 10
@show l
k(xi, xj)      = exp(-norm(xi - xj)^2 / l^2)
grad_k(xi, xj) = -2/l^2 * (xi - xj) * exp(-norm(xi - xj)^2/l^2)

# iteration parameters
step_size = 1e-8
iter      = 10000

# run SVGD and make some intermediate plots
gr(size=(300, 300))
pls = []
push!(pls, plot_distribution(xd, logc, c_true_func.(xd), title="Iteration 0", legend=:topright, ylabel="logc"))
display(pls[end])
saveiters = [0, 100, 1000, 10000]
for i=2:length(saveiters)
  println("Running SVGD iterations $(saveiters[i-1]) - $(saveiters[i])...")
  logc .= SVGD(
    k, grad_k,
    grad_logp, logc, 
    step_size, saveiters[i] - saveiters[i-1], 
    verbose=false
    # bounds=(fill(-3, n), fill(-1, n))
    )
  push!(pls, plot_distribution(xd, logc, c_true_func.(xd), title="Iteration $(saveiters[i])"))
  display(pls[end])
end

# plot data
pl_data = plot(
  x_ref, u_ref, 
  label="true",
  linewidth=3, c=:blue,
  size=(300, 200)
  )
plot!(pl_data,
  xd, solve_poisson(exp.(sum(logc) / M), fd, a, b), 
  label="using post. mean",
  linewidth=3, c=:red, linestyle=:dash
  )
scatter!(pl_data, 
  x, y, 
  label="data",
  marker=4, c=:black
  )
display(pl_data)
readline(stdin)

l = @layout [grid(2,2)]
pl = plot(pls..., layout=l, size=(600, 600))
display(pl)
println("\nDisplaying final plot...")
readline(stdin)
