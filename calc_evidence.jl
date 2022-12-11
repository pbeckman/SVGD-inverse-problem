using LinearAlgebra, Plots, Plots.Measures, SparseArrays

include("./svgd.jl")
include("./pde-setup.jl")

function plot_distribution(x, logc, c_true; title="", legend=:none, ylabel="")
  M = length(logc)
  mu = reduce(.+, logc) / M
  pl = plot( 
    x, logc, 
    label="",
    linewidth=1, alpha=0.2, c=:darkorange2, 
    ylims=(minimum(log.(c_true)) - 0.2, maximum(log.(c_true)) + 0.2),
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

function evidence(y, f, u_0, c_0, σ2, Λ, S)
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
  Σ = σ2*S*S' + S*Linv*M*Λ*M'*Linv'*S'
  Sc = cholesky(Σ)
  return 1/sqrt(2pi)*exp((y.-μ)'*(Sc\(y.-μ)) - 0.5*logdet(S))
end

function bisect_srch(x,Y)
  n = length(Y)
  lo = 1
  hi = n
  while lo+1 < hi
    ix = (hi+lo)>>1
    yy = Y[ix]
    if yy <= x
      lo = ix
    else
      hi = ix
    end
  end
  return lo
end

function S_mat(xx,X)
  # interpolation matrix constructor
  m = length(xx)
  n = length(X)
  S = zeros(m,n)
  for i = 1:m
    j = bisect_srch(xx[i],X)
    d = (xx[i]-X[j])/(X[j+1]-X[j])
    S[i,j] = d
    S[i,j+1] = 1-d
  end
  return sparse(S)
end

println("Setting up problem...\n")

# boundary conditions u(0) = a, u(1) = b
a, b = 0, 0
# true permeability
c_true_func(x) = 1e-1*(1 + x^2 + 0.2sin(4pi*x))
# iid noise variance for gaussian likelihood
s2 = 1e-2
# compute reference solution
n_ref = 40
f_ref = ones(n_ref-2)
x_ref = range(0, stop=1.0, length=n_ref)
u_ref = solve_poisson(c_true_func.(x_ref), f_ref, a, b)
# observation locations
n = 12
x = rand(n)
# generate data
c_true = c_true_func.(x_ref)
y = sample(x, x_ref, u_ref) .+ sqrt(s2) * randn(n)
# discretization of PDE
nd = 15
fd = ones(nd-2)
xd = range(0, stop=1.0, length=nd)

# Matern nu = 3/2 covariance function for prior
sc_prior, l_prior = 0.01, 0.5
function k_prior(xi, xj) 
  d = norm(xi - xj) / l_prior
  return sc_prior * (1 + sqrt(3)*d) * exp(-sqrt(3)*d)
end
# compute prior covariance matrix
K_prior = [k_prior(xi, xj) for xi in xd, xj in xd]
# select prior mean
mu_prior = fill(sum(log.(c_true_func.(xd)) / nd), nd)

# choose initial particles from prior
M    = 10
logc = collect.(eachcol(cholesky(K_prior).L * randn(nd, M) .+ mu_prior))

# squared exponential kernel function for SVGD space
l = sum([norm(xi - xj) for xi in logc, xj in logc]) / M^2
l /= 1
@show l
k(xi, xj)      = exp(-norm(xi - xj)^2 / l^2)
grad_k(xi, xj) = -2/l^2 * (xi - xj) * exp(-norm(xi - xj)^2/l^2)

# iteration parameters
step_size = 1e-7
iter      = 1000

# run SVGD and make some intermediate plots
gr(size=(300, 300))
pls = []
push!(pls, plot_distribution(xd, logc, c_true_func.(xd), title="Iteration 0", legend=:topright, ylabel="logc"))
display(pls[end])
saveiters = [100, 1000, 10000]
for i=1:length(saveiters)
  println("Running SVGD iterations $(i == 1 ? 0 : saveiters[i-1]) - $(saveiters[i])...")
  logc .= SVGD(
    k, grad_k,
    logc -> grad_logp(logc, xd, fd, a, b, x, y, s2, K_prior, mu_prior), logc, 
    step_size, saveiters[i] - (i == 1 ? 0 : saveiters[i-1]), 
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
# readline(stdin)
l = @layout [grid(2,2)]
pl = plot(pls..., layout=l, size=(600, 600))
gui(pl)
println("\nDisplaying final plot...")


# Model selection
n_trial = 8:17
ev_trial = zeros(size(n_trial))
c_0 = reduce(.+,logc)
u_0 = 
for i = 1:length(n_trial)
  n_t = n_trial[i]
  f_t = ones(n_trial)   # hax -- should correspond to f.(x_t)
  x_t = range(0, stop=1.0, length=n_t)
  K_prior = [k_prior(xi, xj) for xi in x_t, xj in x_t]
  Kc = cholesky(K_prior)
  Λ = inv(Kc)
  S = S_mat(x,x_t)
  ev_trial[i] = evidence(y, f_t, u_0, c_0, s2, Λ, S)
end

gui(plot(n_trial, ev_trial, title="Evidence vs. number of grid pts"))