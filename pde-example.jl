using LinearAlgebra, Plots, Plots.Measures,Random

include("./svgd.jl")
include("./pde-setup.jl")

Random.seed!(42)

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

function plot_dist_nosamples(x, logc, c_true; title="", legend=:none, ylabel="")
  M = length(logc)
  mu = reduce(.+, logc) / M
  pl = plot( 
    x, mu, 
    label="post. mean",
    # marker=2,
    linewidth=3, c=:darkorange, linestyle=:dash,
    ribbon=2sqrt.(sum((.^).([lc .- mu for lc in logc], 2)) / M), fillalpha=0.3,
    ylims=(minimum(log.(c_true)) - 0.2, maximum(log.(c_true)) + 0.2),
    title=title, legend=legend, xlabel="x", ylabel=ylabel
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

# boundary conditions u(0) = a, u(1) = b
a, b = 0, 0
# true permeability
c_true_func(x) = 1e-1*(1 + x^2 + 0.2sin(4pi*x))
# iid noise variance for gaussian likelihood
s2 = 1e-3
# compute reference solution
n_ref = 40
f_ref = ones(n_ref-2)
x_ref = range(0, stop=1.0, length=n_ref)
u_ref = solve_poisson(c_true_func.(x_ref), f_ref, a, b)
# observation locations
n = 30
x = rand(n)
# generate data
y = sample(x, x_ref, u_ref) .+ sqrt(s2) * randn(n)
# discretization of PDE
nd = 1000
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
iter      = 100000

# run MCMC and plot
logc0 = cholesky(K_prior).L * randn(nd) .+ mu_prior
nsamp = 2
step = 0.001
logcMCMC,accrate = post_MCMC(xd, fd, a, b, x, y, s2, K_prior, mu_prior, nsamp, step, logc0)
@show accrate
plMCMC = plot_dist_nosamples(xd, logcMCMC, c_true_func.(xd), title="MCMC", legend=:topright, ylabel="m")
display(plMCMC)
#=

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
=#

# plot data
pl_data = plot(
  x_ref, u_ref, 
  label="true",
  linewidth=3, c=:blue,
  size=(300, 200)
  )
plot!(pl_data,
  xd, solve_poisson(exp.(sum(logcMCMC) / length(logcMCMC)), fd, a, b), 
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

#=
l = @layout [grid(2,2)]
pl = plot(pls..., layout=l, size=(600, 600))
display(pl)
println("\nDisplaying final plot...")
# readline(stdin)
=#