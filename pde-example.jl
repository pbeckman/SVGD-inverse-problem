using LinearAlgebra, Random, Plots, Plots.Measures

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
    title=title, legend=legend, xlabel="x", ylabel=ylabel, 
    size=(800, 600), margins=2mm
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

# no surprises
Random.seed!(5)

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
n = 30
x = rand(n)
# generate data
y = sample(x, x_ref, u_ref) .+ sqrt(s2) * randn(n)
# discretization of PDE
nd = 40
fd = ones(nd-2)
xd = range(0, stop=1.0, length=nd)

# plot true solution and data
# pl_data = plot(
#   x_ref, u_ref, 
#   label="true",
#   linewidth=2, c=:blue,
#   xlabel="x", ylabel="u",
#   size=(600, 400)
#   )
# gr(size=(1600,300))
pl_data = scatter(
  x, y, 
  label="data",
  marker=4, c=:black,
  xlabel="x", ylabel="u",
  # size=(800, 600)
  )
l = @layout [a]
plot(pl_data, layout=l, size=(1000, 800), leftmargin=-3mm)
gui(pl_data)
readline(stdin)

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
l = sum([norm(xi - xj) for xi in logc, xj in logc]) / M^2 / 2
println("SVGD lengthscale ")
k(xi, xj, l)      = exp(-norm(xi - xj)^2 / l^2)
grad_k(xi, xj, l) = -2/l^2 * (xi - xj) * exp(-norm(xi - xj)^2/l^2)

# iteration parameters
step_size = 1e-7
iter      = 10000

# run SVGD and make some intermediate plots
# gr(size=(300,300))
pls = []
push!(pls, plot_distribution(xd, logc, c_true_func.(xd), title="Iteration 0", legend=:topright, ylabel="m"))
gui(pls[end])
saveiters = [100, 1000, 10000]
plotiters = 10:10:saveiters[end]
# plotiters = vcat(100:100:9900, 10000:1000:49000, 50000:10000:100000)
for i=1:length(plotiters)
  println("Running SVGD iterations $(i == 1 ? 0 : plotiters[i-1]) - $(plotiters[i])...")
  logc .= SVGD(
    (xi, xj) -> k(xi, xj, l),
    (xi, xj) -> grad_k(xi, xj, l),
    logc -> grad_logp(logc, xd, fd, a, b, x, y, s2, K_prior, mu_prior), logc, 
    plotiters[i]^0.3 * step_size, 
    # step_size,
    plotiters[i] - (i == 1 ? 0 : plotiters[i-1]), 
    verbose=false
    # bounds=(fill(-3, n), fill(-1, n))
    )
  pli = plot_distribution(xd, logc, c_true_func.(xd), title="Iteration $(plotiters[i])", ylabel="m")
  gui(pli)
  if plotiters[i] in saveiters
    push!(pls, pli)
  end
end
readline(stdin)

# plot data
# gr(size=(600, 300))
minlogc, maxlogc = extrema(vcat(logc...))
logc_grid = range(minlogc, stop=maxlogc, length=100)
logc_smooth = mapreduce(
  i -> [k(lcg, lc, 0.2) for lcg in logc_grid, lc in logc[i]], 
  .+, 
  eachindex(logc)
  )
logc_map = [logc_grid[findmax(col)[2]] for col in eachcol(logc_smooth)]
u_map =  solve_poisson(exp.(logc_map), fd, a, b)
us    = [solve_poisson(exp.(lc), fd, a, b) for lc in logc]
pl_data = plot( 
    xd, us,
    label="",
    linewidth=1, alpha=0.2, c=:red, 
    xlabel="x", ylabel="u",
    # size=(800, 600)
)
plot!(pl_data,
  xd, u_map,
  label="using post. mean",
  linewidth=3, c=:red, linestyle=:dash,
  ribbon=2sqrt.(sum((.^).([u .- u_map for u in us], 2)) / M), fillalpha=0.2
  )
plot!(pl_data,
  x_ref, u_ref, 
  label="true",
  linewidth=2, c=:blue
  )
scatter!(pl_data, 
  x, y, 
  label="data",
  marker=4, c=:black
  )
# gui(pl_data)
# readline(stdin)

l = @layout [
  grid(2,2) 
  a{0.3h}
]
pl = plot(pls..., pl_data, layout=l, size=(1000, 700))
gui(pl)
println("\nDisplaying final plot...")
# readline(stdin)