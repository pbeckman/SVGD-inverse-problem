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

function L_mat(m)
  n = length(m)
  c = exp.(m)
  h = 1/(n-1)
  L = -1/h^2 * Tridiagonal(
      c[1:end-1],
      -vcat(2c[1], c[2:end-1]+c[3:end], 2c[end]), 
      c[2:end]
  )
  L[1, 1:2]         .= [1, 0]
  L[end, end-1:end] .= [0, 1]
  return L
end

function M_mat(u,m)
  # Implicitly applies c_{i+1} = 0 BCs on c!
  n = length(u)
  h = 1/(n-1)
  z = [exp(m[i])/m[i]*(u[i]-u[i-1])/h for i = 2:n]
  M = 1/h * Bidiagonal(
      vcat(0.0,z[1:n-2],0.0),
      vcat(0.0,-z[2:n-1]),
      :U
  )
  return M
end

function evidence(y, f, u_0, m_0, σ2, Λ, S)
  # p(y|f, σ2, Λ, u_0, m_0)
  # where f is PDE rhs [-∇.(c∇u) = f]
  # σ2 is Gaussian noise variance
  # Λ is precision of Gaussian prior for m
  # u_0, m_0 are (u,m) around which PDE is linearized
  n = length(y)
  L = L_mat(m_0)
  Linv = inv(L)
  M = M_mat(u_0, m_0)
  μ = -S*(L\(M*2f))
  Σ = 1e-12I(n) + σ2*S*S' + S*Linv*M*Λ*M'*Linv'*S'
  Sc = cholesky(0.5*(Σ+Σ'))
  return -(y.-μ)'*(Sc\(y.-μ)) - 0.5*logdet(Sc) - 0.5*log(2pi)
end

# Computing evidence using basic Metropolis-Hastings MCMC
function evid_MCMC(xd, fd, a, b, x, y, s2, K_prior, mu_prior, nsamp, step, m_0)
  n = length(xd)
  m = zeros(n,nsamp)
  vals = zeros(nsamp)
  # hard-coded initial logc
  m[:,1] = m_0
  accepted = 0
  for idx = 2:nsamp
    # isotropic Gaussian proposal distribution
    mnew = m[:,idx-1] + step*randn(n,1)
    pnew = logprior(mnew, K_prior, mu_prior)
    pold = logprior(m[:,idx-1], K_prior, mu_prior)
    #println(pnew, " \t", pold)
    # accept or reject
    if min(1,exp(pnew-pold)) > rand()
      accepted += 1
      m[:,idx] = mnew
    else
      m[:,idx] = m[:,idx-1]
    end
    vals[idx] = loglike(m[:,idx], xd, fd, a, b, x, y, s2)
  end
  # remove burnin
  vals = vals[(trunc(Int,nsamp/10)+1):end]
  return sum(vals)/length(vals), accepted/nsamp
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
  return S
end

println("Setting up problem...\n")

# boundary conditions u(0) = a, u(1) = b
a, b = 0, 0
# true permeability
c_true_func(x) = 1e-1*(1 + x^2 + 0.2sin(4pi*x))
# iid noise variance for gaussian likelihood
s2 = 1e-2
# compute reference solution
n_ref = 12
f_ref = ones(n_ref-2)
x_ref = range(0, stop=1.0, length=n_ref)
u_ref = solve_poisson(c_true_func.(x_ref), f_ref, a, b)
# observation locations
n = 8
x = rand(n)
# generate data
c_true = c_true_func.(x_ref)
y = sample(x, x_ref, u_ref) .+ sqrt(s2) * randn(n)

# Matern nu = 3/2 covariance function for prior
sc_prior, l_prior = 0.01, 0.5
function k_prior(xi, xj) 
  d = norm(xi - xj) / l_prior
  return sc_prior * (1 + sqrt(3)*d) * exp(-sqrt(3)*d)
end

# choose initial particles from prior
np = 10

# squared exponential kernel function for SVGD space
kl(xi, xj,l)      = exp(-norm(xi - xj)^2 / l^2)
grad_kl(xi, xj,l) = -2/l^2 * (xi - xj) * exp(-norm(xi - xj)^2/l^2)

# iteration parameters
step_size = 1e-7
iter      = 1000

# MCMC parameters
nsamp = 100000
step = 0.01

println("Performing model selection\n")

# Model selection
n_trial = 15:5:45
ev_trial = zeros(size(n_trial))
ev_MCMC = zeros(size(n_trial))
for i = 1:length(n_trial)
  # Setting up grid
  n_t = n_trial[i]
  f_t = ones(n_t-2)   # hax -- should correspond to f.(x_t)
  x_t = range(0, stop=1.0, length=n_t)

  print("n = ", n_t)

  # Calculating prior distribution
  K_prior = [k_prior(xi, xj) for xi in x_t, xj in x_t]
  #println(-logdet(K_prior))
  mu_prior = fill(sum(log.(c_true_func.(x_t)) / n_t), n_t)
  Kc = cholesky(K_prior)
  logc = collect.(eachcol(cholesky(K_prior).L * randn(n_t, np) .+ mu_prior))
  
  # Repulsion functions for SVGD
  l = sum([norm(xi - xj) for xi in logc, xj in logc]) / np^2
  k(xi,xj) = kl(xi,xj,l)
  grad_k(xi,xj) = grad_kl(xi,xj,l)
  
  # Find MAP logc for this grid
  logc .= SVGD(
    k, grad_k,
    logc -> grad_logp(logc, x_t, f_t, a, b, x, y, s2, K_prior, mu_prior), logc, 
    step_size, iter, 
    verbose=false
    )
  c_0 = reduce(.+,logc)/length(logc)
  u_0 = solve_poisson(c_0, f_t, a, b)
  
  # Linearize likelihood around this MAP solution and calculate evidence
  Λ = inv(Kc)
  S = S_mat(x,x_t)
  ev_trial[i] = evidence(y, vcat(a,f_t,b), u_0, c_0, s2, Λ, S)
  ev_MCMC[i], accrate = evid_MCMC(x_t, f_t, a, b, x, y, s2, K_prior, mu_prior, nsamp, step, c_0)
  println("accept rate = ", accrate)
end


gui(plot(n_trial, ev_trial, title="Log evidence vs. number of grid pts",
                            xlabel="n",
                            ylabel="log evidence",
                            linewidth=2,
                            legend=false))
                            
gui(plot(n_trial, ev_MCMC, title="MCMC log evidence vs. number of grid pts",
                            xlabel="n",
                            ylabel="log evidence",
                            linewidth=2,
                            legend=false))
