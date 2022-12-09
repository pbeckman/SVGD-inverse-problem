using LinearAlgebra
import Interpolations: linear_interpolation
import ForwardDiff: gradient, Dual

function sample(xx,X,U)
  # sample a rectangular grid (X,U) onto irregular points xx[i]
  # xx should be an array (in 1D) or an array of tuples (in ND)
  # X should be an array (in 1D) or a tuple of arrays (in ND)
  interp_linear = linear_interpolation(X,U)
  return [interp_linear(xx[i]) for i = 1:length(xx)]
end

function solve_poisson(c, f, a, b; verbose=false)
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

  # print statements to help debug singularity exceptions
  if verbose
    if eltype(c) <: Dual
      # @show getfield.(c, :value)
      @show eigvals(getfield.(Matrix(A), :value))
    else
      # @show c
      @show eigvals(Matrix(A))
    end
  end

  # solve PDE
  u = A \ vcat(a, f, b)
  
  return u
end

# log posterior
function logp(logc, xd, fd, a, b, x, y, s2, K_prior, mu_prior)
  ud = solve_poisson(exp.(logc), fd, a, b)
  return -norm(y - sample(x, xd, ud))^2 / (2*s2) - dot((logc .- mu_prior), K_prior \ (logc .- mu_prior)) / 2
end

# gradient of log posterior, computed using adjoint method
function adj_grad_logp(logc, xd, fd, a, b, x, y, s2, K_prior, mu_prior)
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
  end
  return grad
end

# gradient of log posterior, computed using automatic differentiation
function grad_logp(logc, xd, fd, a, b, x, y, s2, K_prior, mu_prior)
  return gradient(logc -> logp(logc, xd, fd, a, b, x, y, s2, K_prior, mu_prior), logc)
end
  