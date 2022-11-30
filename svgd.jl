function SVGD(k, grad_k, grad_logp, x0, step_size, iter; verbose=false)
  x = deepcopy(x0)
  M = length(x)
  for i=1:iter
    if verbose
      println("iteration: $i\nx: $x\n")
    end
    K   = [k(xi, xj)      for xi in x, xj in x]
    dK  = [grad_k(xi, xj) for xi in x, xj in x]
    dl  = grad_logp.(x)
    phi = (K*dl .+ dK*ones(M)) / M
    x .+= step_size * phi
  end
  return x
end