import ../tensor,
       math,
       sugar

proc batch_norm_backward*[T](input, 
                             gamma, 
                             beta, 
                             gradOutput: Tensor[T],
                             gradInput,
                             gradGamma,
                             gradBeta: var Tensor[T]) {.inline.} =
  # TODO : docs

  let
    dout = gradOutput
    N = input.shape[0]
    D = input.shape[1]
    eps = T(1e-8)

    # FIXME : use cache values to speed up ?
    mu = input.mean(axis=0)
    var_x = input.variance(axis=0)
    xmu = input .- mu
    xeps = var_x.map(x => x+eps)
    sqrtvar = xeps.map(x => sqrt(x))
    ivar = sqrtvar.map(x => T(1.0)/x)
    xhat = xmu .* ivar

    # FIXME : remove intermediate variables
    dbeta = dout.sum(axis=0)
    dgamma_1 = dout .* xhat
    dgamma = dgamma1.sum(axis=0)
    dxhat = dout .* gamma.transpose
    divar_1 = dxhat .* xmu
    divar = divar_1.sum(axis=0)
    dxmu1 = dxhat .* ivar
    sqrtvar_pow2 = sqrtvar.map(x => x*x)
    dsqrtvar = sqrtvar_pow2.map(x => T(-1.0)/x) .* divar
    dvar_1 = sqrtvar.map(x => T(0.5)/x)
    dvar = dvar_1 .* dsqrtvar
    dsq = (ones[T](N, D) .* dvar).map(x => x/T(N))
    dxmu2 = (xmu .* dsq).map(x => x*T(2.0))
    dx1 = dxmu1 + dxmu2
    dmu = dx1.sum(axis=0).map(x => x*(T(-1.0)))
    dx2 = ones[T](N, D).map(x => 1/T(N)) .* dmu
    dx = dx1 + dx2
  
  gradInput = dx
  gradGamma = dgamma
  gradBeta = dbeta
  

proc batch_norm*[T](input, gamma, beta: Tensor[T], output: var Tensor[T]) {.inline.} =
  # TODO : docs

  let 
    mu = input.mean(axis=0)
    var_x = input.variance(axis=0)
    eps = T(1e-8)
    xmu = input .- mu
    var_x_eps = var_x.map(x => x+eps)
    sqrtvar = var_x_eps.map(x => sqrt(x))
    ivar = sqrtvar.map(x => T(1.0)/x)
    xhat = xmu .* ivar
  
  output = (xhat .* gamma.transpose) .+ beta.transpose

