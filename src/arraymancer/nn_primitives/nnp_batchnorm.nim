import ../tensor,
       math,
       sugar

proc batch_norm_backward*[T](input, gamma, beta, gradOutput: Tensor[T],
                             gradInput, gradGamma, gradBeta: var Tensor[T]) {.inline.} =
  # TODO : docs
  let
    batchSize = input.shape[0]
    eps = T(1e-8)
    # TODO : use cache values [xmu, sqrtvar, ivar, xhat] to speed up ?
    xmu = input .- input.mean(axis=0)
    sqrtvar = input.variance(axis=0).map(x => x+eps).map(x => sqrt(x))
    ivar = sqrtvar.map(x => T(1.0)/x)
    xhat = xmu .* ivar
    dxhat = gradOutput .* gamma.transpose
    divar = (dxhat .* xmu).sum(axis=0)
    dxmu1 = dxhat .* ivar
    dsqrtvar = (sqrtvar.map(x => x*x)).map(x => T(-1.0)/x) .* divar
    dvar = sqrtvar.map(x => T(0.5)/x) .* dsqrtvar
    dsq = (ones_like[T](input) .* dvar).map(x => x/T(batchSize))
    dxmu2 = (xmu .* dsq).map(x => x*T(2.0))
    dx1 = dxmu1 + dxmu2
    dmu = dx1.sum(axis=0).map(x => x*(T(-1.0)))
    dx2 = ones_like[T](input).map(x => 1/T(batchSize)) .* dmu
  
  gradInput = dx1 + dx2
  gradGamma = (gradOutput .* xhat).sum(axis=0)
  gradBeta = gradOutput.sum(axis=0)
  

proc batch_norm_forward*[T](input, gamma, beta: Tensor[T], 
                            output: var Tensor[T]) {.inline.} =
  # TODO : docs

  let
    eps = T(1e-8)
    xmu = input .- input.mean(axis=0)
    sqrtvar = input.variance(axis=0).map(x => x+eps).map(x => sqrt(x))
    ivar = sqrtvar.map(x => T(1.0)/x)
    xhat = xmu .* ivar
  
  output = (xhat .* gamma.transpose) .+ beta.transpose


# proc batch_norm_inference*[T](input, gamma, beta: Tensor[T], 
#                               output: var Tensor[T]) {.inline.} =
#   # TODO : docs
#   
#   let 
#     mu = input.mean(axis=0) # TODO find the way to access saved mean()
#     var_x = input.variance(axis=0) # TODO find the way to access saved var()
#     eps = T(1e-8)
#     xmu = input .- mu
#     var_x_eps = var_x.map(x => x+eps)
#     sqrtvar = var_x_eps.map(x => sqrt(x))
#     ivar = sqrtvar.map(x => T(1.0)/x)
#     xhat = xmu .* ivar
  
#   output = (xhat .* gamma.transpose) .+ beta.transpose

