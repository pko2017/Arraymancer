# Steps and reference python code taken from: 
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# def batch_norm(input, running_mean, running_var, weight=None, bias=None,
#                training=False, momentum=0.1, eps=1e-5):

import  ../../tensor,
        #../../nn_primitives,
        ../../autograd
import math, sugar


type BatchNormGate*[TT] {.final.} = ref object of Gate[TT]
  input, gamma, beta: Variable[TT]


proc batch_norm_backward_ag[TT](self: BatchNormGate[TT], payload: Payload[TT]) : SmallDiffs[TT] =
  # result[0] grad w.r.t. input
  # result[1] grad w.r.t. gamma
  # result[2] grad w.r.t. beta

  let
    dout = payload.variable.grad
    N = self.input.value.shape[0]
    D = self.input.value.shape[1]
    eps = 1e-5'f32

    # FIXME : use cache values to speed up ?
    # FIXME : remove intermediate variables
    mu = self.input.value.mean(axis=0)
    var_x = self.input.value.variance(axis=0)
    xmu = self.input.value .- mu
    xeps = var_x.map(x => x+eps)
    sqrtvar = xeps.map(x => sqrt(x))
    ivar = sqrtvar.map(x => 1.0'f32/x)
    xhat = xmu .* ivar

  let 
    dbeta = dout.sum(axis=0)
    # 8 
    dgamma_1 = dout .* xhat
    dgamma = dgamma1.sum(axis=0)
    dxhat = dout .* self.gamma.value.transpose
    # 7
    divar_1 = dxhat .* xmu
    divar = divar_1.sum(axis=0)
    dxmu1 = dxhat .* ivar
    # 6
    sqrtvar_pow2 = sqrtvar.map(x => x*x)
    dsqrtvar = sqrtvar_pow2.map(x => -1'f32/x) .* divar
    # 5
    dvar_1 = sqrtvar.map(x => 0.5'f32/x) 
    dvar = dvar_1 .* dsqrtvar
    # 4
    dsq = (ones[float32](N, D) .* dvar).map(x => x/float32(N))
    # 3
    dxmu2 = (xmu .* dsq).map(x => x*2.0'f32)
    # 2
    dx1 = dxmu1 + dxmu2
    dmu = dx1.sum(axis=0).map(x => x*(-1.0'f32))
    # 1
    dx2 = ones[float32](N, D).map(x => 1/float32(N)) .* dmu
    # 0
    dx = dx1 + dx2

  # results
  result = newDiffs[TT](3)

  if self.input.requires_grad:
    # FIXME: I gues in nets where BN is first layer (my TestNet for tests)
    # the input.requires_grad is false and therefore result[0] is empty
    result[0] = dx

  # FIXME: assuming self.input.requires_grad is true (for tests only)
  result[0] = dx

  if self.gamma.requires_grad:
    # result[1] = zeros[float32](D,1)
    result[1] = dgamma

  if self.beta.requires_grad:
    # result[2] = zeros[float32](D,1)
    result[2] = dbeta

  # echo("-----------batch_norm_backward_ag[TT]--------------")
  # echo("batch_norm_backward_ag: dout: " & $dout)
  # echo("batch_norm_backward_ag: N : " & $N )
  # echo("batch_norm_backward_ag: D: " & $D )
  # echo("batch_norm_backward_ag: self.gamma: " & $self.gamma.value )
  # echo("batch_norm_backward_ag: dbeta: "& $dbeta)
  # echo("batch_norm_backward_ag: self.input: "& $self.input.value)
  # echo("batch_norm_backward_ag: sqrtvar: "& $sqrtvar)
  # echo("batch_norm_backward_ag: ivar: "& $ivar)
  # echo("batch_norm_backward_ag: xmu: "& $xmu)
  # echo("batch_norm_backward_ag: xhat: "& $xhat)

  # echo("step 8 : dxhat: " & $dxhat)
  # echo("step 7 : dxmu1: " & $dxmu1)
  # echo("step 7 : divar: " & $divar)
  # echo("step 6 : divar: " & $divar)
  # echo("step 6 : sqrtvar_pow2: " & $sqrtvar_pow2)
  # echo("step 6 : dsqrtvar: " & $dsqrtvar) 
  # echo("step 5 : dvar: " & $dvar)
  # echo("step 4 : dsq: " & $dsq)
  # echo("step 3 : dxmu2: " & $dxmu2)
  # echo("step 2 : dx1: " & $dx1)
  # echo("step 1 : dx2: " & $dx2)

  echo("result: " & $result)
  echo("-----------batch_norm_backward_ag[TT]--------------")


proc batch_norm_cache[TT](result: Variable[TT], input, gamma, beta: Variable[TT]) =
  var gate: BatchNormGate[TT]
  new gate
  gate.input = input
  gate.gamma = gamma
  gate.beta = beta

  result.grad = zeros_like(result.value)
  result.requires_grad = true

  register_node(
      "BatchNorm1D",
      gate,
      batch_norm_backward_ag[TT],
      result,
      input, gamma, beta # Parents (varargs)
    )

proc batch_norm*[TT](input, gamma, beta: Variable[TT]): Variable[TT] =
  ## Input:
  ## input Variable of shape [batch_size, in_features]
  ## gamma Variable of shape [1, in_features] initialized at 1.0
  ## beta  Variable of shape [1, in_features] initialized at 0.0
  ## Returns (gamma*input_normalized)+beta
  
  let 
    mu = input.value.mean(axis=0)
    var_x = input.value.variance(axis=0)
    eps = 1e-5'f32
    xmu = input.value .- mu
    var_x_eps = var_x.map(x => x+eps)
    sqrtvar = var_x_eps.map(x => sqrt(x))
    ivar = sqrtvar.map(x => 1.0'f32/x)
    xhat = xmu .* ivar
    xhatgamma = xhat .* gamma.value.transpose
    xhatgammabeta = xhatgamma .+ beta.value.transpose

  # echo("batch_norm: input.value.shape: " & $input.value.shape)
  # echo("batch_norm: input.value" & $input.value)
  # echo("batch_norm \n  mean:" & $mu & " batch var:" & $var_x)
  # echo("batch_norm \n  sqrtvar:" & $sqrtvar )
  # echo("batch_norm \n  ivar:" & $ivar )
  # echo("batch_norm: xhat " & $xhat)
  # echo("batch_norm: \n gamma:" & $gamma.value & " beta:" & $beta.value)
  # echo("batch_norm: \n gamma.shape:" & $gamma.value.shape & " beta.shape:" & $beta.value.shape)
  # echo("batch_norm: xhat.shape " & $xhat.shape)
  # echo("batch_norm: result.value: " & $xhatgammabeta)

  # Resulting var
  new result
  result.value = xhatgammabeta
  result.context = input.context

  # Caching for backprop
  result.batch_norm_cache(input, gamma, beta)