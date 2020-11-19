# Steps and reference python code taken from: 
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# def batch_norm(input, running_mean, running_var, weight=None, bias=None,
#                training=False, momentum=0.1, eps=1e-5):

import  ../../tensor,
        ../../nn_primitives,
        ../../autograd
# import math, sugar


type BatchNormGate*[TT] {.final.} = ref object of Gate[TT]
  input, gamma, beta: Variable[TT]


proc batch_norm_backward_ag[TT](self: BatchNormGate[TT], payload: Payload[TT]) : SmallDiffs[TT] =
  # result[0] grad w.r.t. input
  # result[1] grad w.r.t. gamma
  # result[2] grad w.r.t. beta

  var gradInput, gradGamma, gradBeta: Tensor[TT.T]

  batch_norm_backward(self.input.value, 
                      self.gamma.value,
                      self.beta.value,
                      payload.variable.grad,
                      gradInput,
                      gradGamma,
                      gradBeta)

  result = newDiffs[TT](3)

  if self.input.requires_grad:
    # FIXME: I guess in nets where BN is first layer (my TestNet for tests)
    # the input.requires_grad is set to false and therefore result[0] is empty
    result[0] = gradInput

  # FIXME: assuming self.input.requires_grad is true
  # anyways BN shouldn't be the 1st layer in real nets
  result[0] = gradInput

  if self.gamma.requires_grad:
    result[1] = gradGamma

  if self.beta.requires_grad:
    result[2] = gradBeta



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

  new result
  result.context = input.context

  if input.is_grad_needed or gamma.is_grad_needed or beta.is_grad_needed:
    batch_norm_forward(input.value, gamma.value, beta.value, result.value)
    # Caching for backprop
    result.batch_norm_cache(input, gamma, beta)
  # else:
  #   batch_norm_inference(input.value, gamma.value, beta.value, result.value)

  
