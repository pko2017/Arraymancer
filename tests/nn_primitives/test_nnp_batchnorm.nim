import ../../src/arraymancer, ../testutils
import unittest

testSuite "BatchNorm 1D primitives":
  let
    input = @[@[0.25, 1.0, 1.0],
              @[1.0,  0.5, 1.0]].toTensor().astype(float32)
    gamma = ones[float32](3, 1)
    beta = zeros[float32](3, 1)
    target = @[@[-0.7071068286895752,  0.7071067094802856, 0.0],
               @[ 0.7071068286895752, -0.7071067094802856, 0.0]].toTensor().astype(float32)

  test "Forward pass from nnp_batchnorm for tensor of shape (2,3)":
    var output = zeros_like[float32](input)
    # result will be written into "output"
    # TODO: add momentum
    batch_norm_forward(input, gamma, beta, output)
    check: mean_absolute_error(output, target) <= float32(1e-8)
  
  # test "Backward pass from nnp_batchnorm for of shape (2,3)":
  #   echo(input)
  #   var gradInput, gradGamma, gradBeta: Tensor[float32]
  #   batch_norm_backward(input, gamma, beta, target, 
  #                       gradInput, gradGamma, gradBeta)
  #   echo(gradInput, gradGamma, gradBeta)

