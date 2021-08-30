import { Tensor, norm, div, max, sub, abs, lessEqual } from "@tensorflow/tfjs-core";

export const tensorNormalize = (tensor: Tensor): Tensor => {
  return div(tensor, norm(tensor));
};

/**
 * Check that two tensors are equal to within some additive tolerance.
 * @param tensor1
 * @param tensor2
 * @param
 */
export const tensorEqual = (tensor1: Tensor, tensor2: Tensor, tol = 0): boolean => {
  const isEqual = lessEqual(max(abs(sub(tensor1, tensor2))), tol).dataSync();
  if (isEqual) {
    return true;
  } else {
    return false;
  }
};


