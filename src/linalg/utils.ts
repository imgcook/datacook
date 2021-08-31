import { Tensor, norm, div, max, sub, abs, lessEqual, slice, tensor } from "@tensorflow/tfjs-core";

/**
 * Normalize tensor by dividing its norm
 * @param tensor
 * @returns
 */
export const tensorNormalize = (tensor: Tensor): Tensor => {
  return div(tensor, norm(tensor));
};

/**
 * Check that if two tensor are of same shape
 * @param tensor1
 * @param tensor2
 * @returns
 */
export const shapeEqual = (tensor1: Tensor, tensor2: Tensor): boolean => {
  const shape1 = tensor1.shape;
  const shape2 = tensor2.shape;
  if (shape1.length != shape2.length) {
    return false;
  } else {
    for (let i = 0; i < shape1.length; i++) {
      if (shape1[i] != shape2[i]) {
        return false;
      }
    }
  }
  return true;
};

/**
 * Check that two tensors are equal to within some additive tolerance.
 * @param tensor1
 * @param tensor2
 * @param
 */
export const tensorEqual = (tensor1: Tensor, tensor2: Tensor, tol = 0): boolean => {
  if (!shapeEqual(tensor1, tensor2)) {
    throw new Error('tensor1 and tensor2 not of same shape');
  }
  const isEqual = lessEqual(max(abs(sub(tensor1, tensor2))), tol).dataSync();
  if (isEqual) {
    return true;
  } else {
    return false;
  }
};

/**
 * Fill a diag matrix with shape m and n
 * @param values diag values
 * @param m number of rows
 * @param n number of columns
 */
export const fillDiag = (values: Tensor | number[], m: number, n: number): Tensor => {
  const svSize = m > n ? n : m;
  const valuesData = values instanceof Tensor ? values.dataSync() : values;
  const vSize = valuesData.length;
  const diagData = new Array(m).fill(0).map( () => new Array(n).fill(0));
  for (let i = 0; i < svSize; i++) {
    if (i < vSize) {
      diagData[i][i] = Number(slice(valuesData, i, 1).dataSync());
    }
  }
  return tensor(diagData);
};
