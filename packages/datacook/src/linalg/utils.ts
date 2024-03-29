import { Tensor, RecursiveArray, norm, div, max, sub, abs, lessEqual, slice, tensor, isNaN, where, tidy, stack, squeeze } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';

/**
 * Normalize tensor by dividing its norm
 * @param vector tensor to be normalized
 * @returns
 */
export const tensorNormalize = (vector: Tensor): Tensor => {
  return div(vector, norm(vector));
};

/**
 * Check if matrix of target dimension
 * @param matrix matrix tensor
 * @param dim target dimension
 */
export const checkDimension = (matrix: Tensor, dim: number): boolean => {
  return matrix.shape.length === dim;
};

/**
 * Check if matrix of target shape
 * @param matrix matrix tensor
 * @param shape target shape
 * @returns
 */
export const checkShape = (matrix: Tensor, shape: number[]): boolean => {
  const mShape = matrix.shape;
  if (mShape.length != shape.length) {
    return false;
  } else {
    for (let i = 0; i < mShape.length; i++) {
      if (shape[i] === -1) continue;
      if (shape[i] !== mShape[i]) return false;
    }
  }
  return true;
};

/**
 * Check that if a tensor is a square matrix
 * @param matrix target matrix
 */
export const isSquareMatrix = (matrix: Tensor): boolean => {
  return tidy(() => checkDimension(matrix, 2) && matrix.shape[0] === matrix.shape[1]);
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
  }
  for (let i = 0; i < shape1.length; i++) {
    if (shape1[i] !== shape2[i]) {
      return false;
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
  return tidy(() => {
    if (!shapeEqual(tensor1, tensor2)) {
      throw new TypeError('tensor1 and tensor2 not of same shape');
    }
    return Boolean(lessEqual(max(abs(sub(tensor1, tensor2))), tol).dataSync()[0]);
  });
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

/**
 * Fill the nan data with given value
 * TODO: support multi-dimensional data
 * @param xData input data
 * @param fillV value to replace NaN, should be number
 * @return replaced data
 */
export const fillNaN = (xData: Tensor | RecursiveArray<number>, fillV = 0): Tensor => {
  return tidy(() => {
    const xTensor = checkArray(xData, 'float32', 2);
    const cond = isNaN(xTensor);
    return where(cond, fillV, xTensor);
  });
};

/**
 * Get the digonal elements in a matrix
 * @param matrix target matrix
 * @returns tensor of diagnal elements
 */
export const getDiagElements = (matrix: Tensor | number[]): Tensor => {
  return tidy(() => {
    const matrixTensor = checkArray(matrix, 'any', 2);
    const [ m, n ] = matrixTensor.shape;
    const rank = m > n ? n : m;
    const diagElements = [];
    for (let i = 0; i < rank; i++) {
      diagElements.push(slice(matrixTensor, [ i, i ], [ 1, 1 ]));
    }
    return squeeze(stack(diagElements));
  });
};
