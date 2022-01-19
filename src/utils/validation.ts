import { Tensor, Tensor1D, RecursiveArray, cast, tensor, tidy } from '@tensorflow/tfjs-core';

export function checkArray (array: Tensor | RecursiveArray<number>, dtype = 'any', ensureDimension = -1): Tensor {
  return tidy(() => {
    if (array instanceof Array) {
      array = tensor(array);
    }
    if (array instanceof Tensor) {
      const dim = array.rank;
      const arr_dtype = array.dtype;
      if (ensureDimension != -1 && dim != ensureDimension){
        throw new TypeError(`Dimension of input require to be ${ensureDimension} but receive ${dim}`);
      }
      if (dtype === 'string' || dtype === 'float32' || dtype === 'bool' || dtype === 'int32' || dtype === 'complex64'){
        if (arr_dtype != dtype){
          array = cast(array, dtype);
        }
      }
      return array;
    } else {
      throw new TypeError('invalid input');
    }
  });
}

/**
 * Check that if yTrue and yPred are of same length
 * @param yTrue true labels
 * @param yPred predicted labels
 */
export const checkSameLength = (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): [ Tensor1D, Tensor1D, number ] => {
  const yTrueTensor = checkArray(yTrue, 'any', 1);
  const yPredTensor = checkArray(yPred, 'any', 1);
  const yTrueCount = yTrueTensor.shape[0];
  const yPredCount = yPredTensor.shape[0];
  if (yTrueCount != yPredCount) {
    throw new Error('Shape of yTrue should match shape of yPred');
  }
  return [ yTrueTensor as Tensor1D, yPredTensor as Tensor1D, yTrueCount ];
};
