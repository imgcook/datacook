import { Tensor, cast, DataTypeMap, tensor, RecursiveArray } from '@tensorflow/tfjs-core';

export function checkArray (array: Tensor | RecursiveArray<number>, dtype: keyof DataTypeMap = 'float32', ensureDimension = -1): Tensor {
  if (array instanceof Array) {
    array = tensor(array);
  }
  if (array instanceof Tensor) {
    const dim = array.rank;
    const arr_dtype = array.dtype;
    if (ensureDimension != -1 && dim != ensureDimension){
      throw new TypeError(`Dimension of input require to be ${ensureDimension} but receive ${dim}`);
    }
    if (dtype != arr_dtype){
      array = cast(array, dtype);
    }
    return array;
  } else {
    throw new Error('invalid input');
  }
}
