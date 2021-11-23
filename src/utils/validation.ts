<<<<<<< HEAD
import { Tensor, cast, tensor, RecursiveArray } from '@tensorflow/tfjs-core';

export function checkArray (array: Tensor | RecursiveArray<number>, dtype = 'any', ensureDimension = -1): Tensor {
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
=======
import { Tensor, RecursiveArray, cast, tensor, tidy } from '@tensorflow/tfjs-core';

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
>>>>>>> a204a8a299a83931037e6addd72fe7816251b3e4
    }
  });
}
