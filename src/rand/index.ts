import { add, mul, sub, tensor, Tensor } from '@tensorflow/tfjs-core';
import { lgamma } from 'tfjs-special';
import { sizeFromShape } from '../utils';

export function randoms(shape: number[]): Tensor {
  let size = -1;
  if (Array.isArray(shape)) {
    size = sizeFromShape(shape);
  } else {
    size = shape;
  }
  const arr = new Float32Array(size);
  for(let i = 0; i < arr.length; i++) {
    arr[i] = Math.random();
  }
  return tensor(arr, shape);
}

export function beta(a: number | number[], b: number | number[], shape: number[]=[1]): Tensor {
  const aTensor = tensor(a, shape);
  const bTensor = tensor(b, shape);
  const Ga = lgamma(aTensor);
  const Gb = lgamma(bTensor);
  const Gab = lgamma(add(aTensor, bTensor));
  const beta = sub(sub(Gab, Ga),Gb);

  const randomTensor = randoms(shape);
  const aPart = mul(sub(aTensor, 1), randomTensor.log());
  const bPart = mul((sub(bTensor, 1)), sub(1, randomTensor).log());
  return add(beta, add(aPart, bPart)).exp();
}
