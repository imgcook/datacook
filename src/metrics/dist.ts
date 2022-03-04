import { Tensor1D } from "@tensorflow/tfjs-core";
import { checkJSArray } from "../utils/validation";

export const getKLDivergence = (p: number[] | Tensor1D, q: number[] | Tensor1D): number => {
  const pArray = checkJSArray(p, 'float32', 1) as number[];
  const qArray = checkJSArray(q, 'float32', 1) as number[];
  if (pArray.length !== qArray.length) {
    throw new TypeError('Input dimensions must be the same');
  }
  let kl = 0;
  const sumP = pArray.reduce((d, i) => d + i);
  const sumQ = qArray.reduce((d, i) => d + i);
  for (let i = 0; i < pArray.length; i++) {
    const pi = pArray[i] / sumP;
    const qi = qArray[i] / sumQ;
    if (pi < 0 || qi < 0) {
      throw new TypeError('Array item should be greater or equal to zero');
    }
    if (pi > 0 && qi > 0) {
      kl += pi * Math.log(pi / qi);
    }
  }
  return kl;
};

export const getJSDivergence = (p: number[] | Tensor1D, q: number[] | Tensor1D): number => {
  const klPQ = getKLDivergence(p, q);
  const klQP = getKLDivergence(q, p);
  return 0.5 * klPQ + 0.5 * klQP;
};
