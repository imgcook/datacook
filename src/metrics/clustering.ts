import { matMul, RecursiveArray, sqrt, Tensor, Tensor1D, Tensor2D, transpose, slice, squeeze, logicalNot, slice1d, equal, booleanMaskAsync, unique, min, div, divNoNan, stack, sum, sub, max, mean } from "@tensorflow/tfjs-core";
import { checkArray } from "../utils/validation";
export const getSilhouetteCoefficient = async (xData: Tensor | RecursiveArray<number>, labels: Tensor | RecursiveArray<number>): Promise<number> => {
  const xTensor = checkArray(xData, 'float32', 2) as Tensor2D;
  const yTensor = checkArray(labels, 'any', 1) as Tensor1D;
  const dist = sqrt(matMul(xTensor, transpose(xTensor)));
  const { values } = unique(yTensor);
  const nClasses = values.shape[0];
  const a: Tensor[] = [];
  const b: Tensor[] = [];
  const s: Tensor[] = [];
  for (let i = 0; i < nClasses; i++) {
    const label = slice1d(values, i, 1).dataSync()[0];
    const iFlags = equal(yTensor, label);
    const bi: Tensor[] = [];
    for (let j = 0; j < nClasses; j++) {
      const label = slice1d(values, j, 1).dataSync()[0];
      const jFlags = equal(yTensor, label);
      const nSamples = sum(jFlags).dataSync()[0];
      const distsJ = await booleanMaskAsync(dist, jFlags, 1);
      const distsIJ = await booleanMaskAsync(distsJ, iFlags, 0);
      if (i === j) {
        a.push(divNoNan(sum(distsIJ), nSamples - 1));
      } else {
        bi.push(divNoNan(sum(distsIJ), nSamples));
      }
    }
    b.push(min(stack(bi), 0));
  }
  for (let i = 0; i < nClasses; i++) {
    const deNominator = max(stack([ a[i], b[i] ]), 0);
    s.push(mean(divNoNan(sub(b[i], a[i]), deNominator)));
  }
  return max(stack(s)).dataSync()[0];
  // for (let i = 0; i < nData; i++) {
  //   const disti = squeeze(slice(dists, i));
  //   const yi = slice1d(yTensor, i, 1).dataSync()[0];
  //   const sameFlags = equal(yTensor, yi)
  //   const distSame = await booleanMaskAsync(disti, sameFlags);
  //   const distDiff = await booleanMaskAsync(disti, logicalNot(sameFlags));

  //   // let ai be the mean distance between i and all other data points in the same cluster
  //   const ai =
  // }
};
