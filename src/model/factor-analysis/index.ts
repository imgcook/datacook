import { Tensor, mean, matMul, sub, transpose, RecursiveArray, ones, sqrt, add, divNoNan } from "@tensorflow/tfjs-core";
import { setFlagsFromString } from "node:v8";
import { svd } from "../../linalg";
import { checkArray } from "../../utils/validation";

export type FactorAnalysisParams = {
  nComponent: number,
  tol?: number,
  maxIterTimes?: number
}

/**
 * Factor Analysis.
 *
 * Factor analysis is a decomposition algorithm which similar to probabilistic PCA.
 * In factor analysis, visible vector variable v is related to the vector hidden
 * variable h by a linear mapping, with independent additive Gaussian noise on each
 * visible variable.
 *
 * The implementation of factor analysis uses an EM-based method to find the best
 * solution for factor loading.
 */
export class FactorAnalysis {
  public nComponents: number;
  public tol: number;
  public maxIterTimes: number;

  constructor(params: FactorAnalysisParams) {
    this.nComponents = params.nComponent ? params.nComponent : -1;
    this.tol = params.tol ? params.tol : 1e-2;
    this.maxIterTimes = params.maxIterTimes ? params.maxIterTimes : 10000;
  }

  public fit(xData: Tensor | RecursiveArray<number>): void {
    const xTensor = checkArray(xData, 'float32', 2);
    const [ nSamples, nFeatures ] = xTensor.shape;
    const nComponent = (!this.nComponents || this.nComponents === -1 || this.nComponents > nFeatures) ? nFeatures : this.nComponents;
    const xMeans = mean(xTensor, 0);
    const nSqrt = Math.sqrt(nSamples);
    const xCentered = sub(xTensor, xMeans);
    const s = matMul(transpose(xCentered), xCentered);
    const small = 1e-12;
    let psi = ones([ nFeatures ]);
    /*for (let i = 0; i < this.maxIterTimes; i++) {
      const sqrtPsi = add(sqrt(psi), small);
      const [ u, v, m ] = svd(divNoNan(xCentered, ))
    }
    */
  }
}
