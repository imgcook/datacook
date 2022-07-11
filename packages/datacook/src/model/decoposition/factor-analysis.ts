import { Tensor, mean, matMul, sub, transpose, RecursiveArray, ones, sqrt, add, divNoNan, mul, gather, slice, linalg, diag, eye, log, sum, clipByValue, neg, range, cast, reshape, square } from "@tensorflow/tfjs-core";
import { svd, tensorEqual } from "../../linalg";
import { checkArray } from "../../utils/validation";
import { getVariance } from "../../stat";

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
  public factorLoadings: Tensor;

  constructor(params: FactorAnalysisParams) {
    this.nComponents = params.nComponent ? params.nComponent : -1;
    this.tol = params.tol ? params.tol : 1e-2;
    this.maxIterTimes = params.maxIterTimes ? params.maxIterTimes : 20;
  }

  public async fit(xData: Tensor | RecursiveArray<number>): Promise<FactorAnalysis> {
    const xTensor = checkArray(xData, 'float32', 2);
    const [ nSamples, nFeatures ] = xTensor.shape;
    const nComponent = (!this.nComponents || this.nComponents === -1 || this.nComponents > nFeatures) ? nFeatures : this.nComponents;
    const xMeans = mean(xTensor, 0);
    const nSqrt = Math.sqrt(nSamples);
    const xCentered = sub(xTensor, xMeans);
    const sigma = getVariance(xTensor, 0);
    const small = 1e-12;
    const llConst = nFeatures * Math.log(2.0 * Math.PI) + nComponent;
    let oldllDelta: Tensor;
    let psi: Tensor = ones([ nFeatures ]);

    for (let i = 0; i < this.maxIterTimes; i++) {
      const sqrtPsi = add(sqrt(psi), small);
      const xd = divNoNan(xCentered, mul(sqrtPsi, nSqrt));

      const [ u, m, w ] = await svd(transpose(xd));
      const firstNInd = cast(range(0, nComponent), 'int32');
      const lastNInd = cast(range(nComponent, nFeatures), 'int32');
      const uh = gather(u, firstNInd, 1);
      const ms = mul(m, m);
      const mh = gather(mul(m, m), firstNInd);
      // compute factor loadings
      const f = mul(mul(uh, sqrt(clipByValue(sub(mh, 1), 0, 1e12))), reshape(sqrtPsi, [ -1, 1 ]));
      // update psi
      psi = clipByValue(sub(sigma, sum(square(f), 1)), 0, 1e12);
      const mNorm = lastNInd.shape ? sum(gather(ms, lastNInd)) : 0;
      const llDelta = divNoNan(mul(add(add(sum(log(psi)), sum(log(mh))), mNorm), nSamples), 2.0);

      if (oldllDelta && tensorEqual(llDelta, oldllDelta, this.tol)) {
        this.factorLoadings = f;
        break;
      }
      if (i == this.maxIterTimes - 1) {
        this.factorLoadings = f;
      }
      oldllDelta = llDelta;
    }
    return this;
  }
}
