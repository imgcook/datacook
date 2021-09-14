import { Tensor, mean, matMul, sub, transpose, RecursiveArray, ones, sqrt, add, divNoNan, mul, gather, slice, linalg, diag, eye, log, sum } from "@tensorflow/tfjs-core";
import { svd, tensorEqual } from "../../linalg";
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
  public factorLoadings: Tensor;

  constructor(params: FactorAnalysisParams) {
    this.nComponents = params.nComponent ? params.nComponent : -1;
    this.tol = params.tol ? params.tol : 1e-2;
    this.maxIterTimes = params.maxIterTimes ? params.maxIterTimes : 10000;
  }

  public async fit(xData: Tensor | RecursiveArray<number>): Promise<FactorAnalysis> {
    const xTensor = checkArray(xData, 'float32', 2);
    const [ nSamples, nFeatures ] = xTensor.shape;
    const nComponent = (!this.nComponents || this.nComponents === -1 || this.nComponents > nFeatures) ? nFeatures : this.nComponents;
    const xMeans = mean(xTensor, 0);
    const nSqrt = Math.sqrt(nSamples);
    const xCentered = sub(xTensor, xMeans);
    const sigmaDiag = linalg.bandPart(matMul(transpose(xCentered), xCentered), 0, 0);
    const small = 1e-12;
    let oldll: Tensor;
    let psi = ones([ nFeatures ]);
    for (let i = 0; i < this.maxIterTimes; i++) {
      const sqrtPsi = add(sqrt(psi), small);
      const xd = divNoNan(xCentered, mul(sqrtPsi, nSqrt));
      const [ u, m, w ] = await svd(xd);
      const firstNInd = [ ...Array(nComponent).keys() ];
      const uh = gather(u, firstNInd, 1);
      const mh = gather(mul(m, m), firstNInd);
      const f = matMul(matMul(sqrtPsi, uh), sqrt(sub(diag(mh), eye(nComponent))));
      // update pasi
      psi = sub(sigmaDiag, diag(matMul(transpose(f), f)));
      const llDelta = add(sum(log(psi)), sum(log(mh)));
      if (oldll && tensorEqual(llDelta, oldll, this.tol)) {
        this.factorLoadings = f;
        break;
      }
      if (i == this.maxIterTimes - 1) {
        this.factorLoadings = f;
      }
    }
    return this;
  }
}
