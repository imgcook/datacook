import { Tensor, RecursiveArray, slice, concat, ones, Tensor1D, tidy, Tensor2D, tensor } from '@tensorflow/tfjs-core';
import { BaseRegressor } from '../base';
import { arrayMatmul2D, arrayTranpose2D, inverseArray } from '../../linalg';
import { getResidualVariance, getRSquare, getAdjustedRSquare, getAICLM } from '../../metrics/regression';
import { cdf } from '../../stat/t';

/**
 * Parameters for linear regression
 */
export interface LinearRegressionParams {
  /**
   * Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
   */
  fitIntercept?: boolean,
  /**
   * This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
   */
  normalize?: boolean,
}


export type SignificanceCode = '***' | '**' | '*' | '.' | ' ';

export interface CoefficientSummary {
  name: string,
  estimate: number,
  standardError: number,
  tValue: number,
  pValue: number,
  significance: SignificanceCode
}

/**
 * Get significance code according to p value
 * @param pValue p-value
 */
export const getSignificanceCode = (pValue: number): SignificanceCode => {
  if (pValue < 0.001) {
    return '***';
  }
  if (pValue < 0.01) {
    return '**';
  }
  if (pValue < 0.05) {
    return '*';
  }
  if (pValue < 0.1) {
    return '.';
  }
  return ' ';
};

/**
 * Linear regression model
 * LinearRegression fits a linear model with coefficients w = (w1, …, wp)
 * to minimize the residual sum of squares between the observed targets in
 * the dataset, and the targets predicted by the linear approximation.
 */
export class LinearRegressionAnalysis extends BaseRegressor {
  public fitIntercept: boolean;
  public normalize: boolean;
  private featureSize: number;
  public coefficients: Tensor;
  public varianceBase: Tensor;
  public nData: number;
  public residualVariance: number;
  public rSquare: number;
  public adjustedRSquare: number;
  public aic: number;
  public featureNames: string[];
  /**
   * Construction function of linear regression model
   * @param params LinearRegressionParams
   *
   * Options in params
   * ------------
   *
   * `fitIntercept`: Whether to calculate the intercept for this model. If set to False,
   * no intercept will be used in calculations
   *
   * `normalize`: This parameter is ignored when fit_intercept is set to False. If True,
   * the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
   *
   */
  constructor(params: LinearRegressionParams = {}) {
    super();
    this.fitIntercept = params.fitIntercept !== false;
    this.normalize = params.normalize;
  }

  /** Fit linear regression model according to X, y. Here we use ordinary least square model for paramter estimation.
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target values
   * @param featureNames features names, optional
   * @returns classifier itself
   */
  public async fit(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>, featureNames?: string[]): Promise<LinearRegressionAnalysis> {
    const { x, y } = this.validateData(xData, yData);
    const nFeature = x.shape[1];
    const nData = x.shape[0];
    if (featureNames && featureNames instanceof Array) {
      if (featureNames.length != nFeature) {
        throw new TypeError('Size of feature names should match the size of input features');
      }
    }
    const xTensor = this.fitIntercept ? tidy(() => concat([ ones([ nData, 1 ]), x ], 1)) : x;
    const yTensor = y as Tensor1D;
    const xArray = await (xTensor as Tensor2D).array();
    const yArray = await yTensor.array();
    const varianceBase = await inverseArray(arrayMatmul2D(arrayTranpose2D(xArray), xArray));
    // console.log(varianceBase);
    //matMul(transpose(xTensor), xTensor).print();
    const coefs = arrayMatmul2D(arrayMatmul2D(varianceBase, arrayTranpose2D(xArray)), yArray.map((d) => [ d ])).map((d) => d[0]);
    //const coefs = squeeze(matMul(matMul(varianceBase, transpose(xTensor)), reshape(yTensor, [ -1, 1 ])));
    // const predY = tidy(() => squeeze(matMul(xTensor, reshape(coefs, [ -1, 1 ]))) as Tensor1D);
    const predY = arrayMatmul2D(xArray, coefs.map((d) => [ d ])).map((d) => d[0]);
    // console.log(predY.dataSync());
    // console.log(yTensor.dataSync());
    const residualVariance = getResidualVariance(yTensor, predY, nFeature + 1);
    this.varianceBase = tensor(varianceBase);
    this.coefficients = tensor(coefs);
    this.featureSize = nFeature;
    this.residualVariance = residualVariance;
    this.nData = nData;
    this.rSquare = getRSquare(yTensor, predY);
    this.adjustedRSquare = getAdjustedRSquare(yTensor, predY, nFeature);
    this.aic = getAICLM(yTensor, predY, nFeature);
    this.featureNames = featureNames;
    // dispose([ xTensor, yTensor ]);
    // if (!(xData instanceof Tensor)) {
    //   dispose(x);
    // }
    // if (!(yData instanceof Tensor)) {
    //   dispose(y);
    // }
    // dispose(coefs);
    return this;
  }

  /**
   * Print summary of model
   */
  public printSummary(): void {
    const summary = this.summary();
    console.log('coefficients:');
    console.table(summary.coefficients);
    console.log('r-square:', summary.rSquare);
    console.log('Adjusted r-square:', summary.adjustedRSquare);
    console.log('Residual Standard Error', summary.residualStandardError);
    console.log('AIC', summary.aic);
  }

  /**
   * Get summary of linear regression results.
   * @returns
   */
  public summary(): {
    coefficients: Array<CoefficientSummary>,
    rSquare: number,
    adjustedRSquare: number,
    residualStandardError: number,
    residualDegreeOfFreedom: number,
    aic: number,
    } {
    const df = this.nData - this.featureSize - 1;
    const coefficients: Array<CoefficientSummary> = [];
    const featureSize = this.fitIntercept ? this.featureSize + 1 : this.featureSize;
    for (let i = 0; i < featureSize; i++) {
      const estimate = tidy(() => slice(this.coefficients, i, 1).dataSync()[0]);
      const traceI = tidy(() => slice(this.varianceBase, [ i, i ], [ 1, 1 ]).dataSync()[0]);
      const standardError = Math.sqrt(this.residualVariance * traceI);
      const tValue = estimate / standardError;
      const cdfT = cdf(tValue, df);
      const pValue = cdfT > 0.5 ? ((1 - cdfT) * 2) : (cdfT * 2);
      const name = this.fitIntercept ?
        ((i === 0) ? 'Intercept' : this.featureNames ? this.featureNames[i - 1] : `coeff_${i}`) :
        ( this.featureNames ? this.featureNames[i] : `coeff_${i + 1}` );
      const significance = getSignificanceCode(pValue);
      coefficients.push({ name, estimate, standardError, tValue, pValue, significance });
    }
    return {
      coefficients,
      rSquare: this.rSquare,
      adjustedRSquare: this.adjustedRSquare,
      residualStandardError: Math.sqrt(this.residualVariance),
      residualDegreeOfFreedom: this.nData - this.featureSize - 1,
      aic: this.aic
    };
  }
}
