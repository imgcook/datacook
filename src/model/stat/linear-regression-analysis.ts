import { Tensor, RecursiveArray, squeeze, transpose, matMul, slice, concat, ones, Tensor1D, reshape, tidy, dispose } from '@tensorflow/tfjs-core';
import { BaseRegressor } from '../base';
import { inverse } from '../../linalg';
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
  normalize?: boolean
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
   * @returns classifier itself
   */
  public async fit(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>): Promise<LinearRegressionAnalysis> {
    const { x, y } = this.validateData(xData, yData);
    const nFeature = x.shape[1];
    const nData = x.shape[0];
    const xTensor = tidy(() => concat([ ones([ nData, 1 ]), x ], 1));
    const yTensor = y as Tensor1D;
    const varianceBase = await inverse(tidy(() => matMul(transpose(xTensor), xTensor)), 1e-4, 200, true);
    squeeze(matMul(matMul(varianceBase, transpose(xTensor)), reshape(yTensor, [ -1, 1 ]))).print();
    const coefs = squeeze(matMul(matMul(varianceBase, transpose(xTensor)), reshape(yTensor, [ -1, 1 ])));
    const predY = tidy(() => squeeze(matMul(xTensor, reshape(coefs, [ -1, 1 ]))) as Tensor1D);
    const residualVariance = getResidualVariance(yTensor, predY, nFeature + 1);
    this.varianceBase = varianceBase;
    this.coefficients = coefs;
    this.featureSize = nFeature;
    this.residualVariance = residualVariance;
    this.nData = nData;
    this.rSquare = getRSquare(yTensor, predY);
    this.adjustedRSquare = getAdjustedRSquare(yTensor, predY, nFeature);
    this.aic = getAICLM(yTensor, predY, nFeature);
    dispose([ xTensor, yTensor, predY ]);
    if (!(xData instanceof Tensor)) {
      dispose(x);
    }
    if (!(yData instanceof Tensor)) {
      dispose(y);
    }
    dispose(coefs);
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
    for (let i = 0; i < this.featureSize + 1; i++) {
      const estimate = tidy(() => slice(this.coefficients, i, 1).dataSync()[0]);
      const traceI = tidy(() => slice(this.varianceBase, [ i, i ], [ 1, 1 ]).dataSync()[0]);
      const standardError = Math.sqrt(this.residualVariance * traceI);
      const tValue = estimate / standardError;
      const cdfT = cdf(tValue, df);
      const pValue = cdfT > 0.5 ? ((1 - cdfT) * 2) : (cdfT * 2);
      const name = (i === 0) ? 'Intercept' : `coeff_${i}`;
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
