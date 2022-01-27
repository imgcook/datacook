import { Rank, Tensor, scalar, variable, Tensor1D, RecursiveArray, Variable, Scalar, sub, add, gather, stack, tidy, dispose } from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-layers';
import * as tf from '@tensorflow/tfjs-core';
import { checkArray } from '../../utils/validation';
import { BaseEstimator } from '../base';
import { getJacobian } from '../../linalg/jacobian';
import { LinearRegressionAnalysis } from '../stat/linear-regression-analysis';

export interface NonLinearRegressionParams {
  initParams: number[] | Tensor1D,
  tol?: number,
  maxIterTimes?: number
}

export class NonLinearRegression extends BaseEstimator {
  public expr: (tf: any, ...args: Tensor<Rank>[]) => Scalar;
  public coeffs: Variable[];
  constructor() {
    super();
  }
  public async fit(expr: string | ((tf: any, features: Tensor<Rank>, ...args: Variable[]) => Scalar),
    x: Tensor | RecursiveArray<number>,
    y: Tensor | RecursiveArray<number>,
    params: NonLinearRegressionParams): Promise<void> {

    if (typeof(expr) != 'string' && typeof(expr) != 'function') {
      throw new TypeError('Invalid input function');
    }
    if (typeof(expr) == 'string') {
      this.expr = eval(expr) as (tf: any, x: Tensor, ...coeffs: Variable[]) => Scalar;
    } else {
      this.expr = expr;
    }
    const xTensor = checkArray(x);
    const yTensor = checkArray(y, 'float32', 1);
    const coeffs: Variable[] = [];
    const { initParams, tol = 1e-4, maxIterTimes = 100 } = params;
    if (!initParams) {
      throw new TypeError('Initial values should be provided');
    }
    this.checkAndSetNFeatures(xTensor, true);
    const initParamsArray = initParams instanceof Tensor ? initParams.arraySync() : initParams;
    for (let i = 0; i < initParamsArray.length; i++) {
      tidy(() => coeffs.push(variable(scalar(initParamsArray[i]))));
    }
    for (let i = 0; i < maxIterTimes; i++) {
      const { jacobian, values } = getJacobian(this.expr, xTensor, ...coeffs);
      const residuals = tidy(() => sub(yTensor, values));
      const lm = new LinearRegressionAnalysis({ fitIntercept: false });
      await lm.fit(jacobian, residuals);
      const dCoeffs = await (lm.coefficients as Tensor1D).array();
      for (let i = 0; i < coeffs.length; i++) {
        if (isNaN(dCoeffs[i])) {
          throw new Error('Function cannot converge');
        }
        tidy(() => coeffs[i].assign(add(coeffs[i], dCoeffs[i])));
      }
      if (Math.max(...dCoeffs.map((d) => Math.abs(d))) < tol) {
        break;
      }
      dispose([ residuals, jacobian, values ]);
    }
    this.coeffs = coeffs;
  }

  public async predict(x: Tensor): Promise<Tensor> {
    return tidy(() => {
      if (!this.expr || !this.coeffs) {
        throw new TypeError('Please train the model first');
      }
      const yStack = [];
      this.checkAndSetNFeatures(x, false);
      for (let i = 0; i < x.shape[0]; i++) {
        const xi = gather(x, i);
        yStack.push(this.expr(tf, xi, ...this.coeffs));
      }
      return stack(yStack);
    });
  }

  public async fromJson(modelJson: string): Promise<void> {
    const modelParams = JSON.parse(modelJson);
    const { exprStr, coeffs, name, nFeature } = modelParams;
    if (name !== 'NonLinearRegression') {
      throw new TypeError(`${name} is not NonLinearRegression`);
    }
    if (exprStr) {
      this.expr = eval(exprStr);
    }
    this.nFeature = nFeature;
    if (coeffs && coeffs?.length) {
      this.coeffs = [];
      coeffs.forEach((coeff: number) => {
        this.coeffs.push(variable(scalar(coeff)));
      });
    }
  }

  public async toJson(): Promise<string> {
    const exprStr = this.expr.toString();
    const coeffs = this.coeffs.map((d) => d.arraySync());
    const modelParams = {
      name: 'NonLinearRegression',
      exprStr,
      coeffs,
      nFeature: this.nFeature
    };
    return JSON.stringify(modelParams);
  }

}
