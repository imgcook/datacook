// import { Scalar, Tensor, Variable, variableGrads, gather, stack, tidy } from '@tensorflow/tfjs-core';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-layers';
const { variableGrads, gather, stack, tidy } = tf;
/**
 * Solve jacobian matrix for a given expression `expr`, input tensor `x` and varaibles
 * jac[i, j] = d(expr(x[i], var[j])) / d(var[j])
 * @param expr expression function for x[i] and varaibles, return value should be a scalar
 * (expr: (tf: any, x: Tensor, ...coeffs: Variable[]) => Scalar
 * @param x input tensor of shape [n, ....]
 * @param coeffs array of coefficients
 * @returns jacobian
 */
export const getJacobian = (expr: (tf: any, x: tf.Tensor, ...coeffs: tf.Variable[]) => tf.Scalar,
  x: tf.Tensor, ...coeffs: tf.Variable[]): { values: tf.Tensor, jacobian: tf.Tensor} => {
  return tidy(() => {
    const n = x.shape[0];
    const jacStack = [];
    const preds = [];
    for (let i = 0; i < n; i++) {
      const xi = gather(x, i);
      const { value, grads } = variableGrads(() => expr(tf, xi, ...coeffs));
      const gradsI: tf.Tensor[] = [];

      Object.keys(grads).forEach((varName) => {
        gradsI.push(grads[varName]);
      });
      preds.push(value);
      jacStack.push(stack(gradsI));
    }
    return { values: stack(preds), jacobian: stack(jacStack) };
  });
};
