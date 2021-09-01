import { Tensor, matMul, transpose, tensor, slice, squeeze, linalg, reshape } from '@tensorflow/tfjs-core';

/**
 * Solve a linear equation Rx = v, where R is an upper triangular matrix.
 * This type of equation is easy to solve by back substitution.  We work *up*
 * the rows of R solving for the components of x backwards.  For example, the
 * final row in R gives the equation
 *   r_{l,l} x_l = v_l
 * whose solution is simply x_l = v_l / r_{l,l}.  The second to final row gives
 * the equation
 *   r_(l-1,l-1} x_{l-1} + r_{l-1,l} x_{l} = v_l
 * which can be solved by substituting in the value of x_l already found, and
 * then solving the resulting equation for x_{l-1}.  Continuing in this way
 * solves the entire system.
 * @param r upper triangle matrix
 * @param v target value
 */
export const linSolveUpperTriangle = (r: Tensor, v: Tensor): Tensor => {
  const nEq = v.shape[0];
  const solution = new Array(nEq).fill(0);
  let backSubsitute: number;
  for (let i = nEq - 1; i >= 0; i--) {
    backSubsitute = 0;
    if (i < nEq - 1) {
      for (let j = i + 1; j <= nEq - 1; j++) {
        const l = solution[j];
        const w = Number(slice(r, [ i, j ], [ 1, 1 ]).dataSync());
        backSubsitute += l * w;
      }
      const vi = Number(slice(v, [ i ], [ 1 ]).dataSync());
      const ri = Number(slice(r, [ i, i ], [ 1, 1 ]).dataSync());
      solution[i] = (vi - backSubsitute) * 1.0 / ri;
    } else {
      const vi = Number(slice(v, [ i ], [ 1 ]).dataSync());
      const ri = Number(slice(r, [ i, i ], [ 1, 1 ]).dataSync());
      solution[i] = vi * 1.0 / ri;
    }
  }
  return tensor(solution);
};

/* Solve a general linear equation Mx = v using the QR decomposition of M.
 * If QR is the decomposion of M, then using the fact that Q is orthogonal
 *    QRx = v => Rx = transpose(Q)v
 * The matrix product transpose(Q)v is easy to compute, so the decomposition
 * reduces the problem to solving a linear equation Rx = y for an upper
 * triangular matrix R.
*/
export const linSolveFromQR = (q: Tensor, r: Tensor, v: Tensor): Tensor => {
  const rhs = squeeze(matMul(transpose(q), reshape(v, [ -1, 1 ])));
  const solution = linSolveUpperTriangle(r, rhs);
  return solution;
};

export const linSolveQR = (matrix: Tensor, v: Tensor): Tensor => {
  const [ q, r ] = linalg.qr(matrix);
  const solution = linSolveFromQR(q, r, v);
  return solution;
};
