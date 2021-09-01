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
export const linSolveUpperTriangle = async(r: Tensor, v: Tensor): Promise<Tensor> => {
  const nEq = v.shape[0];
  const solution = new Array(nEq).fill(0);
  let backSubsitute: number;
  for (let i = nEq - 1; i >= 0; i--) {
    backSubsitute = 0;
    if (i < nEq - 1) {
      for (let j = i + 1; j <= nEq - 1; j++) {
        const l = solution[j];
        const w = Number(await slice(r, [ i, j ], [ 1, 1 ]).data());
        backSubsitute += l * w;
      }
      const vi = Number(await slice(v, [ i ], [ 1 ]).data());
      const ri = Number(await slice(r, [ i, i ], [ 1, 1 ]).data());
      solution[i] = (vi - backSubsitute) * 1.0 / ri;
    } else {
      const vi = Number(await slice(v, [ i ], [ 1 ]).data());
      const ri = Number(await slice(r, [ i, i ], [ 1, 1 ]).data());
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
export const linSolveFromQR = async(q: Tensor, r: Tensor, v: Tensor): Promise<Tensor> => {
  const rhs = squeeze(matMul(transpose(q), reshape(v, [ -1, 1 ])));
  const solution = await linSolveUpperTriangle(r, rhs);
  return solution;
};

export const linSolveQR = async(matrix: Tensor, v: Tensor): Promise<Tensor> => {
  const [ q, r ] = linalg.qr(matrix);
  const solution = await linSolveFromQR(q, r, v);
  return solution;
};
