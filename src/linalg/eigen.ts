import { linalg, Tensor, matMul, abs, sub, max, transpose } from '@tensorflow/tfjs-core';

/**
 * Compute the eigenvalues and eigenvector of a matrix using the QR algorithm.
  This is a renormalized version of power iteration that converges to a full
  set of eigenvalues.  Starting with the matrix M = M0, we iterate:
    M0 = Q0 R0, M1 = R0 Q0;
    M1 = Q1 R1, M2 = R1 Q1;
    M2 = Q2 R2, M3 = R2 Q2;
    ...
  For a general matrix with a full set of eigenvalues, this sequence will
  converge to an upper diagonal matrix:
    Mi -> upper diagonal matrix
  The diagonal entries of this matrix are the eigenvalues of M.
  Qn is the eigenvector of M.
 * @param matrix square matrix of shape (n, n)
   @param tol tolerence, default to 1e-4
   @param maxIter max iteration time, default to 200
 */
export const eigenSolve = (matrix: Tensor, tol = 1e-6, maxIter = 200): [ Tensor, Tensor ] => {
  let [ q, r ] = linalg.qr(matrix);
  let x = matrix;
  let xTr = linalg.bandPart(x, 0, 0);
  //let preQ = q;
  let qn = q;

  for (let i = 0; i < maxIter; i++) {
    //preQ = q;
    x = matMul(r, q);
    [ q, r ] = linalg.qr(x);
    qn = matMul(qn, q);
    xTr = linalg.bandPart(x, 0, 0);
    const maxDis = max(abs(sub(x, xTr))).arraySync();
    if ( maxDis < tol) {
      break;
    }
    //console.log(maxDis);
  }
  const d = linalg.bandPart(matMul(r, q), 0, 0);

  return [ qn, d ];
};
