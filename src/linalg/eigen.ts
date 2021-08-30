import { linalg, Tensor, matMul, abs, sub, max, transpose, tensor, mul, eye, slice, stack } from '@tensorflow/tfjs-core';
import { linSolveQR } from './linsolve';
import { tensorNormalize, tensorEqual } from './utils';
/**
 * Compute the eigenvalues of a matrix using the QR algorithm.
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
export const solveEigenValues = (matrix: Tensor, tol = 1e-4, maxIter = 200): Tensor => {
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

  return d;
};

/**
 * Solve for the eigenvector associated with an eigenvalue using the inverse
   iteration algorithm.
  Given an approximate eigenvalue lambda, the inverse iteration algorithm
  computes the matrix:
    M' = M - lambda I
  And then solves the following sequence of linear equations:
    v0 = solve(M', random_vector), v0' = normalize(v0);
    v1 = solve(M', v0'), v1' = normalize(v1);
    v2 = solve(<', v1'), v2' = normalize(v2);
    ...
  This algorithm will converge to the eigenvector associated with the eigenvalue
  closest to lambda.
 * @param matrix matrix
 * @param eigenValue eigen value
 * @param tol tolerance, default to 1e-4
 * @param maxIter max iteration time, default to 200
 */
export const eigenBackSolve = (matrix: Tensor, eigenValue: number, tol = 1e-4, maxIter = 200): Tensor => {
  const n = matrix.shape[0];
  let current = tensor(new Array(n).fill(1));
  let previous;
  // Preturb the eigenvalue a litle to prevent our right hand side matrix
  // from becoming singular.
  const lambda = eigenValue + 0.00001;
  const mi = sub(matrix, mul(eye(n), lambda));
  for (let i = 0; i < maxIter; i++) {
    previous = current;
    current = linSolveQR(mi, previous);
    current = tensorNormalize(current);
    if (tensorEqual(current, previous, tol)) {
      break;
    }
  }
  return current;
};

/* Solve for the eigenvectors of a matrix M once the eigenvalues are known
   using inverse iteration.
*/
export const solveEigenVectors = (matrix: Tensor, eigenValues: Tensor, tol = 1e-4, maxIter = 200): Tensor => {
  const nEv = eigenValues.shape[0];
  const eigenVectors = [];
  for (let i = 0; i < nEv; i++) {
    const eigenValue = Number(slice(eigenValues, 1, 1).dataSync());
    const eigenVector = eigenBackSolve(matrix, eigenValue, tol, maxIter);
    eigenVectors.push(eigenVector);
  }
  return stack(eigenVectors);
};

