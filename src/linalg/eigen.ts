import { linalg, Tensor, matMul, abs, sub, max, tensor, mul, eye, slice, stack, squeeze, neg, transpose } from '@tensorflow/tfjs-core';
import { linSolveQR } from './linsolve';
import { tensorNormalize, tensorEqual } from './utils';
/**
 * Compute the eigenvalues of a matrix using the QR algorithm.
 * This is a renormalized version of power iteration that converges to a full
 * set of eigenvalues.  Starting with the matrix M = M0, we iterate:
 *   M0 = Q0 R0, M1 = R0 Q0;
 *   M1 = Q1 R1, M2 = R1 Q1;
 *   M2 = Q2 R2, M3 = R2 Q2;
 *   ...
 * For a general matrix with a full set of eigenvalues, this sequence will
 * converge to an upper diagonal matrix:
 *   Mi -> upper diagonal matrix
 * The diagonal entries of this matrix are the eigenvalues of M.
 * Qn is the eigenvector of M.
 * @param matrix square matrix of shape (n, n)
 * @param tol tolerence, default to 1e-4
 * @param maxIter max iteration time, default to 200
 */
export const solveEigenValues = async(matrix: Tensor, tol = 1e-4, maxIter = 200): Promise<Tensor> => {
  let [ q, r ] = linalg.qr(matrix);
  let x = matrix;
  let prevX: Tensor;
  let prevTr: Tensor;
  const n = matrix.shape[0];
  let xTr = linalg.bandPart(x, 0, 0);
  let qn = q;

  for (let i = 0; i < maxIter; i++) {
    prevX = x;
    x = matMul(r, q);
    [ q, r ] = linalg.qr(x);
    qn = matMul(qn, q);
    xTr = linalg.bandPart(x, 0, 0);
    prevTr = linalg.bandPart(prevX, 0, 0);
    const maxDis = await max(abs(sub(prevTr, xTr))).array();
    if (maxDis < tol) {
      break;
    }
  }
  x = matMul(r, q);
  const eigenValues = [];
  for (let i = 0; i < n; i++) {
    eigenValues.push(slice(x, [ i, i ], [ 1, 1 ]));
  }
  return squeeze(stack(eigenValues));
};

/**
 * Solve for the eigenvector associated with an eigenvalue using the inverse
 * iteration algorithm.
 * Given an approximate eigenvalue lambda, the inverse iteration algorithm
 * computes the matrix:
 *   M' = M - lambda I
 * And then solves the following sequence of linear equations:
 *   v0 = solve(M', random_vector), v0' = normalize(v0);
 *   v1 = solve(M', v0'), v1' = normalize(v1);
 *   v2 = solve(<', v1'), v2' = normalize(v2);
 *   ...
 * This algorithm will converge to the eigenvector associated with the eigenvalue
 * closest to lambda.
 * @param matrix matrix
 * @param eigenValue eigen value
 * @param tol tolerance, default to 1e-4
 * @param maxIter max iteration time, default to 200
 */
export const eigenBackSolve = async(matrix: Tensor, eigenValue: number, tol = 1e-4, maxIter = 200): Promise<Tensor> => {
  const nCols = matrix.shape[0];
  let current = tensor(new Array(nCols).fill(1));
  let previous;
  // Preturb the eigenvalue a litle to prevent our right hand side matrix
  // from becoming singular.
  const lambda = eigenValue;
  const mi = sub(matrix, mul(eye(nCols), lambda));
  for (let i = 0; i < maxIter; i++) {
    previous = current;
    current = await linSolveQR(mi, previous);
    current = tensorNormalize(current);
    /**
     * We reverse the sign of the vector if the first entry is not positive.
     * Often the algorithm will oscilate between a vector and its negative
     * after convergence.
     */
    const pivot = Number(await slice(current, 0, 1).data());
    if (pivot < 0) {
      current = neg(current);
    }
    if (tensorEqual(current, previous, tol)) {
      break;
    }
  }
  return current;
};

/**
 * Solve for the eigenvectors of a matrix M once the eigenvalues are known
 * using inverse iteration.
 * @param matrix target matrix
 * @param eigenValues eigen values
 * @param tol tolerence, default to 1e-4
 * @param maxIter max iteration, default to 200
 * @returns eigen vectors corresponding to target eigen values
 */
export const solveEigenVectors = async (matrix: Tensor, eigenValues: Tensor, tol = 1e-4, maxIter = 200): Promise<Tensor> => {
  const nEv = eigenValues.shape[0];
  const eigenVectors = [];
  for (let i = 0; i < nEv; i++) {
    const eigenValue = Number(await slice(eigenValues, i, 1).data());
    const eigenVector = await eigenBackSolve(matrix, eigenValue, tol, maxIter);
    eigenVectors.push(eigenVector);
  }
  return transpose(stack(eigenVectors));
};

/**
 * Compute the eigenvalues and eigenvectors of a matrix M.
 * The eigenvalues are computed using the QR algorithm, then the eigenvectors
 * are computed by inverse iteration.
 * @param matrix target matrix
 * @param tol stop tolerence, default to 1e-4
 * @param maxIter max iteration times, default to 200
 */
export const eigenSolve = async(matrix: Tensor, tol = 1e-4, maxIter = 200): Promise<[Tensor, Tensor]> => {
  const eigenValues = await solveEigenValues(matrix, tol, maxIter);
  const eigenVectors = await solveEigenVectors(matrix, eigenValues, tol, maxIter);
  return [ eigenValues, eigenVectors ];
};
