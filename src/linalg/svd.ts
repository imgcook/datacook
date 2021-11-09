import { eigenSolve } from './eigen';
import { matMul, Tensor, transpose, sqrt } from '@tensorflow/tfjs-core';

/**
 * Singular value decomposition using QR iteration and inverse iteration algorithm
 * the singular value decomposition (SVD) is a factorization of a real or complex matrix.
 * It generalizes the eigendecomposition of a square normal matrix with
 * an orthonormal eigenbasis to any m * n matrix.
 * @param matrix target matrix
 * @param tol tolerence, default ot 1e-4
 * @param maxIter max iteration times, default to 200
 * @returns [ u, d, v ], u: left singular vector, d: singluar values, v: right singular vector
 */
export const svd = async (matrix: Tensor, tol = 1e-4, maxIter = 200): Promise<[ Tensor, Tensor, Tensor ]> => {
  const m1 = matMul(matrix, transpose(matrix));
  const m2 = matMul(transpose(matrix), matrix);
  const [ m, n ] = matrix.shape;
  const [ eigenValues1, eigenVectors1 ] = await eigenSolve(m1, tol, maxIter);
  const [ eigenValues2, eigenVectors2 ] = await eigenSolve(m2, tol, maxIter);
  const singularValues = m < n ? sqrt(eigenValues1) : sqrt(eigenValues2);
  return [ eigenVectors1, singularValues, eigenVectors2 ];
};
