import { Tensor, greater, min, abs, matMul, diag, transpose, divNoNan } from '@tensorflow/tfjs-core';
import { eigenSolve } from './eigen';
import { isSquareMatrix } from './utils';

/**
 * Compute the inverse of a square matrix.
 * In linear algebra, an n-by-n square matrix A is called invertible (also nonsingular or nondegenerate),
 * if there exists an n-by-n square matrix B such that AB = BA = In.
 * @param matrix target matrix
 * @returns inverse of the target matrix
 */
export const inverse = async (matrix: Tensor, tol = 1e-4, maxIter = 200): Promise<Tensor> => {
  if (!isSquareMatrix(matrix)) {
    throw new TypeError('Not square matrix');
  }
  const [ eigenValues, eigenVectors ] = await eigenSolve(matrix, tol, maxIter);
  const minEigen = min(abs(eigenValues));
  const invertable = Boolean((await greater(minEigen, 1e-4).data())[0]);
  if (!invertable) {
    throw new TypeError('Singular matrix error');
  }
  const inverseEigenValues = divNoNan(1, eigenValues);
  return matMul(matMul(eigenVectors, diag(inverseEigenValues)), transpose(eigenVectors));
};
