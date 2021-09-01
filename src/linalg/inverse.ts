import { Tensor, greater, min, abs, matMul, diag, transpose, divNoNan } from "@tensorflow/tfjs-core";
import { eigenSolve } from "./eigen";
import { isSquareMatrix } from "./utils";

/**
 * Compute the inverse of a square matrix
 * In linear algebra, an n-by-n square matrix A is called invertible (also nonsingular or nondegenerate),
 * if there exists an n-by-n square matrix B such that AB = BA = In.
 * @param matrix target matrix
 * @returns inverse of the target matrix
 */
export const inverse = (matrix: Tensor): Tensor => {
  if (isSquareMatrix) {
    const [ eigenValues, eigenVectors ] = eigenSolve(matrix);
    const minEigen = min(abs(eigenValues));
    const invertable = Boolean(greater(minEigen, 1e-4));
    if (!invertable) {
      throw new Error('Singlular matrix error');
    } else {
      const inverseEigenValues = divNoNan(1, eigenValues);
      const inverseM = matMul(matMul(eigenVectors, diag(inverseEigenValues)), transpose(eigenVectors));
      return inverseM;
    }
  } else {
    throw new Error('Singular matrix error');
  }
};
