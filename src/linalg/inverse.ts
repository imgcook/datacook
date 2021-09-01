import { Tensor, greater, min, abs, matMul, diag, transpose, divNoNan } from "@tensorflow/tfjs-core";
import { eigenSolve } from "./eigen";
import { isSquareMatrix } from "./utils";

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
