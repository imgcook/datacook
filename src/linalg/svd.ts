import { eigenSolve } from "./eigen";
import { matMul, Tensor, transpose, sqrt, slice, tensor } from "@tensorflow/tfjs-core";

export const svd = (matrix: Tensor): [ Tensor, Tensor, Tensor ] => {
  const m1 = matMul(matrix, transpose(matrix));
  const m2 = matMul(transpose(matrix), matrix);
  const [ m, n ] = matrix.shape;
  const svData = new Array(m).fill(0).map( () => new Array(n).fill(0));
  const svSize = m < n ? m : n;
  const [ eigenValues1, eigenVectors1 ] = eigenSolve(m1);
  const [ , eigenVectors2 ] = eigenSolve(m2);
  const singularValues = sqrt(eigenValues1);
  for (let i = 0; i < svSize; i++) {
    svData[i][i] = Number(slice(singularValues, i, 1).dataSync());
  }
  console.log(svData);
  const singularDiag = tensor(svData);
  return [ eigenVectors1, singularDiag, eigenVectors2 ];
};

