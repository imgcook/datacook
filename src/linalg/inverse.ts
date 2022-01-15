import { Tensor, Tensor2D, tensor2d } from '@tensorflow/tfjs-core';

/**
 * Matrix inverse use Gauss-Jordan elimination
 * @param matrix input matrix of shape (m, n)
 * @returns inverse of matrix
 */
export const inverseArray = async (matrix: number[][]): Promise<number[][]> => {
  let i = 0;
  let ii = 0;
  let j = 0;
  const dim = matrix.length;
  let e = 0;
  const I = new Array(dim).fill(0).map(() => new Array(dim).fill(0));
  const C = new Array(dim).fill(0).map(() => new Array(dim).fill(0));

  for (i = 0; i < dim; i += 1) {
    for (j = 0; j < dim; j += 1) {
      if (i === j) {
        I[i][j] = 1;
      }
      C[i][j] = matrix[i][j];
    }
  }

  for (i = 0; i < dim; i += 1) {
    e = C[i][i];

    if (e === 0) {
      for (ii = i + 1; ii < dim; ii += 1) {
        if (C[ii][i] !== 0) {
          for (j = 0; j < dim; j++) {
            e = C[i][j];
            C[i][j] = C[ii][j];
            C[ii][j] = e;
            e = I[i][j];
            I[i][j] = I[ii][j];
            I[ii][j] = e;
          }
          break;
        }
      }
      e = C[i][i];
      if (e === 0) {
        throw new TypeError('Matrix is not invertible');
      }
    }

    for (j = 0; j < dim; j++) {
      C[i][j] = C[i][j] * 1e6 / e / 1e6;
      I[i][j] = I[i][j] * 1e6 / e / 1e6;
    }

    for (ii = 0; ii < dim; ii++) {
      if (ii == i) {
        continue;
      }
      e = C[ii][i];
      for (j = 0; j < dim; j++) {
        C[ii][j] = C[ii][j] - e * 1e6 * C[i][j] / 1e6;
        I[ii][j] = I[ii][j] - e * 1e6 * I[i][j] / 1e6;
      }
    }
  }
  return I;
};

export const inverse = async (matrix: Tensor2D | number[][]): Promise<Tensor2D> => {
  // Gauss-Jordan elimination to invert 2d matrix
  let matrixData: number[][];
  if (matrix instanceof Tensor) {
    matrixData = await matrix.array();
  } else {
    matrixData = matrix;
  }
  return tensor2d(await inverseArray(matrixData));
};
