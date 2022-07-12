import { Matrix, Vector, Scalar } from "./classes";
import { Matrix as MatrixCPU, Vector as VectorCPU, Scalar as ScalarCPU } from '../../backend-cpu/classes';
import { IS_CPU_BACKEND } from '../../env';

export const createZeroMatrix = (n: number, m: number): Matrix => {
  const mat: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    mat[i] = new Array(m).fill(0);
  }
  if (IS_CPU_BACKEND) {
    return new MatrixCPU(mat);
  } else {
    throw new TypeError('');
  }
};

export const matrix = (data: number[][]): Matrix => {
  if (IS_CPU_BACKEND) {
    return new MatrixCPU(data);
  } else {
    throw new TypeError('');
  }
};

export const vector = (data: number[]): Vector => {
  if (IS_CPU_BACKEND) {
    return new VectorCPU(data);
  } else {
    throw new TypeError('');
  }
};

export const scalar = (data: number): Scalar => {
  if (IS_CPU_BACKEND) {
    return new ScalarCPU(data);
  } else {
    throw new TypeError('');
  }
};
