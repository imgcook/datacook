import { Matrix } from "./matrix";

export const createZeroMatrix = (n: number, m: number): Matrix => {
  const mat: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    mat[i] = new Array(m).fill(0);
  }
  return new Matrix(mat);
};

