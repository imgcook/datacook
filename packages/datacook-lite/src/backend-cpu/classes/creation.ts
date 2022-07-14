import { Vector } from "./vector";
import { Matrix } from "./matrix";

export const createZeroMatrix = (n: number, m: number): Matrix => {
  const mat: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    mat[i] = new Array(m).fill(0);
  }
  return new Matrix(mat);
};

export const createOneMatrix = (n: number, m: number): Matrix => {
  const mat: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    mat[i] = new Array(m).fill(1);
  }
  return new Matrix(mat);
};


export const createOneVector = (n: number): Vector => {
  const arr = new Array(n).fill(0);
  return new Vector(arr);
};
