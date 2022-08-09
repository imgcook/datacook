import { Vector } from "./vector";
import { Matrix } from "./matrix";
import { ByAxis } from "../op/basic-impl";
import { transpose2dForward } from "../op/transform";

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

export const createZeroVector = (n: number): Vector => {
  const arr = new Array(n).fill(0);
  return new Vector(arr);
};

export const createOneVector = (n: number): Vector => {
  const arr = new Array(n).fill(1);
  return new Vector(arr);
};

export const createRangeVector = (start: number, end: number, step = 1): Vector => {
  const arr = [];
  for (let i = start; i < end; i += step) {
    arr.push(i);
  }
  return new Vector(arr);
};

export const concatVector = (vecs: Vector[], by: ByAxis = 0): Matrix => {
  let maxVecLen = 0;
  for (let i = 0; i < vecs.length; i++) {
    if (vecs[i].length > maxVecLen) maxVecLen = vecs[i].length;
  }
  const outMat = createZeroMatrix(maxVecLen, vecs.length);
  for (let i = 0; i < vecs.length; i++) {
    outMat.setColumn(i, vecs[i]);
  }
  return by === ByAxis.ByRow ? outMat : transpose2dForward(outMat);
};
