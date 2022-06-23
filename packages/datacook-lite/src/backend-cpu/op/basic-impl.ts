import { vector } from "../../core/classes";
import { Matrix } from "../classes/matrix";
import { Vector } from "../classes/vector";
import { squeeze } from "./transform";

export enum ByAxis {
  ByColumn = 0,
  ByRow = 1
}

export type ImplementFuncBinary = (a: number, b: number) => number;
export type ImplementFuncSingle = (a: number) => number;
export type ImplementFuncReduce = (a: number[]) => number;

export const basicImplement2dBinary = (func: ImplementFuncBinary, x: Matrix, y: Matrix | Vector | number, by?: ByAxis): Matrix => {
  const [ nX, mX ] = x.shape;
  if (!(y instanceof Matrix) && !(y instanceof Vector) && !(typeof y === 'number')) {
    throw new TypeError('Invalid input y, { Matrix |  Vector | number } is required');
  }
  const out: number[][] = [];
  for (let i = 0; i < nX; i++) {
    out[i] = new Array(mX).fill(0);
  }
  if (y instanceof Matrix) {
    const [ nY, mY ] = y.shape;
    if (nX !== nY || mX !== mY) {
      throw new TypeError(`Mismatch of shape [${nX}, ${mX}] and [${nY}, ${mY}]`);
    }
    for (let i = 0; i < nX; i++) {
      for (let j = 0; j < nY; j++) {
        out[i][j] = func(x.get(i, j), y.get(i, j));
      }
    }
  }
  if (y instanceof Vector) {
    const nY = y.length;
    if (by !== ByAxis.ByRow && by !== ByAxis.ByColumn) {
      throw new TypeError('Invalid input for `by`, expect 1 or 0');
    }
    if (by === ByAxis.ByRow) {
      if (nY !== nX) {
        throw new TypeError(`Required length of y is ${nX}, receive ${nY}`);
      }
      for (let i = 0; i < nX; i++) {
        const yVal = y.get(i);
        out[i] = x.getRow(i).data.map((d: number) => func(d, yVal));
      }
    }
    if (by === ByAxis.ByColumn) {
      if (nY !== mX) {
        throw new TypeError(`Required length of y is ${mX}, receive ${nY}`);
      }
      for (let j = 0; j < mX; j++) {
        const yVal = y.get(j);
        for (let i = 0; i < nX; i++) {
          out[i][j] = func(x.get(i, j), yVal);
        }
      }
    }
  }
  if (typeof y === 'number') {
    for (let i = 0; i < nX; i++) {
      for (let j = 0; j < mX; j++) {
        out[i][j] = func(x.get(i, j), y);
      }
    }
  }
  return new Matrix(out);
};

export const basicImplement1dSingle = (func: ImplementFuncSingle, x: Vector): Vector => {
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = func(x.get(i));
  }
  return vector(out);
};

export const basicImplement2dSingle = (func: ImplementFuncSingle, x: Matrix): Matrix => {
  const [ nX, mX ] = x.shape;
  const out: number[][] = [];
  for (let i = 0; i < nX; i++) {
    out[i] = new Array(mX).fill(0);
  }
  for (let i = 0; i < nX; i++) {
    for (let j = 0; j < mX; j++) {
      out[i][j] = func(x.get(i, j));
    }
  }
  return new Matrix(out);
};

export const basicImplement2dReduceAll = (func: ImplementFuncReduce, x: Matrix): number => {
  const squeezed = squeeze(x);
  return func(squeezed.data);
};

export const basicImplement2dReduce = (func: ImplementFuncReduce, x: Matrix, by: ByAxis): Vector => {
  const [ nX, mX ] = x.shape;
  if (! (by === ByAxis.ByRow || by === ByAxis.ByColumn)) {
    throw new TypeError(`Invalid input for 'by': ${by}`);
  }
  if (by === ByAxis.ByRow) {
    const out: number[] = new Array(nX);
    for (let i = 0; i < nX; i++) {
      out[i] = func(x.getRow(i).data);
    }
    return new Vector(out);
  }
  if (by === ByAxis.ByColumn) {
    const out: number[] = new Array(nX);
    for (let i = 0; i < mX; i++) {
      out[i] = func(x.getColumn(i).data);
    }
    return new Vector(out);
  }
};
