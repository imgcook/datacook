import { vector } from "../../core/classes";
import { Matrix } from "../classes/matrix";
import { Vector } from "../classes/vector";
import { Scalar } from "../classes/scalar";
import { squeeze } from "./transform";
import { sum1dForward, sum2dForward } from "./reduce-op";

export enum ByAxis {
  ByColumn = 0,
  ByRow = 1
}

export type ImplementFuncBinary = (a: number, b: number) => number;
export type ImplementFuncSingle = (a: number) => number;
export type ImplementFuncReduce = (a: number[]) => number;

export type BinaryFunc2d = (x: Matrix, y: Matrix | Vector | Scalar | number, by?: ByAxis) => Matrix;
export type BinaryGradFunc2d = (grad: Matrix, x?: Matrix, y?: Matrix | Vector | Scalar | number, by?: ByAxis) => Matrix;
export type BinaryFunc1d = (x: Vector, y: Vector | Scalar | number) => Vector;
export type BinaryGradFunc1d = (grad: Vector, x?: Vector, y?: Vector | Scalar | number) => Vector;

export type SingleFunc2d = (x: Matrix) => Matrix;
export type SingleGradFunc2d = (grad: Matrix, x?: Matrix, param?: any) => Matrix;
export type SingleFunc1d = (x: Vector) => Vector;
export type SingleGradFunc1d = (grad: Vector, x?: Vector, param?: any) => Vector;

export type ReduceAllFunc2d = (x: Matrix) => Scalar;
export type ReduceAllGradFunc2d = (grad: Scalar, x: Matrix) => Matrix;

export type ReduceFunc2d = (x: Matrix) => Vector;
export type ReducedGradFunc2d = (grad: Vector, x: Matrix) => Matrix;

export type ReduceFunc1d = (x: Vector) => Scalar;
export type ReducedGradFunc1d = (grad: Scalar, x: Vector) => Vector;

export const basicImplement2dBinary = (func: ImplementFuncBinary, x: Matrix, y: Matrix | Vector | Scalar | number, by?: ByAxis): Matrix => {
  const [ nX, mX ] = x.shape;
  if (!(y instanceof Matrix) && !(y instanceof Vector) && !(typeof y === 'number') && !(y instanceof Scalar) ) {
    throw new TypeError('Invalid input y, { Matrix |  Vector | Scalar | number } is required');
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
  if (typeof y === 'number' || y instanceof Scalar) {
    const t = y instanceof Scalar ? y.data : y;
    for (let i = 0; i < nX; i++) {
      for (let j = 0; j < mX; j++) {
        out[i][j] = func(x.get(i, j), t);
      }
    }
  }
  return new Matrix(out);
};

export const trackedImplement2dBinary = (forwardFunc: BinaryFunc2d, backwardFuncX: BinaryGradFunc2d, backwardFuncY: BinaryGradFunc2d, x: Matrix, y: Matrix | Vector | Scalar | number, by?: ByAxis): Matrix => {
  const outMat = forwardFunc(x, y, by);
  outMat.dependency.push({
    target: x,
    gradFunc: (grad: Matrix): Matrix => backwardFuncX(grad, x, y)
  });
  if (y instanceof Matrix) {
    outMat.dependency.push({
      target: y,
      gradFunc: (grad: Matrix): Matrix => backwardFuncY(grad, x, y)
    });
  }
  if (y instanceof Vector) {
    outMat.dependency.push({
      target: y,
      gradFunc: (grad: Matrix): Vector => sum2dForward(backwardFuncY(grad, x, y), by) as Vector
    });
  }
  if (y instanceof Scalar) {
    outMat.dependency.push({
      target: y,
      gradFunc: (grad: Matrix): Scalar => sum2dForward(backwardFuncY(grad, x, y)) as Scalar
    });
  }
  return outMat;
};

export const basicImplement1dBinary = (func: ImplementFuncBinary, x: Vector, y: Vector | Scalar | number): Vector => {
  const nX = x.length;
  const out = new Array(nX);
  if (y instanceof Vector) {
    const nY = y.length;
    if (nX !== nY) {
      throw new TypeError(`Required length of y is ${nX}, receive ${nY}`);
    }
    for (let i = 0; i < nX; i++) {
      out[i] = func(x.get(i), y.get(i));
    }
    return vector(out);
  }
  if (typeof y === 'number') {
    const out = x.data.map((d: number) => d + y);
    return vector(out);
  }
};

export const trackedImplement1dBinary = (forwardFunc: BinaryFunc1d, backwardFuncX: BinaryGradFunc1d, backwardFuncY: BinaryGradFunc1d, x: Vector, y: Vector | Scalar | number): Vector => {
  const outMat = forwardFunc(x, y);
  outMat.dependency.push({
    target: x,
    gradFunc: (grad: Vector): Vector => backwardFuncX(grad, x, y)
  });
  if (y instanceof Vector) {
    outMat.dependency.push({
      target: y,
      gradFunc: (grad: Vector): Vector => backwardFuncY(grad, x, y)
    });
  }
  if (y instanceof Scalar) {
    outMat.dependency.push({
      target: y,
      gradFunc: (grad: Vector): Scalar => sum1dForward(backwardFuncY(grad, x, y)) as Scalar
    });
  }
  return outMat;
};

export const basicImplement1dSingle = (func: ImplementFuncSingle, x: Vector): Vector => {
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = func(x.get(i));
  }
  return vector(out);
};

export const trackedImplement1dSingle = (forwardFunc: SingleFunc1d, backwardFunc: SingleGradFunc1d, x: Vector): Vector => {
  const outMat = forwardFunc(x);
  outMat.dependency.push({
    target: x,
    gradFunc: (grad: Vector): Vector => backwardFunc(grad, x)
  });
  return outMat;
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

export const trackedImplement2dSingle = (forwardFunc: SingleFunc2d, backwardFuncX: SingleGradFunc2d, x: Matrix): Matrix => {
  const outMat = forwardFunc(x);
  outMat.dependency.push({
    target: x,
    gradFunc: (grad: Matrix): Matrix => backwardFuncX(grad, x)
  });
  return outMat;
};

export const basicImplement2dReduceAll = (func: ImplementFuncReduce, x: Matrix): Scalar => {
  const squeezed = squeeze(x);
  return new Scalar(func(squeezed.data));
};

export const trackedImplement2dReduceAll = (forwardFunc: ReduceAllFunc2d, backwardFunc: ReduceAllGradFunc2d, x: Matrix): Scalar => {
  const outMat = forwardFunc(x);
  outMat.dependency.push({
    target: x,
    gradFunc: (grad: Scalar): Matrix => backwardFunc(grad, x)
  });
  return outMat;
};

export const basicImplement1dReduce = (func: ImplementFuncReduce, x: Vector): Scalar => {
  return new Scalar(func(x.data));
};

export const trackedImplement1dReduce = (forwardFunc: ReduceFunc1d, backwardFunc: ReducedGradFunc1d, x: Vector): Scalar => {
  const outMat = forwardFunc(x);
  outMat.dependency.push({
    target: x,
    gradFunc: (grad: Scalar): Vector => backwardFunc(grad, x)
  });
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

export const trackedImplement2dReduce = (forwardFunc: ReduceFunc2d, backwardFunc: ReducedGradFunc2d, x: Matrix): Vector => {
  const outMat = forwardFunc(x);
  outMat.dependency.push({
    target: x,
    gradFunc: (grad: Vector): Matrix => backwardFunc(grad, x)
  });
  return outMat;
};
