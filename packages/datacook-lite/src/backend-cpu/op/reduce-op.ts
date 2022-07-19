import { Scalar, vector } from "../../core/classes";
import { createOneMatrix, createZeroMatrix, Matrix, Vector } from "../classes";
import { concatVector, createRangeVector } from "../classes/creation";
import {
  basicImplement2dReduce,
  basicImplement2dReduceAll,
  basicImplement1dReduce,
  trackedImplement1dReduce,
  ByAxis,
  trackedImplement2dReduce
} from "./basic-impl";
import { div2dForward, mul2dForward } from "./binary-op";

const sumFunc = (a: number[]): number => {
  return a.reduce((p, q) => p + q);
};

const meanFunc = (a: number[]): number => {
  return a.reduce((p, q) => p + q) / a.length;
};

const minFunc = (a: number[]): number => {
  let min = Number.MAX_SAFE_INTEGER;
  for (let i = 0; i < a.length; i++) {
    if (a[i] < min) {
      min = a[i];
    }
  }
  return min;
};

const maxFunc = (a: number[]): number => {
  let max = Number.MIN_SAFE_INTEGER;
  for (let i = 0; i < a.length; i++) {
    if (a[i] > max) {
      max = a[i];
    }
  }
  return max;
};

const argMinFunc = (a: number[]): number => {
  let min = Number.MAX_SAFE_INTEGER;
  let argMin = -1;
  for (let i = 0; i < a.length; i++) {
    if (a[i] < min) {
      min = a[i];
      argMin = i;
    }
  }
  return argMin;
};

const argMaxFunc = (a: number[]): number => {
  let max = Number.MIN_SAFE_INTEGER;
  let argMax = -1;
  for (let i = 0; i < a.length; i++) {
    if (a[i] > max) {
      max = a[i];
      argMax = i;
    }
  }
  return argMax;
};


export const sum1dForward = (x: Vector): Scalar => basicImplement1dReduce(sumFunc, x);
export const sum1dBackward = (grad: Scalar, x: Vector): Vector => vector(new Array(x.length).fill(grad.data));

export const sum1d = (x: Vector): Scalar => {
  return trackedImplement1dReduce(sum1dForward, sum1dBackward, x);
};

export const mean1dForward = (x: Vector): Scalar => basicImplement1dReduce(meanFunc, x);
export const mean1dBackward = (grad: Scalar, x: Vector): Vector => vector(new Array(x.length).fill(grad.data / x.length));

export const mean1d = (x: Vector): Scalar => {
  return trackedImplement1dReduce(mean1dForward, mean1dBackward, x);
};

export const argMax1dForward = (x: Vector): Scalar => basicImplement1dReduce(argMaxFunc, x);

export const argMin1dForward = (x: Vector): Scalar => basicImplement1dReduce(argMinFunc, x);

export const min1dForawrd = (x: Vector): Scalar => basicImplement1dReduce(minFunc, x);
export const min1dBackward = (grad: Scalar, x: Vector): Vector => {
  const minInd = argMin1dForward(x).data;
  const arr = new Array(x.length).fill(0);
  arr[minInd] = grad.data;
  return vector(arr);
};

export const min1d = (x: Vector): Scalar => {
  return trackedImplement1dReduce(min1dForawrd, min1dBackward, x);
};

export const max1dForawrd = (x: Vector): Scalar => basicImplement1dReduce(maxFunc, x);
export const max1dBackward = (grad: Scalar, x: Vector): Vector => {
  // const minInd = argMin1dForward(grad, x);
  const maxInd = argMax1dForward(x).data;
  const arr = new Array(x.length).fill(0);
  arr[maxInd] = grad.data;
  return vector(arr);
};

export const max1d = (x: Vector): Scalar => {
  return trackedImplement1dReduce(max1dForawrd, max1dBackward, x);
};

export const argMax1d = (x: Vector): Scalar => basicImplement1dReduce(argMaxFunc, x);

export const argMin1d = (x: Vector): Scalar => basicImplement1dReduce(argMinFunc, x);

export const sum2dForward = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === ByAxis.ByAll) {
    return basicImplement2dReduceAll(sumFunc, x);
  }
  return basicImplement2dReduce(sumFunc, x, by);
};

export const sum2dBackward = (grad: Vector | Scalar, x: Matrix, by?: ByAxis): Matrix => {
  return mul2dForward(createOneMatrix(x.shape[0], x.shape[1]), grad, by);
};

export const sum2d = (x: Matrix, by?: ByAxis): Scalar | Vector => {
  return trackedImplement2dReduce(sum2dForward, sum2dBackward, x, by);
};

export const mean2dForward = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === ByAxis.ByAll) {
    return basicImplement2dReduceAll(meanFunc, x);
  }
  return basicImplement2dReduce(meanFunc, x, by);
};

export const mean2dBackward = (grad: Vector | Scalar, x: Matrix, by?: ByAxis): Matrix => {
  const mat = mul2dForward(createOneMatrix(x.shape[0], x.shape[1]), grad, by);
  if (grad instanceof Vector) {
    if (by === ByAxis.ByColumn) {
      return div2dForward(mat, x.shape[0], by);
    } else {
      return div2dForward(mat, x.shape[1], by);
    }
  } else {
    return div2dForward(mat, x.shape[1] * x.shape[0]);
  }
};

export const mean2d = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  return trackedImplement2dReduce(mean2dForward, mean2dBackward, x, by);
};

export const argMin2dForward = (x: Matrix, by?: ByAxis): Vector | Matrix => {
  if (by === ByAxis.ByAll) {
    const ind = basicImplement2dReduceAll(argMinFunc, x);
    const indI = Math.floor(ind.data / x.shape[1]);
    const indJ = Math.floor(ind.data % x.shape[1]);
    return vector([ indI, indJ ]);
    // return basicImplement2dReduceAll(argMinFunc, x);
  } else {
    const indI = by === ByAxis.ByColumn ? createRangeVector(0, x.shape[1]) : createRangeVector(0, x.shape[0]);
    const indJ = basicImplement2dReduce(argMinFunc, x, by);
    if (by === ByAxis.ByColumn)
      return concatVector([ indJ, indI ], ByAxis.ByRow);
    else
      return concatVector([ indI, indJ ], ByAxis.ByRow);
  }
  // return basicImplement2dReduce(argMinFunc, x, by);
};

export const argMax2dForward = (x: Matrix, by?: ByAxis): Vector | Matrix => {
  if (by === ByAxis.ByAll) {
    const ind = basicImplement2dReduceAll(argMaxFunc, x);
    const indI = Math.floor(ind.data / x.shape[1]);
    const indJ = Math.floor(ind.data % x.shape[1]);
    return vector([ indI, indJ ]);
    // return basicImplement2dReduceAll(argMinFunc, x);
  } else {
    const indI = by === ByAxis.ByColumn ? createRangeVector(0, x.shape[1]) : createRangeVector(0, x.shape[0]);
    const indJ = basicImplement2dReduce(argMaxFunc, x, by);
    if (by === ByAxis.ByColumn)
      return concatVector([ indJ, indI ], ByAxis.ByRow);
    else
      return concatVector([ indI, indJ ], ByAxis.ByRow);
  }
  // return basicImplement2dReduce(argMinFunc, x, by);
};

export const argMin2d = argMin2dForward;
export const argMax2d = argMax2dForward;

export const min2dForward = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === ByAxis.ByAll) {
    return basicImplement2dReduceAll(minFunc, x);
  }
  return basicImplement2dReduce(minFunc, x, by);
};

export const min2dBackward = (grad: Scalar | Vector, x: Matrix, by?: ByAxis): Matrix => {
  const mat = createZeroMatrix(x.shape[0], x.shape[1]);
  const argMinInds = argMin2dForward(x, by);
  if (by === ByAxis.ByAll) {
    const [ i, j ] = (argMinInds as Vector).data;
    mat.set(i, j, (grad as Scalar).data);
  } else {
    const inds = (argMinInds as Matrix).data;
    inds.forEach((v: [number, number]) => {
      const val = by === ByAxis.ByRow ? (grad as Vector).get(v[0]) : (grad as Vector).get(v[1]);
      mat.set(v[0], v[1], val);
    });
  }
  return mat;
};

export const min2d = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  return trackedImplement2dReduce(min2dForward, min2dBackward, x, by);
};

export const max2dForward = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === ByAxis.ByAll) {
    return basicImplement2dReduceAll(maxFunc, x);
  }
  return basicImplement2dReduce(maxFunc, x, by);
};

export const max2dBackward = (grad: Scalar | Vector, x: Matrix, by?: ByAxis): Matrix => {
  const mat = createZeroMatrix(x.shape[0], x.shape[1]);
  const argMaxInds = argMax2dForward(x, by);
  if (by === ByAxis.ByAll) {
    const [ i, j ] = (argMaxInds as Vector).data;
    mat.set(i, j, (grad as Scalar).data);
  } else {
    const inds = (argMaxInds as Matrix).data;
    inds.forEach((v: [number, number]) => {
      const val = by === ByAxis.ByRow ? (grad as Vector).get(v[0]) : (grad as Vector).get(v[1]);
      mat.set(v[0], v[1], val);
    });
  }
  return mat;
};

export const max2d = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  return trackedImplement2dReduce(max2dForward, max2dBackward, x, by);
};
