import { Scalar, vector } from "../../core/classes";
import { Matrix, Vector } from "../classes";
import {
  basicImplement2dReduce,
  basicImplement2dReduceAll,
  basicImplement1dReduce,
  trackedImplement1dReduce,
  ByAxis
} from "./basic-impl";

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

// export const sum1d = (x: Vector): Scalar => basicImplement1dReduce(sumFunc, x);

export const sum1dForward = (x: Vector): Scalar => basicImplement1dReduce(sumFunc, x);
export const sum1dBackward = (grad: Scalar, x: Vector): Vector => vector(new Array(x.length).fill(grad.data));

export const sum1d = (x: Vector): Scalar => {
  return trackedImplement1dReduce(sum1dForward, sum1dBackward, x);
};

export const mean1d = (x: Vector): Scalar => basicImplement1dReduce(meanFunc, x);

export const min1d = (x: Vector): Scalar => basicImplement1dReduce(minFunc, x);

export const max1d = (x: Vector): Scalar => basicImplement1dReduce(maxFunc, x);

export const argMax1d = (x: Vector): Scalar => basicImplement1dReduce(argMaxFunc, x);

export const argMin1d = (x: Vector): Scalar => basicImplement1dReduce(argMinFunc, x);


export const sum2d = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === undefined) {
    return basicImplement2dReduceAll(sumFunc, x);
  }
  return basicImplement2dReduce(sumFunc, x, by);
};

export const sum2dForward = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === undefined) {
    return basicImplement2dReduceAll(sumFunc, x);
  }
  return basicImplement2dReduce(sumFunc, x, by);
};

export const mean2d = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === undefined) {
    return basicImplement2dReduceAll(meanFunc, x);
  }
  return basicImplement2dReduce(meanFunc, x, by);
};

export const min2d = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === undefined) {
    return basicImplement2dReduceAll(minFunc, x);
  }
  return basicImplement2dReduce(minFunc, x, by);
};

export const max2d = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === undefined) {
    return basicImplement2dReduceAll(maxFunc, x);
  }
  return basicImplement2dReduce(maxFunc, x, by);
};

export const argMin2d = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === undefined) {
    return basicImplement2dReduceAll(argMinFunc, x);
  }
  return basicImplement2dReduce(argMinFunc, x, by);
};

export const argMax2d = (x: Matrix, by?: ByAxis): Vector | Scalar => {
  if (by === undefined) {
    return basicImplement2dReduceAll(argMaxFunc, x);
  }
  return basicImplement2dReduce(argMaxFunc, x, by);
};
