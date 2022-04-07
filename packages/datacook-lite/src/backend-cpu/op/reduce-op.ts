import { Matrix, Vector } from "../classes";
import {
  basicImplement2dReduce,
  basicImplement2dReduceAll,
  ByAxis
} from "./basic-impl";


export const sum2d = (x: Matrix, by?: ByAxis): Vector | number => {
  const func = (a: number[]): number => {
    return a.reduce((p, q) => p + q);
  };
  if (by === undefined) {
    return basicImplement2dReduceAll(func, x);
  }
  return basicImplement2dReduce(func, x, by);
};

export const mean2d = (x: Matrix, by?: ByAxis): Vector | number => {
  const func = (a: number[]): number => {
    return a.reduce((p, q) => p + q) / a.length;
  };
  if (by === undefined) {
    return basicImplement2dReduceAll(func, x);
  }
  return basicImplement2dReduce(func, x, by);
};

export const min2d = (x: Matrix, by?: ByAxis): Vector | number => {
  const func = (a: number[]): number => {
    let min = Number.MAX_SAFE_INTEGER;
    for (let i = 0; i < a.length; i++) {
      if (a[i] < min) {
        min = a[i];
      }
    }
    return min;
  };
  if (by === undefined) {
    return basicImplement2dReduceAll(func, x);
  }
  return basicImplement2dReduce(func, x, by);
};

export const max2d = (x: Matrix, by?: ByAxis): Vector | number => {
  const func = (a: number[]): number => {
    let max = Number.MIN_SAFE_INTEGER;
    for (let i = 0; i < a.length; i++) {
      if (a[i] > max) {
        max = a[i];
      }
    }
    return max;
  };
  if (by === undefined) {
    return basicImplement2dReduceAll(func, x);
  }
  return basicImplement2dReduce(func, x, by);
};

export const argMin2d = (x: Matrix, by?: ByAxis): Vector | number => {
  const func = (a: number[]): number => {
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
  if (by === undefined) {
    return basicImplement2dReduceAll(func, x);
  }
  return basicImplement2dReduce(func, x, by);
};

export const argMax2d = (x: Matrix, by?: ByAxis): Vector | number => {
  const func = (a: number[]): number => {
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
  if (by === undefined) {
    return basicImplement2dReduceAll(func, x);
  }
  return basicImplement2dReduce(func, x, by);
};
