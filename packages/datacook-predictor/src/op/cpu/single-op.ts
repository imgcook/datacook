import { Matrix } from "../../classes";
import {
  basicImplement2dSingle
} from "./basic-impl";


export const exp2d = (x: Matrix): Matrix => {
  const func = (a: number): number => {
    return Math.exp(a);
  };
  return basicImplement2dSingle(func, x);
};

export const log2d = (x: Matrix): Matrix => {
  const func = (a: number): number => {
    return Math.log(a);
  };
  return basicImplement2dSingle(func, x);
};

export const abs2d = (x: Matrix): Matrix => {
  const func = (a: number): number => {
    return Math.abs(a);
  };
  return basicImplement2dSingle(func, x);
};

export const neg2d = (x: Matrix): Matrix => {
  const func = (a: number): number => {
    return -a;
  };
  return basicImplement2dSingle(func, x);
};

export const square2d = (x: Matrix): Matrix => {
  const func = (a: number): number => {
    return Math.pow(a, 2);
  };
  return basicImplement2dSingle(func, x);
};

export const sqrt2d = (x: Matrix): Matrix => {
  const func = (a: number): number => {
    return Math.sqrt(a);
  };
  return basicImplement2dSingle(func, x);
};

export const pow2d = (x: Matrix, y: number): Matrix => {
  const func = (a: number): number => {
    return Math.pow(a, y);
  };
  return basicImplement2dSingle(func, x);
};
