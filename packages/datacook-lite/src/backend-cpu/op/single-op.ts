import { Scalar } from "../../core/classes";
import { scalar } from "../../core/classes/creation";
import { Matrix, Vector } from "../classes";
import { div2dForward, mul1d, mul2dForward } from "./binary-op";
import {
  basicImplement2dSingle,
  basicImplement1dSingle,
  trackedImplement2dSingle,
  trackedImplement1dSingle
} from "./basic-impl";


const squareFunc = (a: number): number => Math.pow(a, 2);
const negFunc = (a: number): number => -a;
const sigmoidFunc = (a: number): number => 1 / (1 + Math.exp(-a));
const logGradFunc = (x: number) => 1 / x;
const absGradFunc = (x: number) => x > 0 ? 1 : -1;
const negGradFunc = () => -1;
const squareGradFunc = (x: number): number => 2 * x;
const powGeadFunc = (x: number, y: number): number => y * Math.pow(x, y - 1);
const sigmoidGradFunc = (x: number): number => sigmoidFunc(x) * (1 - sigmoidFunc(x));
const sqrtGradFunc = (x: number): number => 1 / (2 * Math.sqrt(x));

export const exp2dForward = (x: Matrix): Matrix => {
  return basicImplement2dSingle(Math.exp, x);
};

export const exp2dBackward = (grad: Matrix, x: Matrix): Matrix => {
  return mul2dForward(grad, exp2dForward(x));
};

export const exp2d = (x: Matrix): Matrix => {
  return trackedImplement2dSingle(exp2dForward, exp2dBackward, x);
};

export const log2dFoward = (x: Matrix): Matrix => {
  return basicImplement2dSingle(Math.log, x);
};
export const log2dBackward = (grad: Matrix, x: Matrix): Matrix => {
  return mul2dForward(grad, basicImplement2dSingle(logGradFunc, x));
};

export const log2d = (x: Matrix): Matrix => {
  return trackedImplement2dSingle(log2dFoward, log2dBackward, x);
};

export const abs2dBackward = (grad: Matrix, x: Matrix): Matrix => {
  return mul2dForward(grad, basicImplement2dSingle(absGradFunc, x));
};

export const abs2dForward = (x: Matrix): Matrix => {
  return basicImplement2dSingle(Math.abs, x);
};

export const abs2d = (x: Matrix): Matrix => {
  return trackedImplement2dSingle(abs2dForward, abs2dBackward, x);
};

export const neg2dForward = (x: Matrix): Matrix => {
  return basicImplement2dSingle(negFunc, x);
};

export const neg2dBackward = (grad: Matrix, x: Matrix): Matrix => {
  return mul2dForward(grad, basicImplement2dSingle(negGradFunc, x));
};

export const neg2d = (x: Matrix): Matrix => {
  return trackedImplement2dSingle(neg2dForward, neg2dBackward, x);
};

export const square2dForward = (x: Matrix): Matrix => {
  return basicImplement2dSingle(squareFunc, x);
};

export const square2dBackward = (grad: Matrix, x: Matrix): Matrix => {
  return mul2dForward(grad, basicImplement2dSingle(squareGradFunc, x));
};

export const square2d = (x: Matrix): Matrix => {
  return trackedImplement2dSingle(square2dForward, square2dBackward, x);
};

export const sqrt2dForward = (x: Matrix): Matrix => {
  return basicImplement2dSingle(Math.sqrt, x);
};


export const sqrt2dBackward = (grad: Matrix, x: Matrix): Matrix => {
  return mul2dForward(grad, basicImplement2dSingle(sqrtGradFunc, x));
};

export const sqrt2d = (x: Matrix): Matrix => {
  return trackedImplement2dSingle(sqrt2dForward, sqrt2dBackward, x);
};

export const pow2dForward = (x: Matrix, y: number): Matrix => {
  return basicImplement2dSingle(Math.pow, x, y);
};

export const pow2dBackward = (grad: Matrix, x: Matrix, y: number): Matrix => {
  return mul2dForward(grad, basicImplement2dSingle(powGeadFunc, x, y));
};

export const pow2d = (x: Matrix, y: number): Matrix => {
  return trackedImplement2dSingle(pow2dForward, pow2dBackward, x, y);
};

export const sigmoid2dForward = (x: Matrix): Matrix => {
  return basicImplement2dSingle(sigmoidFunc, x);
};

export const sigmoid2dBackward = (grad: Matrix, x: Matrix): Matrix => {
  return mul2dForward(grad, basicImplement2dSingle(sigmoidGradFunc, x));
};

export const sigmoid2d = (x: Matrix): Matrix => {
  return trackedImplement2dSingle(sigmoid2dForward, sigmoid2dBackward, x);
};

export const sqrt1dForward = (x: Vector): Vector => {
  return basicImplement1dSingle(Math.sqrt, x);
};

export const sqrt1dBackcward = (grad: Vector, x: Vector): Vector => {
  return mul1d(grad, basicImplement1dSingle(sqrtGradFunc, x));
};

export const sqrt1d = (x: Vector): Vector => {
  return trackedImplement1dSingle(sqrt1dForward, sqrt1dBackcward, x);
};

export const exp1d = (x: Vector): Vector => {
  return basicImplement1dSingle(Math.exp, x);
};

export const log1d = (x: Vector): Vector => {
  return basicImplement1dSingle(Math.log, x);
};

export const abs1d = (x: Vector): Vector => {
  return basicImplement1dSingle(Math.abs, x);
};

export const neg1d = (x: Vector): Vector => {
  return basicImplement1dSingle(negFunc, x);
};

export const neg1dForward = (x: Vector): Vector => {
  return basicImplement1dSingle(negFunc, x);
};


export const square1d = (x: Vector): Vector => {
  const func = (a: number): number => {
    return Math.pow(a, 2);
  };
  return basicImplement1dSingle(func, x);
};

export const square1dForward = (x: Vector): Vector => {
  const func = (a: number): number => {
    return Math.pow(a, 2);
  };
  return basicImplement1dSingle(func, x);
};

export const pow1d = (x: Vector, y: number): Vector => {
  const func = (a: number): number => {
    return Math.pow(a, y);
  };
  return basicImplement1dSingle(func, x);
};

export const sigmoid1d = (x: Vector): Vector => {
  return basicImplement1dSingle(sigmoidFunc, x);
};

export const square0dFroward = (x: Scalar): Scalar => {
  return scalar(Math.pow(x.data, 2));
};
