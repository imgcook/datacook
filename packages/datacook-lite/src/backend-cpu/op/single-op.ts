import { Matrix, Vector } from "../classes";
import {
  basicImplement2dSingle,
  basicImplement1dSingle
} from "./basic-impl";


const squareFunc = (a: number): number => Math.pow(a, 2);
const negFunc = (a: number): number => -a;
const sigmoidFunc = (a: number): number => 1 / (1 + Math.exp(-a));

export const exp2d = (x: Matrix): Matrix => {
  return basicImplement2dSingle(Math.exp, x);
};

export const log2d = (x: Matrix): Matrix => {
  return basicImplement2dSingle(Math.log, x);
};

export const abs2d = (x: Matrix): Matrix => {
  return basicImplement2dSingle(Math.abs, x);
};

export const neg2d = (x: Matrix): Matrix => {
  return basicImplement2dSingle(negFunc, x);
};

export const square2d = (x: Matrix): Matrix => {
  return basicImplement2dSingle(squareFunc, x);
};

export const sqrt2d = (x: Matrix): Matrix => {
  return basicImplement2dSingle(Math.sqrt, x);
};

export const pow2d = (x: Matrix, y: number): Matrix => {
  const func = (a: number): number => {
    return Math.pow(a, y);
  };
  return basicImplement2dSingle(func, x);
};

export const sigmoid2d = (x: Matrix): Matrix => {
  return basicImplement2dSingle(sigmoidFunc, x);
};

export const sqrt1d = (x: Vector): Vector => {
  return basicImplement1dSingle(Math.sqrt, x);
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

export const square1d = (x: Vector): Vector => {
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
