import { Matrix, Vector } from "../classes";
import { Scalar } from "../classes/scalar";
import { basicImplement2dBinary, ByAxis, basicImplement1dBinary, trackedImplement2dBinary, trackedImplement1dBinary } from "./basic-impl";
import { neg1dForward, neg2dForward, square0dFroward, square1dForward, square2dForward } from "./single-op";


const addFunc = (a: number, b: number): number => {
  return a + b;
};

const subFunc = (a: number, b: number): number => {
  return a - b;
};

const mulFunc = (a: number, b: number): number => {
  return a * b;
};

const divFunc = (a: number, b: number): number => {
  return a / b;
};


export const add2dForward = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(addFunc, x, y, by);
};

export const add2dBackwardX = (grad: Matrix): Matrix => {
  return grad;
};

export const add2dBackwardY = (grad: Matrix): Matrix => {
  return grad;
};

export const add2d = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return trackedImplement2dBinary(add2dForward, add2dBackwardX, add2dBackwardY, x, y, by);
};

export const sub2dForward = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(subFunc, x, y, by);
};

export const sub2dBackwardX = (grad: Matrix): Matrix => {
  return grad;
};

export const sub2dBackwardY = (grad: Matrix): Matrix => {
  return grad;
};

export const sub2d = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return trackedImplement2dBinary(sub2dForward, sub2dBackwardX, sub2dBackwardY, x, y, by);
};

export const mul2dForward = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(mulFunc, x, y, by);
};

export const mul2dBackwardX = (grad: Matrix, x: Matrix, y: Matrix | Vector | Scalar | number): Matrix => {
  return mul2dForward(grad, y);
};

export const mul2dBackwardY = (grad: Matrix, x: Matrix): Matrix => {
  return mul2dForward(grad, x);
};

export const mul2d = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return trackedImplement2dBinary(mul2dForward, mul2dBackwardX, mul2dBackwardY, x, y, by);
};

export const div2dForward = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(divFunc, x, y, by);
};

export const div2dBackwardX = (grad: Matrix, x: Matrix, y: Matrix | Vector | Scalar | number): Matrix => {
  let ySquare;
  if (y instanceof Matrix) ySquare = square2dForward(y);
  if (y instanceof Vector) ySquare = square1dForward(y);
  if (y instanceof Scalar) ySquare = square0dFroward(y);
  return neg2dForward(div2dForward(mul2dForward(grad, x), ySquare));
};

export const div2dBackwardY = (grad: Matrix, x: Matrix, y: Matrix | Vector | Scalar | number): Matrix => {
  let ySquare;
  if (y instanceof Matrix) ySquare = square2dForward(y);
  if (y instanceof Vector) ySquare = square1dForward(y);
  if (y instanceof Scalar) ySquare = square0dFroward(y);
  return neg2dForward(div2dForward(mul2dForward(grad, x), ySquare));
};

export const div2d = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return trackedImplement2dBinary(div2dForward, div2dBackwardX, div2dBackwardY, x, y, by);
};

export const add1dForward = (x: Vector, y: Vector | Scalar | number): Vector => {
  return basicImplement1dBinary(addFunc, x, y);
};

export const add1dBackward = (grad: Vector): Vector => {
  return grad;
};

export const add1d = (x: Vector, y: Vector | Scalar | number): Vector => {
  return trackedImplement1dBinary(add1dForward, add1dBackward, add1dBackward, x, y);
};

export const sub1dForward = (x: Vector, y: Vector | Scalar | number): Vector => {
  return basicImplement1dBinary(subFunc, x, y);
};

export const sub1dBackward = (grad: Vector): Vector => {
  return grad;
};
export const sub1d = (x: Vector, y: Vector | Scalar | number): Vector => {
  return trackedImplement1dBinary(sub1dForward, sub1dBackward, sub1dBackward, x, y);
};

export const mul1dForward = (x: Vector, y: Vector | Scalar | number): Vector => {
  return basicImplement1dBinary(mulFunc, x, y);
};

export const mul1dBackwardX = (grad:Vector, x: Vector, y: Vector | Scalar | number): Vector => {
  return mul1dForward(grad, y);
};

export const mul1dBackwardY = (grad:Vector, x: Vector, y: Vector | Scalar | number): Vector => {
  return mul1dForward(grad, y);
};

export const mul1d = (x: Vector, y: Vector | Scalar | number): Vector => {
  return trackedImplement1dBinary(mul1dForward, mul1dBackwardX, mul1dBackwardY, x, y);
};

export const div1dForward = (x: Vector, y: Vector | Scalar | number): Vector => {
  return basicImplement1dBinary(divFunc, x, y);
};

export const div1dBackwardX = (grad:Vector, x: Vector, y: Vector | Scalar | number): Vector => {
  return div1dForward(grad, y);
};

export const div1dBackwardY = (grad:Vector, x: Vector, y: Vector | Scalar | number): Vector => {
  let ySquare;
  if (y instanceof Vector) ySquare = square1dForward(y);
  if (y instanceof Scalar) ySquare = square0dFroward(y);
  return neg1dForward(div1dForward(mul1dForward(grad, x), ySquare));
};

export const div1d = (x: Vector, y: Vector | Scalar | number): Vector => {
  return trackedImplement1dBinary(div1dForward, div1dBackwardX, div1dBackwardY, x, y);
};
