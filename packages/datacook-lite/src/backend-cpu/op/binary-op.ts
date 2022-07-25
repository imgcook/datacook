import { Matrix, Vector } from "../classes";
import { basicImplement2dBinary, ByAxis, basicImplement1dBinary } from "./basic-impl";


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

export const add2d = (x: Matrix, y: Matrix | Vector | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(addFunc, x, y, by);
};

export const sub2d = (x: Matrix, y: Matrix | Vector | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(subFunc, x, y, by);
};

export const mul2d = (x: Matrix, y: Matrix | Vector | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(mulFunc, x, y, by);
};

export const div2d = (x: Matrix, y: Matrix | Vector | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(divFunc, x, y, by);
};

export const add1d = (x: Vector, y: Vector | number): Vector => {
  return basicImplement1dBinary(addFunc, x, y);
};

export const sub1d = (x: Vector, y: Vector | number): Vector => {
  return basicImplement1dBinary(subFunc, x, y);
};

export const mul1d = (x: Vector, y: Vector | number): Vector => {
  return basicImplement1dBinary(mulFunc, x, y);
};

export const div1d = (x: Vector, y: Vector | number): Vector => {
  return basicImplement1dBinary(divFunc, x, y);
};
<<<<<<< HEAD
=======

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

// export const add0d = (x: Scalar, y: Scalar | number): Scalar => {
//   return tracked
// };
>>>>>>> cf92074... optimizer
