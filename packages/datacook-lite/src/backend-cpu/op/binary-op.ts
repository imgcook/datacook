import { Matrix, Vector } from "../classes";
import { Scalar } from "../classes/scalar";
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

export const add2d = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(addFunc, x, y, by);
};

export const sub2d = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(subFunc, x, y, by);
};

export const mul2d = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(mulFunc, x, y, by);
};

export const div2d = (x: Matrix, y: Matrix | Vector | Scalar | number, by: ByAxis = 0): Matrix => {
  return basicImplement2dBinary(divFunc, x, y, by);
};

export const add1d = (x: Vector, y: Vector | Scalar | number): Vector => {
  return basicImplement1dBinary(addFunc, x, y);
};

export const sub1d = (x: Vector, y: Vector | Scalar | number): Vector => {
  return basicImplement1dBinary(subFunc, x, y);
};

export const mul1d = (x: Vector, y: Vector | Scalar | number): Vector => {
  return basicImplement1dBinary(mulFunc, x, y);
};

export const div1d = (x: Vector, y: Vector | Scalar | number): Vector => {
  return basicImplement1dBinary(divFunc, x, y);
};
