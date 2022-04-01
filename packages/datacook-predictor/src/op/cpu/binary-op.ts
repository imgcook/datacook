import { Matrix, Vector } from "../../classes";
import { basicImplement2dBinary, ByAxis } from "./basic-impl";

export const add2d = (x: Matrix, y: Matrix | Vector | number, by: ByAxis = 0): Matrix => {
  const func = (a: number, b: number): number => {
    return a + b;
  };
  return basicImplement2dBinary(func, x, y, by);
};

export const sub2d = (x: Matrix, y: Matrix | Vector | number, by: ByAxis = 0): Matrix => {
  const func = (a: number, b: number): number => {
    return a - b;
  };
  return basicImplement2dBinary(func, x, y, by);
};

export const mul2d = (x: Matrix, y: Matrix | Vector | number, by: ByAxis = 0): Matrix => {
  const func = (a: number, b: number): number => {
    return a * b;
  };
  return basicImplement2dBinary(func, x, y, by);
};

export const div2d = (x: Matrix, y: Matrix | Vector | number, by: ByAxis = 0): Matrix => {
  const func = (a: number, b: number): number => {
    return a / b;
  };
  return basicImplement2dBinary(func, x, y, by);
};
