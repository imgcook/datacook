import { Matrix, Vector } from "../../classes";
import { basicImplement2d, ByAxis } from "./basic-impl";

export const add2d = (x: Matrix, y: Matrix | Vector | number, by: ByAxis = 0): Matrix => {
  const addFunc = (a: number, b: number): number => {
    return a + b;
  };
  return basicImplement2d(addFunc, x, y, by);
};
