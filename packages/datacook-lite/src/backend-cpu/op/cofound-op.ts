import { Matrix, Vector } from "../classes";
import { exp2d } from "./single-op";
import { sum1d, sum2d } from "./reduce-op";
import { div2d, mul1d } from "./binary-op";
import { ByAxis } from "./basic-impl";
import { Scalar } from "../../core/classes";

export const softmax2d = (x: Matrix, by: ByAxis = ByAxis.ByRow): Matrix => {
  const expBase = exp2d(x);
  const nominator = sum2d(expBase, by);
  return div2d(expBase, nominator);
};

export const dot1d = (x: Vector, y: Vector): Scalar => {
  return sum1d(mul1d(x, y));
};
