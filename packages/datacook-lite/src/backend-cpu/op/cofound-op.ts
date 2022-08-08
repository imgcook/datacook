import { Matrix } from "../classes";
import { exp2d } from "./single-op";
import { sum2d } from "./reduce-op";
import { div2d } from "./binary-op";
import { ByAxis } from "./basic-impl";

export const softmax2d = (x: Matrix, by: ByAxis = ByAxis.ByRow): Matrix => {
  const expBase = exp2d(x);
  const nominator = sum2d(expBase, by);
  return div2d(expBase, nominator);
};
